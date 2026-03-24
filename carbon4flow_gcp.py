"""
carbon4flow_gcp.py
──────────────────
Schema real confirmado:

  index.parquet
    resource_id  → str  ex: "5295"
    path         → str  ex: "projetos/5295.parquet"

  projetos/{id}.parquet
    geometry     → GeoPandas nativa (MultiPolygon Z / Polygon Z)
    resource_id  → str  ex: "4849"
    project_name → str  ex: "Renove_ALM_Brazil"  ← nome do KML, NÃO usar para match

  df_all (Google Drive)
    resourceIdentifier → int64  ex: 5889
    resourceName_x     → str    ex: "Cabeceira do Xingu REDD Project"

REGRA DE JOIN: sempre por ID numérico.
  index.resource_id (str) ↔ df_all.resourceIdentifier (int64)
  → normaliza tudo para str antes de comparar.
"""

import io
import logging
from typing import Optional, Tuple, List

import pandas as pd
import geopandas as gpd
import streamlit as st
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go
from shapely.geometry import mapping
from shapely.ops import unary_union, transform
from google.cloud import storage
from google.oauth2 import service_account

log = logging.getLogger(__name__)

GCS_BUCKET = "edriano-verra-projects"
GCS_INDEX  = "index.parquet"


# ═══════════════════════════════════════════════════════════════════════
# AUTENTICAÇÃO
# ═══════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def _gcs_client() -> storage.Client:
    info  = dict(st.secrets["gcp_service_account"])
    creds = service_account.Credentials.from_service_account_info(
        info,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    return storage.Client(credentials=creds, project=creds.project_id)


def _blob_bytes(blob_path: str) -> bytes:
    return _gcs_client().bucket(GCS_BUCKET).blob(blob_path).download_as_bytes()


# ═══════════════════════════════════════════════════════════════════════
# INDEX
# ═══════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def load_index() -> pd.DataFrame:
    """
    Carrega index.parquet.
    Garante que resource_id é sempre str para joins consistentes.
    """
    try:
        df = pd.read_parquet(io.BytesIO(_blob_bytes(GCS_INDEX)))
        df["resource_id"] = df["resource_id"].astype(str).str.strip()
        return df
    except Exception as e:
        st.error(f"❌ Erro ao carregar index.parquet: {e}")
        return pd.DataFrame(columns=["resource_id", "path"])


# ═══════════════════════════════════════════════════════════════════════
# GEOMETRIA POR PROJETO
# ═══════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=1800, show_spinner=False)
def _raw_parquet(blob_path: str) -> Optional[bytes]:
    try:
        return _blob_bytes(blob_path)
    except Exception as e:
        log.warning(f"Blob não encontrado: {blob_path} — {e}")
        return None


def load_project_geometry(resource_id: str) -> Optional[gpd.GeoDataFrame]:
    """
    Baixa projetos/{id}.parquet pelo resource_id (str).
    Remove coordenada Z automaticamente.
    """
    idx = load_index()
    rid = str(resource_id).strip()
    row = idx[idx["resource_id"] == rid]

    if row.empty:
        log.warning(f"resource_id '{rid}' não encontrado no index.")
        return None

    raw = _raw_parquet(row.iloc[0]["path"])
    if raw is None:
        return None

    try:
        gdf = gpd.read_parquet(io.BytesIO(raw))

        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        elif gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs("EPSG:4326")

        # Remove Z (Polygon Z → Polygon 2D)
        gdf["geometry"] = gdf["geometry"].apply(
            lambda g: transform(lambda x, y, z=None: (x, y), g)
            if g is not None and g.has_z else g
        )
        gdf["geometry"] = gdf["geometry"].buffer(0)
        gdf = gdf[
            gdf["geometry"].notnull() &
            ~gdf["geometry"].is_empty &
            gdf["geometry"].is_valid
        ]
        return gdf if not gdf.empty else None

    except Exception as e:
        log.error(f"Erro ao ler parquet do projeto {rid}: {e}")
        return None


def get_dissolved_geometry(resource_id: str):
    gdf = load_project_geometry(resource_id)
    if gdf is None or gdf.empty:
        return None
    return unary_union(gdf.geometry)


def get_aoi_geojson(resource_id: str) -> Optional[dict]:
    geom = get_dissolved_geometry(resource_id)
    return mapping(geom) if geom else None


def get_aoi_bbox_str(resource_id: str) -> Optional[str]:
    geom = get_dissolved_geometry(resource_id)
    if geom is None:
        return None
    b = geom.bounds
    return f"{b[0]},{b[1]},{b[2]},{b[3]}"


# ═══════════════════════════════════════════════════════════════════════
# ENRIQUECE df_all PARA USO INTERNO
# Normaliza resourceIdentifier → str para joins com GCS
# ═══════════════════════════════════════════════════════════════════════

def _prep_df_all(df_all: pd.DataFrame) -> pd.DataFrame:
    """
    Retorna cópia de df_all com resourceIdentifier como str,
    pronto para join com index.resource_id (str).
    """
    df = df_all.copy()
    df["resourceIdentifier"] = df["resourceIdentifier"].astype(str).str.strip()
    return df


# ═══════════════════════════════════════════════════════════════════════
# CARREGAR TODAS AS GEOMETRIAS (Visão Geral)
# ═══════════════════════════════════════════════════════════════════════

def carregar_geometrias_gcp(df_all: pd.DataFrame) -> Tuple[gpd.GeoDataFrame, List[str]]:
    """
    Carrega geometrias de todos os projetos disponíveis no GCS
    que também existem em df_all (join por ID numérico como str).
    """
    df = _prep_df_all(df_all)
    idx = load_index()

    if idx.empty:
        return gpd.GeoDataFrame(), ["index.parquet vazio ou inacessível"]

    ids_gcs = set(idx["resource_id"].tolist())
    ids_df  = set(df["resourceIdentifier"].tolist())
    ids     = sorted(ids_gcs & ids_df)

    if not ids:
        return gpd.GeoDataFrame(), [
            f"Nenhum ID em comum. GCS tem {len(ids_gcs)} IDs, df_all tem {len(ids_df)} IDs. "
            f"Exemplo GCS: {list(ids_gcs)[:3]} | Exemplo df_all: {list(ids_df)[:3]}"
        ]

    gdfs  = []
    erros = []
    total = len(ids)
    prog  = st.progress(0, text=f"Carregando geometrias (0/{total})...")

    for i, rid in enumerate(ids):
        blob_path = idx[idx["resource_id"] == rid].iloc[0]["path"]
        raw = _raw_parquet(blob_path)
        prog.progress((i + 1) / total, text=f"Carregando geometrias ({i+1}/{total})...")

        if raw is None:
            erros.append(f"{rid}: blob não encontrado")
            continue

        try:
            gdf = gpd.read_parquet(io.BytesIO(raw))
            if gdf.crs is None:
                gdf = gdf.set_crs("EPSG:4326")
            elif gdf.crs.to_epsg() != 4326:
                gdf = gdf.to_crs("EPSG:4326")

            gdf["geometry"] = gdf["geometry"].apply(
                lambda g: transform(lambda x, y, z=None: (x, y), g)
                if g is not None and g.has_z else g
            )
            gdf["geometry"] = gdf["geometry"].buffer(0)
            gdf = gdf[
                gdf["geometry"].notnull() &
                ~gdf["geometry"].is_empty &
                gdf["geometry"].is_valid
            ]
            if gdf.empty:
                erros.append(f"{rid}: geometria vazia após sanitização")
                continue

            gdf["resourceIdentifier"] = rid   # str
            gdfs.append(gdf)

        except Exception as e:
            erros.append(f"{rid}: {e}")

    prog.empty()

    if not gdfs:
        return gpd.GeoDataFrame(), erros

    gdf_combined = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs="EPSG:4326")

    # Mescla metadados de df_all por ID (str ↔ str)
    meta_cols = [c for c in [
        "resourceIdentifier", "resourceName_x", "state_Recode",
        "vcsAFOLUActivity", "vcsProjectStatus", "vcsMethodology",
        "vcsCreditingPeriodTerm", "vcsAcresHectares", "description",
    ] if c in df.columns]

    df_ref = df[meta_cols].drop_duplicates("resourceIdentifier")

    extra = [c for c in meta_cols if c != "resourceIdentifier" and c in gdf_combined.columns]
    gdf_combined = gdf_combined.drop(columns=extra, errors="ignore")
    gdf_combined = gdf_combined.merge(df_ref, on="resourceIdentifier", how="left")
    gdf_combined = gpd.GeoDataFrame(gdf_combined, geometry="geometry", crs="EPSG:4326")

    return gdf_combined, erros


# ═══════════════════════════════════════════════════════════════════════
# RENDER DA ABA AOI
# ═══════════════════════════════════════════════════════════════════════

def render_aoi_tab(
    df_all: pd.DataFrame,
    GFW_API_KEY: str,
    gfw_tree_cover_loss,
    gfw_glad_alerts,
    gfw_radd_alerts,
    terrabrasilis_wfs,
    ACTIVITY_COLORS: dict,
) -> None:
    st.markdown("## 📍 Análise Espacial por Projeto")
    st.info(
        "Geometrias carregadas do **Google Cloud Storage** (Parquet). "
        "Projeto individual usa a **AOI real** para consultas GFW e WFS."
    )

    # ── Catálogo ──────────────────────────────────────────────────────
    with st.spinner("📋 Carregando catálogo..."):
        idx = load_index()   # resource_id = str

    if idx.empty:
        st.error("❌ Não foi possível carregar o index.parquet do GCS.")
        return

    # df_all com resourceIdentifier como str
    df = _prep_df_all(df_all)

    # Metadados para enriquecer o catálogo
    meta_cols = [c for c in [
        "resourceName_x", "state_Recode", "vcsAFOLUActivity",
        "vcsProjectStatus", "vcsMethodology", "vcsCreditingPeriodTerm",
        "vcsAcresHectares", "description",
    ] if c in df.columns]

    # Join: index.resource_id (str) ↔ df.resourceIdentifier (str)
    df_meta = (
        df[["resourceIdentifier"] + meta_cols]
        .drop_duplicates("resourceIdentifier")
        .rename(columns={"resourceIdentifier": "resource_id"})
    )
    idx = idx.merge(df_meta, on="resource_id", how="left")

    # ── Seletor ───────────────────────────────────────────────────────
    def _label(row) -> str:
        nome   = row.get("resourceName_x") or row["resource_id"]
        estado = row.get("state_Recode")   or "N/A"
        return f"{nome} — {estado}"

    options = ["🌎 Visão Geral (Todos os Projetos)"] + [
        _label(r) for _, r in idx.iterrows()
    ]

    selected    = st.selectbox("📍 Selecione um projeto:", options, key="gcp_project_selector")
    is_overview = selected == "🌎 Visão Geral (Todos os Projetos)"

    # ── Geometria ─────────────────────────────────────────────────────
    if is_overview:
        with st.spinner("🗺️ Carregando geometrias do GCS..."):
            gdf_plot, erros = carregar_geometrias_gcp(df_all)

        if erros:
            with st.expander(f"⚠️ {len(erros)} problema(s)"):
                for e in erros:
                    st.caption(e)

        if gdf_plot.empty:
            st.warning("Nenhuma geometria disponível.")
            return

        centroid     = gdf_plot.geometry.centroid
        center       = [centroid.y.mean(), centroid.x.mean()]
        zoom_start   = 5
        selected_gdf = gdf_plot
        selected_rid = None
        bbox_str     = None
        aoi_geojson  = None

    else:
        # Extrai resource_id a partir do label selecionado
        # Label = "resourceName_x — state_Recode"
        # Busca no idx pelo resourceName_x
        label_nome = selected.split(" — ")[0].strip()
        row_idx = idx[idx["resourceName_x"] == label_nome]

        # Fallback: busca pelo resource_id caso o nome coincida
        if row_idx.empty:
            row_idx = idx[idx["resource_id"] == label_nome]

        if row_idx.empty:
            st.warning("Projeto não encontrado no catálogo.")
            return

        selected_rid = str(row_idx.iloc[0]["resource_id"]).strip()

        with st.spinner(f"🗺️ Carregando geometria..."):
            gdf_single = load_project_geometry(selected_rid)

        if gdf_single is None or gdf_single.empty:
            st.warning(
                f"⚠️ Geometria não disponível no GCS para o projeto ID `{selected_rid}`. "
                "Verifique se o arquivo `projetos/{selected_rid}.parquet` existe no bucket."
            )
            return

        # Injeta metadados
        for col in meta_cols:
            if col not in gdf_single.columns:
                gdf_single[col] = row_idx.iloc[0].get(col, "N/A")
        gdf_single["resourceIdentifier"] = selected_rid

        bounds       = gdf_single.total_bounds
        center       = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]
        zoom_start   = 10
        selected_gdf = gdf_single
        bbox_str     = get_aoi_bbox_str(selected_rid)
        aoi_geojson  = get_aoi_geojson(selected_rid)

    # ── Helpers ───────────────────────────────────────────────────────
    def _info_panel(gdf):
        row = gdf.iloc[0]
        for label, col in [
            ("Projeto",     "resourceName_x"),
            ("ID",          "resourceIdentifier"),
            ("Estado",      "state_Recode"),
            ("Área",        "vcsAcresHectares"),
            ("Tipo",        "vcsAFOLUActivity"),
            ("Acreditação", "vcsCreditingPeriodTerm"),
            ("Protocolo",   "vcsMethodology"),
            ("Status",      "vcsProjectStatus"),
            ("Resumo",      "description"),
        ]:
            st.markdown(f"**{label}:** {row.get(col, 'N/A')}")

    def _base_map(border_color="#FF0000") -> folium.Map:
        m = folium.Map(location=center, zoom_start=zoom_start, tiles=None)
        folium.TileLayer("Esri.WorldImagery", name="Satélite", control=False).add_to(m)

        if not is_overview:
            b = selected_gdf.total_bounds
            folium.PolyLine(
                [[b[1],b[0]],[b[1],b[2]],[b[3],b[2]],[b[3],b[0]],[b[1],b[0]]],
                color="#00FFFF", weight=2, dash_array="6 4",
                tooltip="Bounding Box — área de consulta WFS",
            ).add_to(m)

        for _, row in selected_gdf.iterrows():
            try:
                folium.GeoJson(
                    data=mapping(row["geometry"]),
                    style_function=lambda x, c=border_color: {
                        "fillColor": "transparent", "color": c,
                        "weight": 3, "fillOpacity": 0.08, "dashArray": "5,5",
                    },
                ).add_to(m)
            except Exception:
                pass

        if not is_overview:
            b = selected_gdf.total_bounds
            m.fit_bounds([[b[1], b[0]], [b[3], b[2]]])

        return m

    def _bar(x, y, color, x_title="Ano", y_title="", height=300):
        fig = go.Figure(go.Bar(x=x, y=y, marker_color=color))
        fig.update_layout(
            height=height, template="plotly_white",
            margin=dict(t=10, b=40, l=40, r=10),
            xaxis_title=x_title, yaxis_title=y_title,
            hovermode="x unified"
        )
        return fig

    # ═════════════════════════════════════════════════════════════════
    # TABS INTERNAS
    # ═════════════════════════════════════════════════════════════════
    tab_gfw, tab_prodes, tab_deter_amz, tab_deter_cer = st.tabs([
        "🌳 GFW", "🔴 PRODES", "🟠 DETER Amazônia", "🟡 DETER Cerrado"
    ])

    # ── GFW ──────────────────────────────────────────────────────────
    with tab_gfw:
        col_mapa, col_info = st.columns([8, 2])

        with col_mapa:
            st.markdown("### 🗺️ Mapa GFW")
            c1, c2, c3 = st.columns(3)
            show_loss = c1.toggle("🔴 Tree Cover Loss", value=True,  key="gcp_loss")
            show_glad = c2.toggle("🟡 GLAD Alerts",     value=False, key="gcp_glad")
            show_radd = c3.toggle("🟠 RADD Alerts",     value=False, key="gcp_radd")

            m = _base_map("#FF0000")
            if show_loss:
                folium.TileLayer(
                    "https://tiles.globalforestwatch.org/umd_tree_cover_loss/v1.11/tcd_30/{z}/{x}/{y}.png",
                    name="Tree Cover Loss", attr="GFW", overlay=True, opacity=0.8,
                ).add_to(m)
            if show_glad:
                folium.TileLayer(
                    "https://tiles.globalforestwatch.org/umd_glad_landsat_alerts/v20260320/default/{z}/{x}/{y}.png",
                    name="GLAD Alerts", attr="GFW", overlay=True, opacity=0.8,
                ).add_to(m)
            if show_radd:
                folium.TileLayer(
                    "https://tiles.globalforestwatch.org/wur_radd_alerts/v20260315/default/{z}/{x}/{y}.png",
                    name="RADD Alerts", attr="GFW", overlay=True, opacity=0.8,
                ).add_to(m)
            folium.LayerControl().add_to(m)
            st_folium(m, width=None, height=600, key="gcp_map_gfw")

        with col_info:
            if is_overview:
                st.info("💡 Selecione um projeto para ver detalhes.")
                st.metric("Projetos no mapa", f"{len(selected_gdf['resourceIdentifier'].unique()):,}")
            else:
                st.markdown("### 📋 Info")
                _info_panel(selected_gdf)
                st.caption("✅ AOI real do GCS" if aoi_geojson else "⚠️ AOI indisponível")

        if not is_overview and aoi_geojson:
            st.divider()
            col_g1, col_g2, col_g3 = st.columns(3)
            with col_g1:
                st.markdown("### 📊 Tree Cover Loss")
                with st.spinner("Consultando GFW..."):
                    df_loss = gfw_tree_cover_loss(aoi_geojson, GFW_API_KEY)
                if df_loss.empty:
                    st.warning("Sem dados para esta AOI.")
                else:
                    st.plotly_chart(_bar(df_loss["umd_tree_cover_loss__year"], df_loss["loss_ha"], "#ff4444", y_title="ha"), use_container_width=True)

            with col_g2:
                st.markdown("### 🟡 GLAD Alerts")
                with st.spinner("Consultando GLAD..."):
                    df_glad = gfw_glad_alerts(aoi_geojson, GFW_API_KEY)
                if df_glad.empty:
                    st.warning("Sem dados para esta AOI.")
                else:
                    st.plotly_chart(_bar(df_glad["alert__year"], df_glad["alert_count"], "#FFC300", y_title="Alertas"), use_container_width=True)

            with col_g3:
                st.markdown("### 🟠 RADD Alerts")
                with st.spinner("Consultando RADD..."):
                    df_radd = gfw_radd_alerts(aoi_geojson, GFW_API_KEY)
                if df_radd.empty:
                    st.warning("Sem dados para esta AOI.")
                else:
                    st.plotly_chart(_bar(df_radd["alert__year"], df_radd["alert_count"], "#FF7900", y_title="Alertas"), use_container_width=True)

        elif not is_overview:
            st.info("ℹ️ AOI não disponível — consultas GFW indisponíveis.")

    # ── PRODES ────────────────────────────────────────────────────────
    with tab_prodes:
        col_mapa_p, col_info_p = st.columns([8, 2])

        with col_mapa_p:
            st.markdown("### 🗺️ Mapa PRODES")
            c1, c2 = st.columns(2)
            show_br  = c1.toggle("🔴 PRODES Brasil",    value=True,  key="gcp_prodes_br")
            show_amz = c2.toggle("🟥 PRODES Legal AMZ", value=False, key="gcp_prodes_amz")

            m = _base_map("#49006a")
            if show_br:
                folium.WmsTileLayer(
                    url="https://terrabrasilis.dpi.inpe.br/geoserver/prodes-brasil-nb/prodes_brasil/ows",
                    layers="prodes_brasil", fmt="image/png", transparent=True,
                    name="PRODES Brasil", overlay=True, opacity=0.8,
                ).add_to(m)
            if show_amz:
                folium.WmsTileLayer(
                    url="https://terrabrasilis.dpi.inpe.br/geoserver/prodes-legal-amz/yearly_deforestation/ows",
                    layers="yearly_deforestation", fmt="image/png", transparent=True,
                    name="PRODES Legal AMZ", overlay=True, opacity=0.8,
                ).add_to(m)
            folium.LayerControl().add_to(m)
            st_folium(m, width=None, height=600, key="gcp_map_prodes")

        with col_info_p:
            if is_overview:
                st.info("💡 Selecione um projeto.")
            else:
                st.markdown("### 📋 Info")
                _info_panel(selected_gdf)

        if not is_overview and bbox_str:
            st.divider()
            with st.spinner("Consultando TerraBrasilis (PRODES AMZ)..."):
                df_prodes = terrabrasilis_wfs(
                    "https://terrabrasilis.dpi.inpe.br/geoserver/prodes-legal-amz/yearly_deforestation/ows",
                    "prodes-legal-amz:yearly_deforestation", bbox_str,
                )

            col_g1, col_g2, col_g3 = st.columns(3)
            with col_g1:
                st.markdown("### 📊 Polígonos por Ano")
                if df_prodes.empty:
                    st.warning("Sem dados PRODES para esta AOI.")
                elif "year" in df_prodes.columns:
                    df_prodes["year"] = pd.to_numeric(df_prodes["year"], errors="coerce")
                    by_year = df_prodes.groupby("year").size().reset_index(name="n")
                    st.plotly_chart(_bar(by_year["year"], by_year["n"], "#C0392B", y_title="Polígonos"), use_container_width=True)

            with col_g2:
                st.markdown("### 📊 Área (km²) por Ano")
                if not df_prodes.empty and "area_km" in df_prodes.columns:
                    df_prodes["area_km"] = pd.to_numeric(df_prodes["area_km"], errors="coerce")
                    by_area = df_prodes.groupby("year").agg(a=("area_km", "sum")).reset_index()
                    st.plotly_chart(_bar(by_area["year"], by_area["a"], "#922B21", y_title="km²"), use_container_width=True)
                elif not df_prodes.empty:
                    st.warning("Coluna `area_km` não encontrada.")

            with col_g3:
                st.markdown("### 📊 Classe")
                if not df_prodes.empty and "class_name" in df_prodes.columns:
                    cls = df_prodes["class_name"].value_counts().reset_index()
                    cls.columns = ["Classe", "n"]
                    st.plotly_chart(_bar(cls["Classe"], cls["n"], "#E74C3C", x_title="Classe", y_title="Polígonos"), use_container_width=True)

            if not df_prodes.empty:
                with st.expander("📋 Tabela PRODES"):
                    st.dataframe(df_prodes, use_container_width=True, height=280)
                    st.download_button("⬇️ CSV", df_prodes.to_csv(index=False).encode("utf-8"), "prodes_aoi.csv", "text/csv")

    # ── DETER AMAZÔNIA ────────────────────────────────────────────────
    with tab_deter_amz:
        col_mapa_d, col_info_d = st.columns([8, 2])

        with col_mapa_d:
            st.markdown("### 🗺️ Mapa DETER Amazônia")
            m = _base_map("#FF6600")
            folium.WmsTileLayer(
                url="https://terrabrasilis.dpi.inpe.br/geoserver/deter-amz/deter_amz/ows",
                layers="deter_amz", fmt="image/png", transparent=True,
                name="DETER AMZ", overlay=True, opacity=0.8,
            ).add_to(m)
            folium.LayerControl().add_to(m)
            st_folium(m, width=None, height=600, key="gcp_map_deter_amz")

        with col_info_d:
            if is_overview:
                st.info("💡 Selecione um projeto.")
            else:
                st.markdown("### 📋 Info")
                _info_panel(selected_gdf)
                st.markdown(
                    "<small><b>Legenda:</b><br>"
                    "🔴 Cicatriz queimada &nbsp;⬜ Corte seletivo<br>"
                    "🟣 CS desordenado &nbsp;🟠 CS geométrico<br>"
                    "🟡 Degradação &nbsp;🟤 Desmatamento CR<br>"
                    "🟢 Desmatamento veg &nbsp;🔵 Mineração</small>",
                    unsafe_allow_html=True,
                )

        if not is_overview and bbox_str:
            st.divider()
            with st.spinner("Consultando TerraBrasilis (DETER AMZ)..."):
                df_deter = terrabrasilis_wfs(
                    "https://terrabrasilis.dpi.inpe.br/geoserver/deter-amz/deter_amz/ows",
                    "deter-amz:deter_amz", bbox_str,
                )

            if df_deter.empty:
                st.warning("Sem alertas DETER para esta AOI.")
            else:
                col_g1, col_g2 = st.columns([2, 3])

                with col_g1:
                    if "view_date" in df_deter.columns:
                        df_deter["year"] = pd.to_datetime(df_deter["view_date"], errors="coerce").dt.year
                        by_year = df_deter.groupby("year").size().reset_index(name="n")
                        st.markdown("### 📊 Alertas por Ano")
                        st.plotly_chart(_bar(by_year["year"], by_year["n"], "#E67E22", y_title="Alertas"), use_container_width=True)

                    if "classname" in df_deter.columns:
                        cls = df_deter["classname"].value_counts().reset_index()
                        cls.columns = ["Classe", "n"]
                        st.markdown("### 📊 Por Classe")
                        st.plotly_chart(_bar(cls["Classe"], cls["n"], "#D35400", x_title="Classe", y_title="Alertas"), use_container_width=True)

                with col_g2:
                    st.markdown("### 📊 Área por Classe ao Longo do Tempo (ha)")
                    DETER_COLORS = {
                        "CICATRIZ_DE_QUEIMADA": "#d7191c", "CORTE_SELETIVO":  "#868686",
                        "CS_DESORDENADO":       "#db83ff", "CS_GEOMETRICO":   "#ff7e00",
                        "DEGRADACAO":           "#e6c300", "DESMATAMENTO_CR": "#8a5f4b",
                        "DESMATAMENTO_VEG":     "#abdda4", "MINERACAO":       "#4223e5",
                    }
                    if all(c in df_deter.columns for c in ["classname", "areamunkm", "year"]):
                        df_deter["areamunkm"] = pd.to_numeric(df_deter["areamunkm"], errors="coerce") * 100
                        df_area = df_deter.groupby(["year", "classname"]).agg(ha=("areamunkm", "sum")).reset_index()
                        fig = go.Figure()
                        for cls_name in sorted(df_area["classname"].unique()):
                            df_c = df_area[df_area["classname"] == cls_name]
                            cor  = DETER_COLORS.get(cls_name.upper().replace(" ", "_"), "#999")
                            fig.add_trace(go.Scatter(
                                x=df_c["year"], y=df_c["ha"],
                                mode="lines+markers", name=cls_name,
                                line=dict(width=2, color=cor), marker=dict(size=6, color=cor),
                            ))
                        fig.update_layout(
                            height=600, template="plotly_white",
                            margin=dict(t=10, b=40, l=40, r=10),
                            xaxis_title="Ano", yaxis_title="ha",
                            hovermode="x unified", legend=dict(font=dict(size=10))
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Colunas `classname` / `areamunkm` não encontradas.")

                with st.expander("📋 Tabela DETER AMZ"):
                    st.dataframe(df_deter, use_container_width=True, height=280)
                    st.download_button("⬇️ CSV", df_deter.to_csv(index=False).encode("utf-8"), "deter_amz_aoi.csv", "text/csv")

    # ── DETER CERRADO ─────────────────────────────────────────────────
    with tab_deter_cer:
        col_mapa_c, col_info_c = st.columns([8, 2])

        with col_mapa_c:
            st.markdown("### 🗺️ Mapa DETER Cerrado")
            m = _base_map("#F1C40F")
            folium.WmsTileLayer(
                url="https://terrabrasilis.dpi.inpe.br/geoserver/deter-cerrado-nb/deter_cerrado/ows",
                layers="deter_cerrado", fmt="image/png", transparent=True,
                name="DETER Cerrado", overlay=True, opacity=0.8,
            ).add_to(m)
            folium.LayerControl().add_to(m)
            st_folium(m, width=None, height=600, key="gcp_map_deter_cer")

        with col_info_c:
            if is_overview:
                st.info("💡 Selecione um projeto.")
            else:
                st.markdown("### 📋 Info")
                _info_panel(selected_gdf)

        if not is_overview and bbox_str:
            st.divider()
            with st.spinner("Consultando TerraBrasilis (DETER Cerrado)..."):
                df_cer = terrabrasilis_wfs(
                    "https://terrabrasilis.dpi.inpe.br/geoserver/deter-cerrado-nb/deter_cerrado/ows",
                    "deter-cerrado-nb:deter_cerrado", bbox_str,
                )
            if df_cer.empty:
                st.warning("Sem alertas DETER Cerrado para esta AOI.")
            else:
                st.dataframe(df_cer, use_container_width=True, height=300)
                st.download_button("⬇️ CSV", df_cer.to_csv(index=False).encode("utf-8"), "deter_cerrado_aoi.csv", "text/csv")
