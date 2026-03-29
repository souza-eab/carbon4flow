"""
carbon4flow_gcp.py
──────────────────
Schema real confirmado eg.:

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


"""
carbon4flow_gcp.py
──────────────────
Schema real confirmado:

  index.parquet        → resource_id (str), path (str)
  projetos/{id}.parquet → geometry (GeoPandas nativa Z), resource_id (str), project_name (str)
  df_all               → resourceIdentifier (int64), resourceName_x (str), ...

Regra de join: sempre por ID normalizado para str.

AOI AOI (Área de interesse do projeto):
  - GFW   : polígono GeoJSON enviado via API POST  ✅ já implementado
  - PRODES: WFS bbox → clip na AOI → recalcula área em km² ⚠️ Em desenvolvimento | Implementado
  - DETER : WFS bbox → clip na AOI → recalcula área em ha  ⚠️ Em desenvolvimento | Implementado

 
  - Versão 0.0.5 - 28/03/2026
  CHANGELOG v0.0.5:
  - [PERF] _fetch_and_clip_wfs: nova função cacheada (ttl=900) que une terrabrasilis_wfs +
           clip_and_recalculate em uma única chamada por (resource_id, fonte).
           Evita re-execução das queries WFS e do clip geoespacial ao navegar entre abas.
  - [PERF] render_aoi_tab: tabs PRODES, DETER AMZ e DETER Cerrado agora chamam
           _fetch_and_clip_wfs em vez de terrabrasilis_wfs + clip_and_recalculate diretamente.

  CHANGELOG v0.0.4:
  - [PERF] cache em carregar_geometrias_gcp (ttl=1800) — evita re-download a cada interação
  - [PERF] _get_aoi_bundle: dissolve único por resource_id, compartilhado entre geojson/bbox/geom
  - [CONF] clip_and_recalculate: aviso explícito na UI quando clip não ocorre (sem geometria)
  - [CONF] Nota sobre validação do fator areamunkm→ha adicionada ao código
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
from shapely.geometry import mapping, shape
from shapely.ops import unary_union, transform
from shapely import wkb as shapely_wkb
from google.cloud import storage
from google.oauth2 import service_account

log = logging.getLogger(__name__)

GCS_BUCKET = "edriano-verra-projects"
GCS_INDEX  = "index.parquet"

# CRS métrico para cálculo de área preciso no Brasil
# SIRGAS 2000 / Brasil Policônico (EPSG:5880) — cobre todo o território
CRS_AREA = "EPSG:5880"


# ═══════════════════════════════════════════════════════════════════════
# AUTENTICAÇÃO
# ═══════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def _gcs_client() -> storage.Client:
    info  = dict(st.secrets["gcp_service_account"])
    creds = service_account.Credentials.from_service_account_info(
        info, scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    return storage.Client(credentials=creds, project=creds.project_id)


def _blob_bytes(blob_path: str) -> bytes:
    return _gcs_client().bucket(GCS_BUCKET).blob(blob_path).download_as_bytes()


# ═══════════════════════════════════════════════════════════════════════
# INDEX
# ═══════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def load_index() -> pd.DataFrame:
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

@st.cache_data(ttl=300, show_spinner=False)
def _raw_parquet(blob_path: str) -> Optional[bytes]:
    try:
        return _blob_bytes(blob_path)
    except Exception as e:
        log.warning(f"Blob não encontrado: {blob_path} — {e}")
        return None


def load_project_geometry(resource_id: str) -> Optional[gpd.GeoDataFrame]:
    idx = load_index()
    rid = str(resource_id).strip()
    row = idx[idx["resource_id"] == rid]
    if row.empty:
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

        # Remove coordenada Z
        gdf["geometry"] = gdf["geometry"].apply(
            lambda g: transform(lambda x, y, z=None: (x, y), g)
            if g is not None and g.has_z else g
        )
        gdf["geometry"] = gdf["geometry"].buffer(0)
        gdf = gdf[gdf["geometry"].notnull() & ~gdf["geometry"].is_empty & gdf["geometry"].is_valid]
        return gdf if not gdf.empty else None
    except Exception as e:
        log.error(f"Erro ao ler parquet do projeto {rid}: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════
# [MELHORIA] AOI BUNDLE — dissolve único por resource_id
# Antes: get_dissolved_geometry, get_aoi_geojson e get_aoi_bbox_str
# eram chamadas separadamente em render_aoi_tab, cada uma refazendo
# o unary_union do zero. Agora um único @st.cache_data faz o dissolve
# uma só vez e retorna os três derivados juntos.
# ═══════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300, show_spinner=False)
def _get_aoi_bundle(resource_id: str) -> dict:
    """
    Retorna dict com:
      - geom    : Shapely geometry (dissolvida, 2D) — para clip
      - geojson : dict GeoJSON — para APIs GFW (POST)
      - bbox_str: 'minx,miny,maxx,maxy' — para queries WFS
    Retorna None em todos os campos se a geometria não estiver disponível.
    """
    gdf = load_project_geometry(resource_id)
    if gdf is None or gdf.empty:
        return {"geom": None, "geojson": None, "bbox_str": None}

    geom = unary_union(gdf.geometry)
    b    = geom.bounds
    return {
        "geom":     geom,
        "geojson":  mapping(geom),
        "bbox_str": f"{b[0]},{b[1]},{b[2]},{b[3]}",
    }


# Mantém funções públicas para compatibilidade com código externo
def get_dissolved_geometry(resource_id: str):
    """Geometria AOI dissolvida em um único polígono 2D."""
    return _get_aoi_bundle(resource_id)["geom"]


def get_aoi_geojson(resource_id: str) -> Optional[dict]:
    """GeoJSON da AOI real — para APIs GFW (POST)."""
    return _get_aoi_bundle(resource_id)["geojson"]


def get_aoi_bbox_str(resource_id: str) -> Optional[str]:
    """'minx,miny,maxx,maxy' — para queries WFS (bbox inicial)."""
    return _get_aoi_bundle(resource_id)["bbox_str"]


# ═══════════════════════════════════════════════════════════════════════
# CLIP AOI + RECÁLCULO DE ÁREA
# ═══════════════════════════════════════════════════════════════════════

def clip_and_recalculate(
    df_wfs: pd.DataFrame,
    aoi_geom,
    geom_col: str = "geometry",
    area_col_out: str = "area_km2_aoi",
    unit: str = "km2",          # "km2" ou "ha"
) -> Tuple[pd.DataFrame, bool]:
    """
    Recebe DataFrame do WFS (com coluna geometry como dict GeoJSON ou objeto Shapely),
    faz clip na AOI real e recalcula área dentro da AOI.

    Parâmetros
    ----------
    df_wfs       : DataFrame retornado por terrabrasilis_wfs.
                   Espera coluna 'geometry' (dict GeoJSON) e '_wfs_crs' (str EPSG).
                   Se ausentes, retorna sem clip com flag False.
    aoi_geom     : geometria Shapely da AOI (EPSG:4326)
    geom_col     : nome da coluna de geometria
    area_col_out : nome da nova coluna de área calculada
    unit         : "km2" ou "ha"

    Retorna
    -------
    (DataFrame, clip_realizado: bool)
      - DataFrame com coluna area_col_out adicionada (área dentro da AOI).
      - clip_realizado=False quando o clip não pôde ser executado
        (sem coluna geometry ou aoi_geom=None) — neste caso os dados
        representam a BBox completa, não a AOI.
    """
    if df_wfs.empty or aoi_geom is None:
        return df_wfs, False

    # Detecta CRS que veio do WFS (salvo pela terrabrasilis_wfs como '_wfs_crs')
    # Fallback para EPSG:4326 se coluna não existir (compatibilidade com dados antigos)
    wfs_crs = df_wfs["_wfs_crs"].iloc[0] if "_wfs_crs" in df_wfs.columns else "EPSG:4326"

    # Colunas internas que não devem ir para o DataFrame final
    _internal_cols = ["_wfs_crs"]

    # Se já é GeoDataFrame, usa direto; senão tenta reconstruir a partir da coluna geometry
    if isinstance(df_wfs, gpd.GeoDataFrame) and geom_col in df_wfs.columns:
        gdf = df_wfs.copy()
    elif geom_col in df_wfs.columns:
        try:
            geoms = df_wfs[geom_col].apply(
                lambda g: shape(g) if isinstance(g, dict) else g
            )
            gdf = gpd.GeoDataFrame(df_wfs, geometry=geoms, crs=wfs_crs)
        except Exception as e:
            log.warning(f"clip_and_recalculate: falha ao reconstruir geometria ({e}) — retornando sem clip")
            return df_wfs.drop(columns=_internal_cols, errors="ignore"), False
    else:
        log.warning(
            f"clip_and_recalculate: coluna '{geom_col}' ausente no DataFrame "
            f"— dados representam BBox inteira, não AOI."
        )
        return df_wfs.drop(columns=_internal_cols, errors="ignore"), False

    # Garante EPSG:4326 para o clip (AOI está em 4326)
    # EPSG:4674 (SIRGAS 2000) e EPSG:4326 (WGS 84) são praticamente idênticos
    # mas reprojetamos formalmente para evitar warnings do GeoPandas
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")

    # AOI como GeoDataFrame para o clip
    aoi_gdf = gpd.GeoDataFrame(geometry=[aoi_geom], crs="EPSG:4326")

    try:
        gdf_clipped = gpd.clip(gdf, aoi_gdf)
    except Exception as e:
        log.warning(f"clip falhou: {e} — retornando sem clip")
        return df_wfs.drop(columns=_internal_cols, errors="ignore"), False

    # Remove colunas internas antes de retornar
    drop_cols = [geom_col] + _internal_cols
    if gdf_clipped.empty:
        return gdf_clipped.drop(columns=drop_cols, errors="ignore"), True

    # Recalcula área no CRS métrico
    gdf_metric = gdf_clipped.to_crs(CRS_AREA)
    area_m2    = gdf_metric.geometry.area          # m²

    if unit == "km2":
        gdf_clipped[area_col_out] = area_m2 / 1_000_000
    else:  # ha
        gdf_clipped[area_col_out] = area_m2 / 10_000

    # Remove coluna geometry e colunas internas do retorno (DataFrame limpo para exibição)
    result = pd.DataFrame(gdf_clipped.drop(columns=drop_cols, errors="ignore"))
    return result, True


# ═══════════════════════════════════════════════════════════════════════
# [PERF v0.0.5] WFS + CLIP CACHEADOS POR PROJETO E FONTE
# Antes: terrabrasilis_wfs() e clip_and_recalculate() eram chamados
# diretamente dentro das tabs. Como o Streamlit re-executa render_aoi_tab()
# inteiro a cada troca de aba, as queries WFS (~5-20s cada) e o clip
# geoespacial (~1-3s) rodavam repetidamente sem cache.
# Agora: _fetch_and_clip_wfs() agrupa as duas operações em uma única
# função @st.cache_data com chave (resource_id, fonte), garantindo que
# o resultado é reutilizado em qualquer navegação subsequente entre abas.
# ═══════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=900, show_spinner=False)
def _fetch_and_clip_wfs(
    resource_id: str,
    fonte: str,                  # chave de cache semântica: "prodes" | "deter_amz" | "deter_cer"
    wfs_url: str,
    wfs_typename: str,
    bbox_str: str,
    aoi_geom_wkb: Optional[bytes],  # geometria serializada (Shapely não é hashável pelo cache)
    area_col_out: str,
    unit: str,
    terrabrasilis_wfs_fn,
) -> Tuple[pd.DataFrame, bool]:
    """
    Executa terrabrasilis_wfs + clip_and_recalculate em uma única chamada cacheada.

    Parâmetros
    ----------
    resource_id     : ID do projeto (usado como chave de cache)
    fonte           : string descritiva da fonte — apenas para compor a chave de cache
    wfs_url         : URL base do endpoint WFS
    wfs_typename    : typeName do layer WFS
    bbox_str        : BBox 'minx,miny,maxx,maxy'
    aoi_geom_wkb    : geometria AOI serializada via .wkb (bytes) — None se indisponível
    area_col_out    : nome da coluna de área recalculada no resultado
    unit            : "km2" ou "ha"
    terrabrasilis_wfs_fn : referência à função terrabrasilis_wfs (injetada para evitar
                           dependência circular de importação)

    Retorna
    -------
    (DataFrame resultado, clip_realizado: bool)
    """
    df_raw = terrabrasilis_wfs_fn(wfs_url, wfs_typename, bbox_str)

    if df_raw.empty:
        return df_raw, False

    aoi_geom = shapely_wkb.loads(aoi_geom_wkb) if aoi_geom_wkb is not None else None

    return clip_and_recalculate(
        df_raw, aoi_geom,
        geom_col="geometry",
        area_col_out=area_col_out,
        unit=unit,
    )


# ═══════════════════════════════════════════════════════════════════════
# NORMALIZAÇÃO df_all
# ═══════════════════════════════════════════════════════════════════════

def _prep_df_all(df_all: pd.DataFrame) -> pd.DataFrame:
    df = df_all.copy()
    df["resourceIdentifier"] = df["resourceIdentifier"].astype(str).str.strip()
    return df


# ═══════════════════════════════════════════════════════════════════════
# CARREGAR TODAS AS GEOMETRIAS (Visão Geral)
# [MELHORIA PERF] Adicionado @st.cache_data(ttl=1800) para evitar que
# qualquer interação com widgets re-execute o loop de download.
# ═══════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=1800, show_spinner=False)
def carregar_geometrias_gcp(df_all: pd.DataFrame) -> Tuple[gpd.GeoDataFrame, List[str]]:
    df  = _prep_df_all(df_all)
    idx = load_index()

    if idx.empty:
        return gpd.GeoDataFrame(), ["index.parquet vazio ou inacessível"]

    ids_gcs = set(idx["resource_id"].tolist())
    ids_df  = set(df["resourceIdentifier"].tolist())
    ids     = sorted(ids_gcs & ids_df)

    if not ids:
        return gpd.GeoDataFrame(), [
            f"Nenhum ID em comum. GCS: {list(ids_gcs)[:3]} | df_all: {list(ids_df)[:3]}"
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
            gdf = gdf[gdf["geometry"].notnull() & ~gdf["geometry"].is_empty & gdf["geometry"].is_valid]

            if gdf.empty:
                erros.append(f"{rid}: geometria vazia após sanitização")
                continue

            gdf["resourceIdentifier"] = rid
            gdfs.append(gdf)

        except Exception as e:
            erros.append(f"{rid}: {e}")

    prog.empty()

    if not gdfs:
        return gpd.GeoDataFrame(), erros

    gdf_combined = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs="EPSG:4326")

    meta_cols = [c for c in [
        "resourceIdentifier", "resourceName_x", "state_Recode",
        "vcsAFOLUActivity", "vcsProjectStatus", "vcsMethodology",
        "vcsCreditingPeriodTerm", "vcsAcresHectares", "description",
    ] if c in df.columns]

    df_ref = df[meta_cols].drop_duplicates("resourceIdentifier")
    extra  = [c for c in meta_cols if c != "resourceIdentifier" and c in gdf_combined.columns]
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
        "Geometrias _baixadas_ em *.kml*, _convertidas_ em *.parquet*  e carregadas do **Google Cloud Storage**."
        "🚧Os cálculos de área são estimativas, busca usar o clip na **AOI** (Área de interesse do projeto) e na **BBox** (Bounding Box)."
    )

    # ── Catálogo ──────────────────────────────────────────────────────
    with st.spinner("📋 Carregando catálogo..."):
        idx = load_index()

    if idx.empty:
        st.error("❌ Não foi possível carregar o index.parquet do GCS.")
        return

    df = _prep_df_all(df_all)

    meta_cols = [c for c in [
        "resourceName_x", "state_Recode", "vcsAFOLUActivity",
        "vcsProjectStatus", "vcsMethodology", "vcsCreditingPeriodTerm",
        "vcsAcresHectares", "description",
    ] if c in df.columns]

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
        return f"{nome} | {estado}"

    options     = ["🌎 Visão Geral (Todos os Projetos)"] + [_label(r) for _, r in idx.iterrows()]
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
        aoi_geom     = None

    else:
        label_nome = selected.split(" | ")[0].strip()
        row_idx    = idx[idx["resourceName_x"] == label_nome]
        if row_idx.empty:
            row_idx = idx[idx["resource_id"] == label_nome]
        if row_idx.empty:
            st.warning("Projeto não encontrado no catálogo.")
            return

        selected_rid = str(row_idx.iloc[0]["resource_id"]).strip()

        with st.spinner("🗺️ Carregando geometria..."):
            gdf_single = load_project_geometry(selected_rid)

        if gdf_single is None or gdf_single.empty:
            st.warning(f"⚠️ Geometria não disponível no GCS para ID `{selected_rid}`.")
            return

        for col in meta_cols:
            if col not in gdf_single.columns:
                gdf_single[col] = row_idx.iloc[0].get(col, "N/A")
        gdf_single["resourceIdentifier"] = selected_rid

        bounds       = gdf_single.total_bounds
        center       = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]
        zoom_start   = 10
        selected_gdf = gdf_single

        # [MELHORIA PERF] Uma única chamada a _get_aoi_bundle substitui
        # as três chamadas anteriores (get_aoi_bbox_str, get_aoi_geojson,
        # get_dissolved_geometry) que refaziam o unary_union cada uma.
        with st.spinner("📐 Calculando AOI..."):
            aoi_bundle = _get_aoi_bundle(selected_rid)
        bbox_str    = aoi_bundle["bbox_str"]
        aoi_geojson = aoi_bundle["geojson"]
        aoi_geom    = aoi_bundle["geom"]

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
        #folium.TileLayer("Esri.WorldImagery", name="Satélite", control=False).add_to(m)
        #folium.TileLayer("Cartodb dark_matter", name="CartodB dark_matter", control=False).add_to(m)      
        folium.TileLayer("CartoDB.Positron", name="CartoDB.Positron", control=False).add_to(m)      
        if not is_overview:
            b = selected_gdf.total_bounds
            folium.PolyLine(
                [[b[1],b[0]],[b[1],b[2]],[b[3],b[2]],[b[3],b[0]],[b[1],b[0]]],
                color="#00FFFF", weight=2, dash_array="6 4",
                tooltip="Bounding Box — área de consulta WFS inicial",
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

    # ── GFW (AOI real via POST) ───────────────────────────────────────
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
                st.caption("✅ GFW: AOI (POST)" if aoi_geojson else "⚠️ AOI indisponível")

        if not is_overview and aoi_geojson:
            st.divider()
            col_g1, col_g2, col_g3 = st.columns(3)

            # [MELHORIA PERF] As 3 chamadas GFW agora rodam em paralelo com
            # ThreadPoolExecutor em vez de sequencialmente (antes: até 180 s
            # no pior caso com 3 timeouts de 60 s cada).
            import concurrent.futures

            def _fetch_loss():
                return gfw_tree_cover_loss(aoi_geojson, GFW_API_KEY)

            def _fetch_glad():
                return gfw_glad_alerts(aoi_geojson, GFW_API_KEY)

            def _fetch_radd():
                return gfw_radd_alerts(aoi_geojson, GFW_API_KEY)

            with st.spinner("Consultando GFW (Tree Cover Loss, GLAD e RADD em paralelo)..."):
                with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                    fut_loss = executor.submit(_fetch_loss)
                    fut_glad = executor.submit(_fetch_glad)
                    fut_radd = executor.submit(_fetch_radd)
                    df_loss = fut_loss.result()
                    df_glad = fut_glad.result()
                    df_radd = fut_radd.result()

            with col_g1:
                st.markdown("### 📊 Tree Cover Loss")
                if df_loss.empty:
                    st.warning("Sem dados para esta AOI.")
                else:
                    st.plotly_chart(_bar(df_loss["umd_tree_cover_loss__year"], df_loss["loss_ha"], "#ff4444", y_title="ha"), use_container_width=True)

            with col_g2:
                st.markdown("### 🟡 GLAD Alerts")
                if df_glad.empty:
                    st.warning("Sem dados para esta AOI.")
                else:
                    st.plotly_chart(_bar(df_glad["alert__year"], df_glad["alert_count"], "#FFC300", y_title="Alertas"), use_container_width=True)

            with col_g3:
                st.markdown("### 🟠 RADD Alerts")
                if df_radd.empty:
                    st.warning("Sem dados para esta AOI.")
                else:
                    st.plotly_chart(_bar(df_radd["alert__year"], df_radd["alert_count"], "#FF7900", y_title="Alertas"), use_container_width=True)

        elif not is_overview:
            st.info("ℹ️ AOI não disponível — consultas GFW indisponíveis.")

    # ── PRODES (WFS bbox → clip AOI → recalcula área km²) ─────────────
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
                st.caption("✅ PRODES: clip na AOI  e área recalculada em km²" if aoi_geom else "⚠️ AOI indisponível")

        if not is_overview and bbox_str:
            st.divider()

            # [PERF v0.0.5] _fetch_and_clip_wfs: query WFS + clip cacheados por (resource_id, fonte)
            aoi_geom_wkb = shapely_wkb.dumps(aoi_geom) if aoi_geom is not None else None
            with st.spinner("Consultando TerraBrasilis WFS (PRODES AMZ)..."):
                df_prodes, clip_ok_prodes = _fetch_and_clip_wfs(
                    resource_id     = selected_rid,
                    fonte           = "prodes",
                    wfs_url         = "https://terrabrasilis.dpi.inpe.br/geoserver/prodes-legal-amz/yearly_deforestation/ows",
                    wfs_typename    = "prodes-legal-amz:yearly_deforestation",
                    bbox_str        = bbox_str,
                    aoi_geom_wkb    = aoi_geom_wkb,
                    area_col_out    = "area_km2_aoi",
                    unit            = "km2",
                    terrabrasilis_wfs_fn = terrabrasilis_wfs,
                )

            # Aviso quando clip não ocorreu
            if not clip_ok_prodes and not df_prodes.empty:
                st.warning(
                    "⚠️ **Clip não realizado:** o WFS retornou dados sem coluna de geometria. "
                    "Os valores de área exibidos são da **BBox inteira**, não da AOI do projeto."
                )
            elif clip_ok_prodes and not df_prodes.empty:
                st.caption(
                    f"📐 **{len(df_prodes)} polígonos dentro da AOI** após clip."
                )
            elif df_prodes.empty and not clip_ok_prodes:
                st.warning("⚠️ AOI indisponível — área exibida refere-se à BBox inteira.")

            col_g1, col_g2, col_g3 = st.columns(3)

            with col_g1:
                st.markdown("### 📊 Polígonos por Ano (AOI)")
                if df_prodes.empty:
                    st.warning("Sem dados PRODES para esta AOI.")
                elif "year" in df_prodes.columns:
                    df_prodes["year"] = pd.to_numeric(df_prodes["year"], errors="coerce")
                    by_year = df_prodes.groupby("year").size().reset_index(name="n")
                    st.plotly_chart(_bar(by_year["year"], by_year["n"], "#C0392B", y_title="Polígonos"), use_container_width=True)

            with col_g2:
                area_label = "km² (AOI)" if clip_ok_prodes else "km² (BBox — sem clip)"
                st.markdown(f"### 📊 Área dentro da AOI ({area_label})")
                if not df_prodes.empty:
                    area_col = "area_km2_aoi" if "area_km2_aoi" in df_prodes.columns else "area_km"
                    if area_col in df_prodes.columns and "year" in df_prodes.columns:
                        df_prodes[area_col] = pd.to_numeric(df_prodes[area_col], errors="coerce")
                        by_area = df_prodes.groupby("year").agg(a=(area_col, "sum")).reset_index()
                        fig = go.Figure(go.Bar(x=by_area["year"], y=by_area["a"], marker_color="#922B21"))
                        fig.update_layout(
                            height=300, template="plotly_white",
                            margin=dict(t=10, b=40, l=40, r=10),
                            xaxis_title="Ano",
                            yaxis_title=area_label,
                            hovermode="x unified"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Coluna de área não encontrada.")

            with col_g3:
                st.markdown("### 📊 Classe de Desmatamento")
                if not df_prodes.empty and "class_name" in df_prodes.columns:
                    cls = df_prodes["class_name"].value_counts().reset_index()
                    cls.columns = ["Classe", "n"]
                    st.plotly_chart(_bar(cls["Classe"], cls["n"], "#E74C3C", x_title="Classe", y_title="Polígonos"), use_container_width=True)

            if not df_prodes.empty:
                with st.expander("📋 Tabela PRODES (AOI)"):
                    st.dataframe(df_prodes, use_container_width=True, height=280)
                    st.download_button("⬇️ CSV", df_prodes.to_csv(index=False).encode("utf-8"), "prodes_aoi_clip.csv", "text/csv")

    # ── DETER AMAZÔNIA (WFS bbox → clip AOI → recalcula área ha) ──────
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
                st.caption("✅ DETER: clip na AOI  e área recalculada em ha" if aoi_geom else "⚠️ AOI indisponível")
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

            # [PERF v0.0.5] _fetch_and_clip_wfs: query WFS + clip cacheados por (resource_id, fonte)
            aoi_geom_wkb = shapely_wkb.dumps(aoi_geom) if aoi_geom is not None else None
            with st.spinner("Consultando TerraBrasilis WFS (DETER AMZ)..."):
                df_deter, clip_ok_deter = _fetch_and_clip_wfs(
                    resource_id     = selected_rid,
                    fonte           = "deter_amz",
                    wfs_url         = "https://terrabrasilis.dpi.inpe.br/geoserver/deter-amz/deter_amz/ows",
                    wfs_typename    = "deter-amz:deter_amz",
                    bbox_str        = bbox_str,
                    aoi_geom_wkb    = aoi_geom_wkb,
                    area_col_out    = "area_ha_aoi",
                    unit            = "ha",
                    terrabrasilis_wfs_fn = terrabrasilis_wfs,
                )

            if not clip_ok_deter and not df_deter.empty:
                st.warning(
                    "⚠️ **Clip não realizado:** o WFS retornou dados sem coluna de geometria. "
                    "Os valores de área exibidos são da **BBox inteira**, não da AOI do projeto."
                )
            elif clip_ok_deter and not df_deter.empty:
                st.caption(
                    f"📐 **{len(df_deter)} alertas dentro da AOI** após clip."
                )
            elif df_deter.empty and not clip_ok_deter:
                st.warning("⚠️ AOI indisponível — área exibida refere-se à BBox inteira.")

            if df_deter.empty:
                st.warning("Sem alertas DETER para esta AOI.")
            else:
                # Processa coluna year se existir view_date
                if "view_date" in df_deter.columns and "year" not in df_deter.columns:
                    df_deter["year"] = pd.to_datetime(df_deter["view_date"], errors="coerce").dt.year

                col_g1, col_g2 = st.columns([2, 3])

                with col_g1:
                    if "year" in df_deter.columns:
                        by_year = df_deter.groupby("year").size().reset_index(name="n")
                        st.markdown("### 📊 Alertas por Ano (AOI)")
                        st.plotly_chart(_bar(by_year["year"], by_year["n"], "#E67E22", y_title="Alertas"), use_container_width=True)

                    if "classname" in df_deter.columns:
                        cls = df_deter["classname"].value_counts().reset_index()
                        cls.columns = ["Classe", "n"]
                        st.markdown("### 📊 Por Classe")
                        st.plotly_chart(_bar(cls["Classe"], cls["n"], "#D35400", x_title="Classe", y_title="Alertas"), use_container_width=True)

                with col_g2:
                    area_label_deter = "ha (AOI)" if clip_ok_deter else "ha (BBox — sem clip)"
                    st.markdown(f"### 📊 Área por Classe ao Longo do Tempo ({area_label_deter})")
                    DETER_COLORS = {
                        "CICATRIZ_DE_QUEIMADA": "#d7191c", "CORTE_SELETIVO":  "#868686",
                        "CS_DESORDENADO":       "#db83ff", "CS_GEOMETRICO":   "#ff7e00",
                        "DEGRADACAO":           "#e6c300", "DESMATAMENTO_CR": "#8a5f4b",
                        "DESMATAMENTO_VEG":     "#abdda4", "MINERACAO":       "#4223e5",
                    }

                    # Prefere área recalculada; fallback para areamunkm original
                    area_col = "area_ha_aoi" if "area_ha_aoi" in df_deter.columns else "areamunkm"

                    if "classname" in df_deter.columns and area_col in df_deter.columns and "year" in df_deter.columns:
                        df_deter[area_col] = pd.to_numeric(df_deter[area_col], errors="coerce")
                        # [NOTA CONFIABILIDADE] O campo areamunkm do TerraBrasilis/DETER
                        # representa km² do município interceptado, não do polígono de alerta.
                        # A conversão ×100 para ha é uma aproximação — confirmar na documentação
                        # oficial do INPE/TerraBrasilis antes de usar em análises formais.
                        # Ref: https://terrabrasilis.dpi.inpe.br/
                        if area_col == "areamunkm":
                            df_deter[area_col] = df_deter[area_col] * 100

                        df_area = df_deter.groupby(["year", "classname"]).agg(
                            ha=(area_col, "sum")
                        ).reset_index()

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
                            xaxis_title="Ano",
                            yaxis_title=area_label_deter,
                            hovermode="x unified",
                            legend=dict(font=dict(size=10))
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Colunas necessárias para o gráfico de área não encontradas.")

                with st.expander("📋 Tabela DETER AMZ (AOI)"):
                    st.dataframe(df_deter, use_container_width=True, height=280)
                    st.download_button("⬇️ CSV", df_deter.to_csv(index=False).encode("utf-8"), "deter_amz_aoi_clip.csv", "text/csv")

    # ── DETER CERRADO (WFS bbox → clip AOI → recalcula área ha) ───────
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
                st.caption("✅ DETER Cerrado: clip na AOI e área recalculada em ha" if aoi_geom else "⚠️ AOI indisponível")

        if not is_overview and bbox_str:
            st.divider()

            # [PERF v0.0.5] _fetch_and_clip_wfs: query WFS + clip cacheados por (resource_id, fonte)
            aoi_geom_wkb = shapely_wkb.dumps(aoi_geom) if aoi_geom is not None else None
            with st.spinner("Consultando TerraBrasilis WFS (DETER Cerrado)..."):
                df_cer, clip_ok_cer = _fetch_and_clip_wfs(
                    resource_id     = selected_rid,
                    fonte           = "deter_cer",
                    wfs_url         = "https://terrabrasilis.dpi.inpe.br/geoserver/deter-cerrado-nb/deter_cerrado/ows",
                    wfs_typename    = "deter-cerrado-nb:deter_cerrado",
                    bbox_str        = bbox_str,
                    aoi_geom_wkb    = aoi_geom_wkb,
                    area_col_out    = "area_ha_aoi",
                    unit            = "ha",
                    terrabrasilis_wfs_fn = terrabrasilis_wfs,
                )

            if not clip_ok_cer and not df_cer.empty:
                st.warning(
                    "⚠️ **Clip não realizado:** o WFS retornou dados sem coluna de geometria. "
                    "Os valores de área exibidos são da **BBox inteira**, não da AOI do projeto."
                )
            elif clip_ok_cer and not df_cer.empty:
                st.caption(
                    f"📐 **{len(df_cer)} alertas dentro da AOI** após clip."
                )
            elif df_cer.empty and not clip_ok_cer:
                st.warning("⚠️ AOI indisponível — área exibida refere-se à BBox inteira.")

            if df_cer.empty:
                st.warning("Sem alertas DETER Cerrado para esta AOI.")
            else:
                st.dataframe(df_cer, use_container_width=True, height=300)
                st.download_button("⬇️ CSV", df_cer.to_csv(index=False).encode("utf-8"), "deter_cerrado_aoi_clip.csv", "text/csv")
