#!pip install pandas geopandas shapely google-cloud-storage pyarrow fastparquet streamlit plotly folium streamlit-folium requests scipy numpy

"""
Carbon4Flow — GCP Parquet Edition
Migração: KMLs locais no GitHub → Parquets no Google Cloud Storage

Estrutura GCS esperada:
  gs://edriano-verra-projects/
  ├── index.parquet          ← catálogo leve (resourceIdentifier, name, state, bbox, etc.)
  ├── projetos/
  │   ├── 5859.parquet       ← geometria WKT + atributos por projeto
  │   ├── 5837.parquet
  │   └── ...
  └── metadata/
      └── projects_metadata.parquet

Cada arquivo em projetos/{id}.parquet deve conter ao menos:
  - resourceIdentifier (str)
  - geometry_wkt (str)  ← geometria em WKT, ex: "POLYGON((...))"
  - [outros atributos opcionais]

O index.parquet deve conter:
  - resourceIdentifier
  - resourceName_x
  - state_Recode
  - vcsAFOLUActivity
  - vcsProjectStatus
  - vcsMethodology
  - vcsCreditingPeriodTerm
  - vcsAcresHectares
  - description
  - bbox_minx, bbox_miny, bbox_maxx, bbox_maxy  ← para queries rápidas sem baixar geometria
"""

import os
import io
import json
import logging
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import geopandas as gpd
import streamlit as st
import requests
import plotly.express as px
import plotly.graph_objects as go
import folium
from folium.plugins import MarkerCluster, HeatMap
from streamlit_folium import st_folium
from shapely import wkt
from shapely.geometry import mapping, box
from shapely.errors import TopologicalError
from shapely.ops import unary_union
from datetime import datetime
from scipy import stats
from google.cloud import storage
from google.oauth2 import service_account

logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURAÇÃO GCS
# ============================================================

GCS_BUCKET = "edriano-verra-projects"
GCS_INDEX  = "index.parquet"
GCS_PROJ   = "projetos"          # prefixo dos parquets por projeto
GCS_META   = "metadata/projects_metadata.parquet"


@st.cache_resource(show_spinner=False)
def get_gcs_client() -> storage.Client:
    """
    Retorna cliente GCS autenticado.
    Suporta dois modos:
      1. st.secrets["gcp_service_account"] → dict com credenciais JSON
      2. Application Default Credentials (ADC) — útil em Cloud Run / GKE
    """
    try:
        if "gcp_service_account" in st.secrets:
            creds = service_account.Credentials.from_service_account_info(
                st.secrets["gcp_service_account"],
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            return storage.Client(credentials=creds, project=creds.project_id)
        else:
            # ADC — funciona em ambientes GCP ou com gcloud auth application-default login
            return storage.Client()
    except Exception as e:
        st.error(f"❌ Erro ao autenticar no GCS: {e}")
        raise


def _gcs_read_parquet(blob_path: str) -> pd.DataFrame:
    """Baixa um blob GCS e retorna DataFrame. Sem cache — use wrappers cacheados."""
    client = get_gcs_client()
    bucket = client.bucket(GCS_BUCKET)
    blob   = bucket.blob(blob_path)
    data   = blob.download_as_bytes()
    return pd.read_parquet(io.BytesIO(data))


# ============================================================
# INDEX — catálogo leve (carregado uma vez na sessão)
# ============================================================

@st.cache_data(ttl=3600, show_spinner="📋 Carregando catálogo de projetos...")
def load_index() -> pd.DataFrame:
    """
    Carrega index.parquet — catálogo leve com metadados de todos os projetos.
    Colunas esperadas: resourceIdentifier, resourceName_x, state_Recode,
                       vcsAFOLUActivity, vcsProjectStatus, bbox_minx/y, bbox_maxx/y
    """
    try:
        df = _gcs_read_parquet(GCS_INDEX)
        df["resourceIdentifier"] = df["resourceIdentifier"].astype(str)
        return df
    except Exception as e:
        st.error(f"❌ Erro ao carregar index.parquet: {e}")
        return pd.DataFrame()


# ============================================================
# GEOMETRIA POR PROJETO — lazy loading
# ============================================================

@st.cache_data(ttl=1800, show_spinner="🗺️ Carregando geometria do projeto...")
def load_project_geometry(resource_id: str) -> Optional[gpd.GeoDataFrame]:
    """
    Baixa projetos/{resource_id}.parquet e retorna GeoDataFrame.
    A coluna 'geometry_wkt' é convertida para geometria Shapely.
    Retorna None se o arquivo não existir ou falhar.
    """
    blob_path = f"{GCS_PROJ}/{resource_id}.parquet"
    try:
        df = _gcs_read_parquet(blob_path)
        if "geometry_wkt" not in df.columns:
            logger.warning(f"Projeto {resource_id}: coluna 'geometry_wkt' ausente.")
            return None
        df["geometry"] = df["geometry_wkt"].apply(
            lambda g: wkt.loads(g) if pd.notna(g) else None
        )
        df = df.dropna(subset=["geometry"])
        gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
        gdf["geometry"] = gdf["geometry"].buffer(0)  # sanitiza
        return gdf
    except Exception as e:
        logger.warning(f"Projeto {resource_id} não encontrado no GCS: {e}")
        return None


@st.cache_data(ttl=3600, show_spinner="🗺️ Carregando todas as geometrias...")
def load_all_geometries(resource_ids: Tuple[str, ...]) -> gpd.GeoDataFrame:
    """
    Carrega geometrias de múltiplos projetos em lote.
    Recebe tupla (hashável para cache) de resourceIdentifiers.
    Retorna GeoDataFrame consolidado.
    """
    gdfs = []
    failed = []
    progress = st.progress(0, text="Carregando geometrias...")
    total = len(resource_ids)

    for i, rid in enumerate(resource_ids):
        gdf = load_project_geometry(rid)
        if gdf is not None:
            gdfs.append(gdf)
        else:
            failed.append(rid)
        progress.progress((i + 1) / total, text=f"Carregando {i+1}/{total}...")

    progress.empty()

    if not gdfs:
        return gpd.GeoDataFrame()

    combined = gpd.GeoDataFrame(
        pd.concat(gdfs, ignore_index=True),
        crs="EPSG:4326"
    )

    if failed:
        st.caption(f"⚠️ {len(failed)} projetos sem geometria disponível no GCS.")

    return combined


# ============================================================
# FUNÇÃO PRINCIPAL: carregar geometrias para a aba AOI
# Substitui a antiga carregar_geometrias(kml_dir)
# ============================================================

def carregar_geometrias_gcp(
    df_all: pd.DataFrame,
    mode: str = "overview"          # "overview" | "single"
) -> Tuple[gpd.GeoDataFrame, List[str]]:
    """
    Substituto direto de carregar_geometrias(kml_dir).

    Parâmetros
    ----------
    df_all : DataFrame com coluna 'resourceIdentifier' — lista de projetos a carregar
    mode   : "overview" carrega todos; "single" é chamado por load_project_geometry()

    Retorna
    -------
    (GeoDataFrame com geometrias + colunas de df_all mescladas, lista de erros)
    """
    df_all = df_all.copy()
    df_all["resourceIdentifier"] = df_all["resourceIdentifier"].astype(str)

    ids = tuple(df_all["resourceIdentifier"].unique().tolist())

    if mode == "overview" and len(ids) > 50:
        # Visão geral: usa apenas o index com bbox para não baixar tudo
        idx = load_index()
        if idx.empty:
            return gpd.GeoDataFrame(), ["index.parquet vazio"]

        # Constrói geometrias de bounding box (levíssimas) para visão geral
        bbox_cols = ["bbox_minx", "bbox_miny", "bbox_maxx", "bbox_maxy"]
        if all(c in idx.columns for c in bbox_cols):
            idx = idx.dropna(subset=bbox_cols)
            idx["geometry"] = idx.apply(
                lambda r: box(r.bbox_minx, r.bbox_miny, r.bbox_maxx, r.bbox_maxy),
                axis=1
            )
            gdf = gpd.GeoDataFrame(idx, geometry="geometry", crs="EPSG:4326")
            cols_merge = [c for c in df_all.columns if c not in gdf.columns or c == "resourceIdentifier"]
            gdf = gdf.merge(df_all[cols_merge], on="resourceIdentifier", how="left")
            return gdf, []
        else:
            st.info("ℹ️ index.parquet sem colunas bbox_* — carregando geometrias reais (pode demorar).")

    # Carrega geometrias reais
    gdf_combined = load_all_geometries(ids)

    if gdf_combined.empty:
        return gpd.GeoDataFrame(), ["Nenhuma geometria carregada"]

    # Mescla com atributos de df_all
    cols_merge = [
        "resourceIdentifier", "resourceName_x", "state_Recode",
        "vcsAFOLUActivity", "vcsProjectStatus", "vcsMethodology",
        "vcsCreditingPeriodTerm", "vcsAcresHectares", "description"
    ]
    cols_merge = [c for c in cols_merge if c in df_all.columns]
    df_ref = df_all[cols_merge].drop_duplicates("resourceIdentifier")

    gdf_combined["resourceIdentifier"] = gdf_combined["resourceIdentifier"].astype(str)

    # Remove colunas duplicadas antes do merge (exceto a chave)
    extra_cols = [c for c in cols_merge if c != "resourceIdentifier" and c in gdf_combined.columns]
    gdf_combined = gdf_combined.drop(columns=extra_cols, errors="ignore")

    gdf_combined = gdf_combined.merge(df_ref, on="resourceIdentifier", how="left")
    gdf_combined = gpd.GeoDataFrame(gdf_combined, geometry="geometry", crs="EPSG:4326")

    return gdf_combined, []


# ============================================================
# GEOMETRIA PARA AOI REAL (vs BBox)
# Substitui o uso do total_bounds no código original
# ============================================================

def get_aoi_geojson(resource_id: str) -> Optional[dict]:
    """
    Retorna o GeoJSON da geometria real do projeto para uso nas APIs GFW.
    Antes era approximado pela BBox — agora usa a geometria real do Parquet.
    """
    gdf = load_project_geometry(resource_id)
    if gdf is None or gdf.empty:
        return None

    geom = gdf.geometry.iloc[0]

    # Dissolve se MultiPolygon
    if geom.geom_type == "GeometryCollection":
        polys = [g for g in geom.geoms if g.geom_type in ["Polygon", "MultiPolygon"]]
        geom = unary_union(polys) if polys else None

    if geom is None or geom.geom_type not in ["Polygon", "MultiPolygon"]:
        return None

    return mapping(geom)


def get_aoi_bbox_str(resource_id: str) -> Optional[str]:
    """
    Retorna string 'minx,miny,maxx,maxy' para queries WFS.
    Tenta usar o index.parquet (rápido) antes de baixar a geometria.
    """
    # Tenta index primeiro
    idx = load_index()
    if not idx.empty:
        bbox_cols = ["bbox_minx", "bbox_miny", "bbox_maxx", "bbox_maxy"]
        row = idx[idx["resourceIdentifier"].astype(str) == str(resource_id)]
        if not row.empty and all(c in row.columns for c in bbox_cols):
            r = row.iloc[0]
            return f"{r.bbox_minx},{r.bbox_miny},{r.bbox_maxx},{r.bbox_maxy}"

    # Fallback: baixa geometria real
    gdf = load_project_geometry(resource_id)
    if gdf is None or gdf.empty:
        return None
    b = gdf.total_bounds
    return f"{b[0]},{b[1]},{b[2]},{b[3]}"


# ============================================================
# PIPELINE — como gerar os parquets no GCS
# (rode isto localmente / em notebook para popular o bucket)
# ============================================================

PIPELINE_INSTRUCTIONS = """
# ── PIPELINE: KML → GCS Parquet ──────────────────────────────────────────

Para migrar seus KMLs para o GCS execute o script abaixo uma vez.
Dependências: geopandas, shapely, google-cloud-storage, pyarrow, pandas

```python
import os, json
import geopandas as gpd
import pandas as pd
from shapely.geometry import mapping, box
from shapely.ops import unary_union
from google.cloud import storage

BUCKET_NAME = "edriano-verra-projects"
KML_DIR     = "./kml"           # pasta com seus KMLs atuais

client = storage.Client()       # usa ADC ou GOOGLE_APPLICATION_CREDENTIALS
bucket = client.bucket(BUCKET_NAME)

index_rows = []

for kml_file in os.listdir(KML_DIR):
    if not kml_file.lower().endswith(".kml"):
        continue

    resource_id = kml_file.split("_")[0]
    kml_path    = os.path.join(KML_DIR, kml_file)

    try:
        gdf = gpd.read_file(kml_path, driver="KML")
        gdf = gdf.to_crs("EPSG:4326")
        gdf["geometry"] = gdf["geometry"].buffer(0)

        # Dissolve tudo em uma geometria por projeto
        geom = unary_union(gdf.geometry)
        bounds = geom.bounds  # (minx, miny, maxx, maxy)

        # ── Grava projetos/{id}.parquet ──
        df_out = pd.DataFrame({
            "resourceIdentifier": [resource_id],
            "geometry_wkt":       [geom.wkt],
        })
        buf = df_out.to_parquet(index=False)
        blob = bucket.blob(f"projetos/{resource_id}.parquet")
        blob.upload_from_string(buf, content_type="application/octet-stream")
        print(f"✅ {resource_id}")

        index_rows.append({
            "resourceIdentifier": resource_id,
            "bbox_minx": bounds[0],
            "bbox_miny": bounds[1],
            "bbox_maxx": bounds[2],
            "bbox_maxy": bounds[3],
        })

    except Exception as e:
        print(f"❌ {kml_file}: {e}")

# ── Grava index.parquet ──
df_index = pd.DataFrame(index_rows)
# Junte com seus metadados (df_all) aqui se quiser enriquecer o index
buf = df_index.to_parquet(index=False)
bucket.blob("index.parquet").upload_from_string(buf, content_type="application/octet-stream")
print(f"📋 index.parquet com {len(df_index)} projetos")
```
"""


# ============================================================
# PATCH MÍNIMO PARA O APP EXISTENTE
# (adicione este bloco no topo do app, substituindo a seção de KML)
# ============================================================

PATCH_FOR_EXISTING_APP = '''
# ── Cole este bloco no lugar de carregar_geometrias() e da seção KML ──

from carbon4flow_gcp import (
    carregar_geometrias_gcp,
    get_aoi_geojson,
    get_aoi_bbox_str,
    load_project_geometry,
)

# Na aba story_tabs[1] (Dados AOI), substitua:
#   gdf_kml, erros = carregar_geometrias(KML_DIR)
# por:
#   gdf_kml, erros = carregar_geometrias_gcp(df_all, mode="overview")

# Para projetos individuais, substitua:
#   geojson_poly = mapping(geom)
# por:
#   geojson_poly = get_aoi_geojson(row_proj["resourceIdentifier"])

# Para o bbox_str do WFS, substitua:
#   bbox_str = f"{b[0]},{b[1]},{b[2]},{b[3]}"
# por:
#   bbox_str = get_aoi_bbox_str(row_proj["resourceIdentifier"])
'''


# ============================================================
# VERSÃO COMPLETA DA ABA STORYTELLING 1 (Dados AOI)
# ============================================================
# Copie esta função e chame-a dentro de with story_tabs[1]:

def render_aoi_tab(df_all: pd.DataFrame, GFW_API_KEY: str,
                   gfw_tree_cover_loss, gfw_glad_alerts, gfw_radd_alerts,
                   terrabrasilis_wfs, ACTIVITY_COLORS: dict):
    """
    Versão GCP da aba 'Dados AOI'.
    Substitui a seção com story_tabs[1] do app original.

    Parâmetros
    ----------
    df_all              : DataFrame principal
    GFW_API_KEY         : chave API Global Forest Watch
    gfw_tree_cover_loss : função original do app
    gfw_glad_alerts     : função original do app
    gfw_radd_alerts     : função original do app
    terrabrasilis_wfs   : função original do app
    ACTIVITY_COLORS     : dict de cores por atividade
    """
    st.markdown("## 📍 Análise Espacial por Projeto")
    st.info(
        "**🧞 Novidade:** Geometrias carregadas do GCS (Parquet) com AOI real. "
        "Visão geral usa BBox do catálogo (rápido). Projeto individual usa polígono real."
    )

    # ── Carrega catálogo leve para o seletor ──────────────────────────────
    with st.spinner("📋 Carregando catálogo..."):
        idx = load_index()

    if idx.empty:
        # fallback: usa df_all como catálogo
        idx = df_all[["resourceIdentifier", "resourceName_x", "state_Recode"]].drop_duplicates()

    idx["resourceIdentifier"] = idx["resourceIdentifier"].astype(str)
    df_all = df_all.copy()
    df_all["resourceIdentifier"] = df_all["resourceIdentifier"].astype(str)

    # Enriquece index com colunas extras de df_all
    extra_cols = [c for c in [
        "resourceName_x", "state_Recode", "vcsAFOLUActivity", "vcsProjectStatus",
        "vcsMethodology", "vcsCreditingPeriodTerm", "vcsAcresHectares", "description"
    ] if c in df_all.columns and c not in idx.columns]
    if extra_cols:
        idx = idx.merge(
            df_all[["resourceIdentifier"] + extra_cols].drop_duplicates("resourceIdentifier"),
            on="resourceIdentifier", how="left"
        )

    # ── Seletor de projeto ────────────────────────────────────────────────
    project_options = ["🌎 Visão Geral (Todos os Projetos)"] + [
        f"{row.get('resourceName_x', 'Sem nome')} — {row.get('state_Recode', 'N/A')}"
        for _, row in idx.iterrows()
    ]

    selected_project = st.selectbox(
        "📍 Selecione um projeto para análise:",
        options=project_options,
        key="gcp_project_selector"
    )

    is_overview = (selected_project == "🌎 Visão Geral (Todos os Projetos)")

    # ── Geometria ─────────────────────────────────────────────────────────
    if is_overview:
        # Visão geral: carrega apenas BBoxes do index (muito leve)
        gdf_plot, erros = carregar_geometrias_gcp(df_all, mode="overview")
        if gdf_plot.empty:
            st.warning("Nenhuma geometria disponível.")
            return
        centroid   = gdf_plot.geometry.centroid
        center     = [centroid.y.mean(), centroid.x.mean()]
        zoom_start = 5
        selected_gdf    = gdf_plot
        selected_rid    = None
        bbox_str        = None
        aoi_geojson     = None
    else:
        project_name = selected_project.split(" — ")[0]
        row_idx = idx[idx["resourceName_x"] == project_name]
        if row_idx.empty:
            st.warning("Projeto não encontrado no catálogo.")
            return
        selected_rid = str(row_idx.iloc[0]["resourceIdentifier"])

        with st.spinner(f"🗺️ Carregando geometria real: {project_name}..."):
            gdf_single = load_project_geometry(selected_rid)

        if gdf_single is None or gdf_single.empty:
            st.warning(f"⚠️ Geometria não disponível para {project_name}.")
            # Fallback para BBox do index
            bbox_cols = ["bbox_minx", "bbox_miny", "bbox_maxx", "bbox_maxy"]
            row_data = row_idx.iloc[0]
            if all(c in row_idx.columns for c in bbox_cols):
                from shapely.geometry import box as sbox
                geom_bbox = sbox(row_data.bbox_minx, row_data.bbox_miny,
                                 row_data.bbox_maxx, row_data.bbox_maxy)
                gdf_single = gpd.GeoDataFrame(
                    row_idx, geometry=[geom_bbox], crs="EPSG:4326"
                )
                st.caption("⚠️ Usando BBox aproximada (geometria real não disponível).")
            else:
                return

        # Mescla atributos
        meta_cols = [c for c in [
            "resourceName_x", "state_Recode", "vcsAFOLUActivity", "vcsProjectStatus",
            "vcsMethodology", "vcsCreditingPeriodTerm", "vcsAcresHectares", "description"
        ] if c in df_all.columns]
        meta_row = df_all[df_all["resourceIdentifier"] == selected_rid][meta_cols].drop_duplicates()
        if not meta_row.empty:
            for col in meta_cols:
                if col not in gdf_single.columns:
                    gdf_single[col] = meta_row.iloc[0][col]

        bounds     = gdf_single.total_bounds
        center     = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]
        zoom_start = 10
        selected_gdf = gdf_single
        bbox_str     = get_aoi_bbox_str(selected_rid)
        aoi_geojson  = get_aoi_geojson(selected_rid)   # ← AOI REAL (não BBox)

    # ── Tabs GFW / PRODES / DETER ─────────────────────────────────────────
    tab_gfw, tab_prodes, tab_deter_amz, tab_deter_cer = st.tabs([
        "🌳 GFW", "🔴 PRODES", "🟠 DETER Amazônia", "🟡 DETER Cerrado"
    ])

    # Constrói info panel reutilizável
    def _info_panel(gdf):
        row_proj = gdf.iloc[0]
        st.markdown("### 📋 Info")
        for label, col in [
            ("Projeto",     "resourceName_x"),
            ("ID",          "resourceIdentifier"),
            ("Estado",      "state_Recode"),
            ("Area",        "vcsAcresHectares"),
            ("Tipo",        "vcsAFOLUActivity"),
            ("Acreditação", "vcsCreditingPeriodTerm"),
            ("Protocolo",   "vcsMethodology"),
            ("Status",      "vcsProjectStatus"),
            ("Resumo",      "description"),
        ]:
            val = row_proj.get(col, "N/A")
            st.markdown(f"**{label}:** {val}")

    # ── TAB GFW ──────────────────────────────────────────────────────────
    with tab_gfw:
        col_mapa, col_info = st.columns([8, 2])

        with col_mapa:
            st.markdown("### 🗺️ Mapa GFW")
            c1, c2, c3 = st.columns(3)
            show_loss = c1.toggle("🔴 Tree Cover Loss", value=True,  key="gcp_toggle_loss")
            show_glad = c2.toggle("🟡 GLAD Alerts",     value=False, key="gcp_toggle_glad")
            show_radd = c3.toggle("🟠 RADD Alerts",     value=False, key="gcp_toggle_radd")

            m_gfw = folium.Map(location=center, zoom_start=zoom_start, tiles=None)
            folium.TileLayer("Esri.WorldImagery", name="Satélite", control=False).add_to(m_gfw)

            if not is_overview:
                b = selected_gdf.total_bounds
                folium.PolyLine(
                    [[b[1],b[0]],[b[1],b[2]],[b[3],b[2]],[b[3],b[0]],[b[1],b[0]]],
                    color="#00FFFF", weight=2, dash_array="6 4",
                    tooltip="Bounding Box (área de consulta WFS)"
                ).add_to(m_gfw)

            if show_loss:
                folium.TileLayer(
                    "https://tiles.globalforestwatch.org/umd_tree_cover_loss/v1.11/tcd_30/{z}/{x}/{y}.png",
                    name="Tree Cover Loss", attr="GFW", overlay=True, opacity=0.8
                ).add_to(m_gfw)
            if show_glad:
                folium.TileLayer(
                    "https://tiles.globalforestwatch.org/umd_glad_landsat_alerts/v20260320/default/{z}/{x}/{y}.png",
                    name="GLAD Alerts", attr="GFW", overlay=True, opacity=0.8
                ).add_to(m_gfw)
            if show_radd:
                folium.TileLayer(
                    "https://tiles.globalforestwatch.org/wur_radd_alerts/v20260315/default/{z}/{x}/{y}.png",
                    name="RADD Alerts", attr="GFW", overlay=True, opacity=0.8
                ).add_to(m_gfw)

            for _, row in selected_gdf.iterrows():
                try:
                    folium.GeoJson(
                        data=mapping(row["geometry"]),
                        style_function=lambda x: {
                            "fillColor": "transparent", "color": "#FF0000",
                            "weight": 3, "fillOpacity": 0.1, "dashArray": "5, 5"
                        }
                    ).add_to(m_gfw)
                except Exception:
                    pass

            if not is_overview:
                b = selected_gdf.total_bounds
                m_gfw.fit_bounds([[b[1], b[0]], [b[3], b[2]]])

            folium.LayerControl().add_to(m_gfw)
            st_folium(m_gfw, width=None, height=600, key="gcp_map_gfw")

        with col_info:
            if is_overview:
                st.info("💡 Selecione um projeto.")
                st.metric("Total de Projetos", f"{len(selected_gdf):,}")
            else:
                _info_panel(selected_gdf)
                if aoi_geojson:
                    st.success("✅ AOI real carregada")
                else:
                    st.warning("⚠️ Usando BBox aproximada")

        # Charts GFW (só projeto individual com AOI real)
        if not is_overview and aoi_geojson:
            st.divider()
            col_g1, col_g2, col_g3 = st.columns(3)

            with col_g1:
                st.markdown("### 📊 Tree Cover Loss")
                with st.spinner("Consultando GFW..."):
                    df_loss = gfw_tree_cover_loss(aoi_geojson, GFW_API_KEY)
                if df_loss.empty:
                    st.warning("Sem dados.")
                else:
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=df_loss["umd_tree_cover_loss__year"],
                        y=df_loss["loss_ha"], marker_color="#ff4444"
                    ))
                    fig.update_layout(height=300, template="plotly_white",
                                      margin=dict(t=10,b=40,l=40,r=10),
                                      xaxis_title="Ano", yaxis_title="ha",
                                      hovermode="x unified")
                    st.plotly_chart(fig, use_container_width=True)

            with col_g2:
                st.markdown("### 🟡 GLAD")
                with st.spinner("Consultando GLAD..."):
                    df_glad = gfw_glad_alerts(aoi_geojson, GFW_API_KEY)
                if df_glad.empty:
                    st.warning("Sem dados.")
                else:
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=df_glad["alert__year"],
                        y=df_glad["alert_count"], marker_color="#FFC300"
                    ))
                    fig.update_layout(height=300, template="plotly_white",
                                      margin=dict(t=10,b=40,l=40,r=10),
                                      xaxis_title="Ano", yaxis_title="Alertas",
                                      hovermode="x unified")
                    st.plotly_chart(fig, use_container_width=True)

            with col_g3:
                st.markdown("### 🟠 RADD")
                with st.spinner("Consultando RADD..."):
                    df_radd = gfw_radd_alerts(aoi_geojson, GFW_API_KEY)
                if df_radd.empty:
                    st.warning("Sem dados.")
                else:
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=df_radd["alert__year"],
                        y=df_radd["alert_count"], marker_color="#FF7900"
                    ))
                    fig.update_layout(height=300, template="plotly_white",
                                      margin=dict(t=10,b=40,l=40,r=10),
                                      xaxis_title="Ano", yaxis_title="Alertas",
                                      hovermode="x unified")
                    st.plotly_chart(fig, use_container_width=True)

        elif not is_overview and not aoi_geojson:
            st.info("ℹ️ AOI real não disponível para este projeto — GFW charts indisponíveis.")

    # ── TAB PRODES ────────────────────────────────────────────────────────
    with tab_prodes:
        col_mapa_p, col_info_p = st.columns([8, 2])

        with col_mapa_p:
            st.markdown("### 🗺️ Mapa PRODES")
            c1, c2 = st.columns(2)
            show_prodes_br  = c1.toggle("🔴 PRODES Brasil",    value=True,  key="gcp_prodes_br")
            show_prodes_amz = c2.toggle("🟥 PRODES Legal AMZ", value=False, key="gcp_prodes_amz")

            m_prodes = folium.Map(location=center, zoom_start=zoom_start, tiles=None)
            folium.TileLayer("Esri.WorldImagery", name="Satélite", control=False).add_to(m_prodes)

            if not is_overview:
                b = selected_gdf.total_bounds
                folium.PolyLine(
                    [[b[1],b[0]],[b[1],b[2]],[b[3],b[2]],[b[3],b[0]],[b[1],b[0]]],
                    color="#00FFFF", weight=2, dash_array="6 4"
                ).add_to(m_prodes)

            if show_prodes_br:
                folium.WmsTileLayer(
                    url="https://terrabrasilis.dpi.inpe.br/geoserver/prodes-brasil-nb/prodes_brasil/ows",
                    layers="prodes_brasil", fmt="image/png", transparent=True,
                    name="PRODES Brasil", overlay=True, opacity=0.8
                ).add_to(m_prodes)

            if show_prodes_amz:
                folium.WmsTileLayer(
                    url="https://terrabrasilis.dpi.inpe.br/geoserver/prodes-legal-amz/yearly_deforestation/ows",
                    layers="yearly_deforestation", fmt="image/png", transparent=True,
                    name="PRODES Legal AMZ", overlay=True, opacity=0.8
                ).add_to(m_prodes)

            for _, row in selected_gdf.iterrows():
                try:
                    folium.GeoJson(
                        data=mapping(row["geometry"]),
                        style_function=lambda x: {
                            "fillColor": "transparent", "color": "#49006a",
                            "weight": 3, "fillOpacity": 0.1, "dashArray": "5, 5"
                        }
                    ).add_to(m_prodes)
                except Exception:
                    pass

            if not is_overview:
                b = selected_gdf.total_bounds
                m_prodes.fit_bounds([[b[1], b[0]], [b[3], b[2]]])

            folium.LayerControl().add_to(m_prodes)
            st_folium(m_prodes, width=None, height=600, key="gcp_map_prodes")

        with col_info_p:
            if is_overview:
                st.info("💡 Selecione um projeto.")
            else:
                _info_panel(selected_gdf)

        if not is_overview and bbox_str:
            st.divider()
            col_g1, col_g2, col_g3 = st.columns(3)

            with col_g1:
                st.markdown("### 📊 PRODES Legal AMZ — Polígonos por Ano")
                with st.spinner("Consultando TerraBrasilis WFS..."):
                    df_prodes = terrabrasilis_wfs(
                        "https://terrabrasilis.dpi.inpe.br/geoserver/prodes-legal-amz/yearly_deforestation/ows",
                        "prodes-legal-amz:yearly_deforestation", bbox_str
                    )
                if df_prodes.empty:
                    st.warning("Sem dados PRODES para esta AOI.")
                else:
                    if "year" in df_prodes.columns:
                        df_prodes["year"] = pd.to_numeric(df_prodes["year"], errors="coerce")
                        df_prodes_year = df_prodes.groupby("year").size().reset_index(name="poligonos")
                        fig_p = go.Figure()
                        fig_p.add_trace(go.Bar(
                            x=df_prodes_year["year"], y=df_prodes_year["poligonos"],
                            marker_color="#C0392B"
                        ))
                        fig_p.update_layout(
                            xaxis_title="Ano", yaxis_title="Nº Polígonos",
                            height=300, template="plotly_white",
                            margin=dict(t=10,b=40,l=40,r=10), hovermode="x unified"
                        )
                        st.plotly_chart(fig_p, use_container_width=True)

                    with st.expander("📋 Tabela PRODES"):
                        st.dataframe(df_prodes, use_container_width=True, height=300)
                        st.download_button("⬇️ Download CSV",
                            df_prodes.to_csv(index=False).encode("utf-8"),
                            "prodes_aoi.csv", "text/csv")

            with col_g2:
                st.markdown("### 📊 Área (km²) por Ano")
                if not df_prodes.empty and "area_km" in df_prodes.columns:
                    df_prodes["area_km"] = pd.to_numeric(df_prodes["area_km"], errors="coerce")
                    df_prodes_area = df_prodes.groupby("year").agg(
                        area_total=("area_km", "sum")).reset_index()
                    fig_a = go.Figure()
                    fig_a.add_trace(go.Bar(
                        x=df_prodes_area["year"], y=df_prodes_area["area_total"],
                        marker_color="#922B21"
                    ))
                    fig_a.update_layout(
                        xaxis_title="Ano", yaxis_title="km²",
                        height=300, template="plotly_white",
                        margin=dict(t=10,b=40,l=40,r=10), hovermode="x unified"
                    )
                    st.plotly_chart(fig_a, use_container_width=True)
                else:
                    st.warning("Coluna `area_km` não encontrada.")

            with col_g3:
                st.markdown("### 📊 Classe de Desmatamento")
                if not df_prodes.empty and "class_name" in df_prodes.columns:
                    class_counts = df_prodes["class_name"].value_counts().reset_index()
                    class_counts.columns = ["Classe", "Count"]
                    fig_cls = go.Figure()
                    fig_cls.add_trace(go.Bar(
                        x=class_counts["Classe"], y=class_counts["Count"],
                        marker_color="#E74C3C"
                    ))
                    fig_cls.update_layout(
                        xaxis_title="Classe", yaxis_title="Nº Polígonos",
                        height=300, template="plotly_white",
                        margin=dict(t=10,b=40,l=40,r=10)
                    )
                    st.plotly_chart(fig_cls, use_container_width=True)

    # ── TAB DETER AMZ ────────────────────────────────────────────────────
    with tab_deter_amz:
        col_mapa_d, col_info_d = st.columns([8, 2])

        with col_mapa_d:
            st.markdown("### 🗺️ Mapa DETER Amazônia")
            m_deter = folium.Map(location=center, zoom_start=zoom_start, tiles=None)
            folium.TileLayer("Esri.WorldImagery", name="Satélite", control=False).add_to(m_deter)

            if not is_overview:
                b = selected_gdf.total_bounds
                folium.PolyLine(
                    [[b[1],b[0]],[b[1],b[2]],[b[3],b[2]],[b[3],b[0]],[b[1],b[0]]],
                    color="#00FFFF", weight=2, dash_array="6 4"
                ).add_to(m_deter)

            folium.WmsTileLayer(
                url="https://terrabrasilis.dpi.inpe.br/geoserver/deter-amz/deter_amz/ows",
                layers="deter_amz", fmt="image/png", transparent=True,
                name="DETER AMZ", overlay=True, opacity=0.8
            ).add_to(m_deter)

            for _, row in selected_gdf.iterrows():
                try:
                    folium.GeoJson(
                        data=mapping(row["geometry"]),
                        style_function=lambda x: {
                            "fillColor": "transparent", "color": "#FF6600",
                            "weight": 3, "fillOpacity": 0.1, "dashArray": "5, 5"
                        }
                    ).add_to(m_deter)
                except Exception:
                    pass

            if not is_overview:
                b = selected_gdf.total_bounds
                m_deter.fit_bounds([[b[1], b[0]], [b[3], b[2]]])

            folium.LayerControl().add_to(m_deter)
            st_folium(m_deter, width=None, height=600, key="gcp_map_deter_amz")

        with col_info_d:
            if is_overview:
                st.info("💡 Selecione um projeto.")
            else:
                _info_panel(selected_gdf)

        if not is_overview and bbox_str:
            st.divider()
            st.markdown("### 📊 DETER AMZ — Alertas na AOI")
            with st.spinner("Consultando TerraBrasilis WFS..."):
                df_deter = terrabrasilis_wfs(
                    "https://terrabrasilis.dpi.inpe.br/geoserver/deter-amz/deter_amz/ows",
                    "deter-amz:deter_amz", bbox_str
                )

            if df_deter.empty:
                st.warning("Sem alertas DETER para esta AOI.")
            else:
                col_g1, col_g2 = st.columns([2, 3])

                with col_g1:
                    st.markdown("### 📊 Alertas por Ano")
                    if "view_date" in df_deter.columns:
                        df_deter["year"] = pd.to_datetime(df_deter["view_date"], errors="coerce").dt.year
                        df_deter_year = df_deter.groupby("year").agg(
                            alertas=("year","count"), area_km2=("areauckm","sum")
                        ).reset_index()
                        fig_d = go.Figure()
                        fig_d.add_trace(go.Bar(
                            x=df_deter_year["year"], y=df_deter_year["alertas"],
                            marker_color="#E67E22"
                        ))
                        fig_d.update_layout(
                            xaxis_title="Ano", yaxis_title="Nº Alertas",
                            height=300, template="plotly_white",
                            margin=dict(t=10,b=40,l=40,r=10), hovermode="x unified"
                        )
                        st.plotly_chart(fig_d, use_container_width=True)

                    st.markdown("### 📊 Classe de Alerta")
                    if "classname" in df_deter.columns:
                        class_d = df_deter["classname"].value_counts().reset_index()
                        class_d.columns = ["Classe","Count"]
                        fig_cd = go.Figure()
                        fig_cd.add_trace(go.Bar(
                            x=class_d["Classe"], y=class_d["Count"], marker_color="#D35400"
                        ))
                        fig_cd.update_layout(
                            xaxis_title="Classe", yaxis_title="Alertas",
                            height=300, template="plotly_white",
                            margin=dict(t=10,b=40,l=40,r=10)
                        )
                        st.plotly_chart(fig_cd, use_container_width=True)

                with col_g2:
                    st.markdown("### 📊 Área por Classe ao Longo do Tempo (ha)")
                    if all(c in df_deter.columns for c in ["classname","areamunkm","year"]):
                        df_deter["areamunkm"] = pd.to_numeric(df_deter["areamunkm"], errors="coerce") * 100
                        df_area_linha = df_deter.groupby(["year","classname"]).agg(
                            area_total=("areamunkm","sum")
                        ).reset_index()

                        DETER_COLORS = {
                            "CICATRIZ_DE_QUEIMADA": "#d7191c",
                            "CORTE_SELETIVO":       "#868686",
                            "CS_DESORDENADO":       "#db83ff",
                            "CS_GEOMETRICO":        "#ff7e00",
                            "DEGRADACAO":           "#ffffbf",
                            "DESMATAMENTO_CR":      "#8a5f4b",
                            "DESMATAMENTO_VEG":     "#abdda4",
                            "MINERACAO":            "#4223e5",
                        }

                        fig_da = go.Figure()
                        for classe in sorted(df_area_linha["classname"].unique()):
                            df_cls = df_area_linha[df_area_linha["classname"] == classe]
                            cor = DETER_COLORS.get(classe.upper().replace(" ","_"), "#999999")
                            fig_da.add_trace(go.Scatter(
                                x=df_cls["year"], y=df_cls["area_total"],
                                mode="lines+markers", name=classe,
                                line=dict(width=2, color=cor), marker=dict(size=6, color=cor)
                            ))

                        fig_da.update_layout(
                            xaxis_title="Ano", yaxis_title="ha",
                            height=640, template="plotly_white",
                            margin=dict(t=10,b=40,l=40,r=10), hovermode="x unified",
                            legend=dict(font=dict(size=10), orientation="v")
                        )
                        st.plotly_chart(fig_da, use_container_width=True)

                with st.expander("📋 Tabela DETER AMZ"):
                    st.dataframe(df_deter, use_container_width=True, height=300)
                    st.download_button("⬇️ Download CSV",
                        df_deter.to_csv(index=False).encode("utf-8"),
                        "deter_amz_aoi.csv", "text/csv")

    # ── TAB DETER CERRADO ─────────────────────────────────────────────────
    with tab_deter_cer:
        col_mapa_c, col_info_c = st.columns([8, 2])

        with col_mapa_c:
            st.markdown("### 🗺️ Mapa DETER Cerrado")
            m_cer = folium.Map(location=center, zoom_start=zoom_start, tiles=None)
            folium.TileLayer("Esri.WorldImagery", name="Satélite", control=False).add_to(m_cer)

            if not is_overview:
                b = selected_gdf.total_bounds
                folium.PolyLine(
                    [[b[1],b[0]],[b[1],b[2]],[b[3],b[2]],[b[3],b[0]],[b[1],b[0]]],
                    color="#00FFFF", weight=2, dash_array="6 4"
                ).add_to(m_cer)

            folium.WmsTileLayer(
                url="https://terrabrasilis.dpi.inpe.br/geoserver/deter-cerrado-nb/deter_cerrado/ows",
                layers="deter_cerrado", fmt="image/png", transparent=True,
                name="DETER Cerrado", overlay=True, opacity=0.8
            ).add_to(m_cer)

            for _, row in selected_gdf.iterrows():
                try:
                    folium.GeoJson(
                        data=mapping(row["geometry"]),
                        style_function=lambda x: {
                            "fillColor": "transparent", "color": "#F1C40F",
                            "weight": 3, "fillOpacity": 0.1, "dashArray": "5, 5"
                        }
                    ).add_to(m_cer)
                except Exception:
                    pass

            if not is_overview:
                b = selected_gdf.total_bounds
                m_cer.fit_bounds([[b[1], b[0]], [b[3], b[2]]])

            folium.LayerControl().add_to(m_cer)
            st_folium(m_cer, width=None, height=600, key="gcp_map_deter_cer")

        with col_info_c:
            if is_overview:
                st.info("💡 Selecione um projeto.")
            else:
                _info_panel(selected_gdf)

        if not is_overview and bbox_str:
            st.divider()
            st.markdown("### 📊 DETER Cerrado — Alertas na AOI")
            with st.spinner("Consultando TerraBrasilis WFS..."):
                df_cer = terrabrasilis_wfs(
                    "https://terrabrasilis.dpi.inpe.br/geoserver/deter-cerrado-nb/deter_cerrado/ows",
                    "deter-cerrado-nb:deter_cerrado", bbox_str
                )
            if df_cer.empty:
                st.warning("Sem alertas DETER Cerrado para esta AOI.")
            else:
                st.dataframe(df_cer, use_container_width=True, height=300)
                st.download_button("⬇️ Download CSV",
                    df_cer.to_csv(index=False).encode("utf-8"),
                    "deter_cerrado_aoi.csv", "text/csv")
