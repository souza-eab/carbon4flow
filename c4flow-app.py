#!pip install pandas rapidfuzz streamlit pandas requests io plotly datetime folium numpy scipy


import streamlit as st
import pandas as pd
import requests
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import folium
from folium.plugins import MarkerCluster, HeatMap
from streamlit_folium import st_folium
import numpy as np
from scipy import stats
import os
import geopandas as gpd
from shapely.geometry import mapping
from shapely.errors import TopologicalError

# =====================================
# CONFIGURAÇÃO DA PÁGINA
# =====================================
st.set_page_config(
    page_title="Carbon4Flow",
    page_icon="🌎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================
# FUNÇÕES AUXILIARES
# =====================================

@st.cache_data(ttl=3600, show_spinner=True)
def load_parquet_from_gdrive(file_id: str) -> pd.DataFrame:
    try:
        download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        response = requests.get(download_url, timeout=60)
        response.raise_for_status()
        df = pd.read_parquet(BytesIO(response.content))
        return df
    except Exception as e:
        st.error(f"❌ Erro ao carregar dados: {str(e)}")
        return None

def clean_numeric_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    df_clean = df.copy()
    for col in columns:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    return df_clean

def prepare_map_data(df: pd.DataFrame) -> pd.DataFrame:
    df_map = df.copy()
    coord_cols = ["new_latitude", "new_longitude", "latitude", "longitude"]
    df_map = clean_numeric_columns(df_map, coord_cols)
    if "new_latitude" in df_map.columns and "new_longitude" in df_map.columns:
        df_map = df_map.dropna(subset=["new_latitude", "new_longitude"])
    else:
        df_map = df_map.dropna(subset=["latitude", "longitude"])
        df_map.rename(columns={"latitude": "new_latitude", "longitude": "new_longitude"}, inplace=True)
    return df_map

@st.cache_data(show_spinner=False)
def calcular_intervalo_confianca(df_grouped, confidence=0.95):
    result = []
    for name, group in df_grouped.groupby('resourceName_x'):
        values = group['totalVintageQuantity'].dropna()
        n = len(values)
        if n >= 2:
            mean = values.mean()
            se = stats.sem(values)
            h = se * stats.t.ppf((1 + confidence) / 2., n - 1)
            result.append({
                'resourceName_x': name,
                'Mean': mean,
                'IC': h,
                'IC_Menos': mean - h,
                'IC_Mais': mean + h
            })
    return pd.DataFrame(result)

@st.cache_data(show_spinner=False)
def analise_vcu_por_vintage(df_full):
    df = df_full.copy()
    required_cols = ['resourceName_x', 'totalVintageQuantity', 'quantity', 'Vintage']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"❌ Colunas ausentes: {', '.join(missing_cols)}")
        return pd.DataFrame()
    df = df.dropna(subset=['resourceName_x', 'Vintage', 'totalVintageQuantity'])
    df['totalVintageQuantity'] = pd.to_numeric(df['totalVintageQuantity'], errors='coerce')
    df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
    if 'retiredCancelled' not in df.columns:
        df['retiredCancelled'] = False
    group_cols = ['state_Recode', 'resourceName_x', 'Vintage', 'protocol',
                  'vcsProjectStatus', 'vcsEstimatedAnnualEmissionReductions']
    group_cols = [col for col in group_cols if col in df.columns]
    estatisticas = df.groupby(group_cols, dropna=False).agg(
        TotalVintageQuantity=('totalVintageQuantity', 'first'),
        SumQuantity=('quantity', 'sum'),
        Sum_Retired=('quantity', lambda x: x[df.loc[x.index, 'retiredCancelled'] == True].sum()),
        Sum_Active=('quantity', lambda x: x[df.loc[x.index, 'retiredCancelled'] == False].sum())
    ).reset_index()
    ic_df = calcular_intervalo_confianca(df)
    estatisticas = estatisticas.merge(ic_df, on='resourceName_x', how='left')
    estatisticas['Ano_Periodo'] = estatisticas['Vintage'].apply(
        lambda x: (
            f"{x.split(' e ')[0][:4]}-{x.split(' e ')[1][:4]}"
            if isinstance(x, str) and ' e ' in x
            else str(x)[:4] if pd.notna(x) else 'N/A'
        )
    )
    if 'vcsEstimatedAnnualEmissionReductions' in estatisticas.columns:
        estatisticas.rename(columns={'vcsEstimatedAnnualEmissionReductions': 'EAER'}, inplace=True)
    return estatisticas

# =====================================
# FUNÇÕES GFW — escopo global
# =====================================

@st.cache_data(show_spinner=True)
def carregar_geometrias(df_all, kml_dir: str):
    lista_gdfs = []
    erros = []
    for file in os.listdir(kml_dir):
        if file.lower().endswith(".kml"):
            resource_id = file.split("_")[0]
            try:
                gdf = gpd.read_file(os.path.join(kml_dir, file), driver="KML")
                gdf["resourceIdentifier"] = str(resource_id)
                lista_gdfs.append(gdf)
            except Exception as e:
                erros.append((file, str(e)))
    if not lista_gdfs:
        return gpd.GeoDataFrame(), erros
    gdf_all = pd.concat(lista_gdfs, ignore_index=True)
    if gdf_all.crs is None:
        gdf_all.set_crs("EPSG:4326", inplace=True)
    else:
        gdf_all = gdf_all.to_crs("EPSG:4326")
    df_all["resourceIdentifier"] = df_all["resourceIdentifier"].astype(str)
    gdf_all["resourceIdentifier"] = gdf_all["resourceIdentifier"].astype(str)
    gdf_all["geometry"] = gdf_all["geometry"].buffer(0)
    try:
        gdf_all = gdf_all.dissolve(by="resourceIdentifier")
    except TopologicalError:
        gdf_all["geometry"] = gdf_all["geometry"].buffer(0)
        gdf_all = gdf_all.dissolve(by="resourceIdentifier")
    gdf_all = gdf_all.merge(df_all, on="resourceIdentifier", how="left")
    return gdf_all, erros

@st.cache_data(show_spinner=False)
def gfw_tree_cover_loss(geojson, api_key):
    url = "https://data-api.globalforestwatch.org/dataset/umd_tree_cover_loss/v1.11/query"
    headers = {"x-api-key": api_key.strip(), "Content-Type": "application/json"}
    payload = {
        "geometry": geojson,
        "sql": "SELECT umd_tree_cover_loss__year, SUM(umd_tree_cover_loss__ha) as loss_ha FROM data GROUP BY umd_tree_cover_loss__year ORDER BY umd_tree_cover_loss__year"
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=30)
        if r.status_code == 200:
            return pd.DataFrame(r.json().get("data", []))
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def gfw_glad_alerts(geojson, api_key):
    url = "https://data-api.globalforestwatch.org/dataset/umd_glad_landsat_alerts/v20260320/query"
    headers = {"x-api-key": api_key.strip(), "Content-Type": "application/json"}
    payload = {
        "geometry": geojson,
        "sql": "SELECT umd_glad_landsat_alerts__date, umd_glad_landsat_alerts__confidence FROM data LIMIT 50000"
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        if r.status_code == 200:
            data = r.json().get("data", [])
            if not data:
                return pd.DataFrame()
            df = pd.DataFrame(data)
            df['alert__year'] = pd.to_datetime(df['umd_glad_landsat_alerts__date']).dt.year
            return df.groupby('alert__year').agg(alert_count=('alert__year', 'count')).reset_index()
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def gfw_radd_alerts(geojson, api_key):
    url = "https://data-api.globalforestwatch.org/dataset/wur_radd_alerts/v20260315/query"
    headers = {"x-api-key": api_key.strip(), "Content-Type": "application/json"}
    payload = {
        "geometry": geojson,
        "sql": "SELECT wur_radd_alerts__date, wur_radd_alerts__confidence FROM data LIMIT 50000"
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        if r.status_code == 200:
            data = r.json().get("data", [])
            if not data:
                return pd.DataFrame()
            df = pd.DataFrame(data)
            df['alert__year'] = pd.to_datetime(df['wur_radd_alerts__date']).dt.year
            return df.groupby('alert__year').agg(alert_count=('alert__year', 'count')).reset_index()
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

# =====================================
# FUNÇÕES TERRABRASILIS — escopo global
# =====================================

@st.cache_data(show_spinner=False)
def terrabrasilis_wfs(url, type_name, bbox, max_features=5000):
    """Consulta WFS do TerraBrasilis filtrado por bbox."""
    try:
        r = requests.get(url, params={
            "SERVICE":      "WFS",
            "VERSION":      "1.0.0",
            "REQUEST":      "GetFeature",
            "typeName":     type_name,
            "BBOX":         bbox,
            "outputFormat": "application/json",
            "maxFeatures":  str(max_features)
        }, timeout=60)
        if r.status_code == 200:
            data = r.json()
            feats = data.get('features', [])
            if feats:
                return pd.DataFrame([f['properties'] for f in feats])
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

# =====================================
# FUNÇÕES MAPBIOMAS — escopo global
# =====================================

@st.cache_data(ttl=3600, show_spinner=False)
def mapbiomas_get_token(email, password):
    mutation = """
    mutation($email: String!, $password: String!) {
      signIn(email: $email, password: $password) {
        token
      }
    }
    """
    try:
        r = requests.post(
            "https://plataforma.alerta.mapbiomas.org/api/v2/graphql",
            json={"query": mutation, "variables": {"email": email, "password": password}},
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        return r.json()['data']['signIn']['token']
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def mapbiomas_alerts(bbox, token, start_date="2019-01-01", end_date="2024-12-31"):
    query = """
    query($boundingBox: [Float!], $startDate: BaseDate, $endDate: BaseDate, $limit: Int, $page: Int) {
      alerts(
        boundingBox: $boundingBox
        startDate: $startDate
        endDate: $endDate
        limit: $limit
        page: $page
      ) {
        metadata {
          totalCount
          totalPages
          currentPage
          limitValue
        }
        summary {
          total
          area
          alertsByYear { year value }
          deforestationAreaByYear { year value }
        }
        collection {
          alertCode
          areaHa
          detectedAt
          publishedAt
          sources
          deforestationClasses
          statusName
          crossedBiomes
          crossedStates
        }
      }
    }
    """
    variables = {
        "boundingBox": bbox,
        "startDate": start_date,
        "endDate": end_date,
        "limit": 1000,
        "page": 1
    }
    try:
        r = requests.post(
            "https://plataforma.alerta.mapbiomas.org/api/v2/graphql",
            json={"query": query, "variables": variables},
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
            timeout=60
        )
        if r.status_code == 200 and r.json().get('data'):
            return r.json()['data']['alerts']
        return None
    except Exception:
        return None

# =====================================
# FUNÇÕES PRODES GPKG — escopo global
# =====================================

#@st.cache_data(ttl=86400, show_spinner=False)
#def baixar_gpkg_drive(file_id: str, dest_path: str) -> str:
#    """Baixa GPKG do Google Drive para arquivo temporário. Cache de 24h."""
#    if os.path.exists(dest_path):
#        return dest_path
#    try:
#        # Tenta download direto
#        url = f"https://drive.google.com/uc?export=download&id={file_id}"
#        session = requests.Session()
#        r = session.get(url, stream=True, timeout=120)
#
#        # Arquivo grande — precisa confirmar o aviso do Drive
#        for key, value in r.cookies.items():
#            if 'download_warning' in key:
#                params = {'id': file_id, 'confirm': value}
#                r = session.get(url, params=params, stream=True, timeout=120)
#                break
#
#        with open(dest_path, 'wb') as f:
#            for chunk in r.iter_content(chunk_size=32768):
#                if chunk:
#                    f.write(chunk)
#        return dest_path
#    except Exception as e:
#        return None


#@st.cache_data(show_spinner=False)
#def carregar_prodes_bbox(file_id: str, bbox_tuple: tuple) -> pd.DataFrame:
#    """Carrega PRODES layer yearly_deforestation filtrado por bbox."""
#    import tempfile
#
#    dest_path = os.path.join(tempfile.gettempdir(), "prodes_amazonia_legal.gpkg")
#
#    with st.spinner("📥 Baixando GPKG PRODES (primeira vez pode demorar)..."):
#        gpkg_path = baixar_gpkg_drive(file_id, dest_path)
#
#    if not gpkg_path:
#        return gpd.GeoDataFrame()
#
#    try:
#        gdf = gpd.read_file(
#            gpkg_path,
#            layer="yearly_deforestation",
#            bbox=bbox_tuple  # (minx, miny, maxx, maxy)
#        )
#        return gdf
#    except Exception as e:
#        return gpd.GeoDataFrame()

# Na função carregar_prodes_bbox, forçar limpeza após uso
#@st.cache_data(show_spinner=False)
#def carregar_prodes_bbox(file_id: str, bbox_tuple: tuple):
#    import tempfile, gc
#    dest_path = os.path.join(tempfile.gettempdir(), "prodes_amazonia_legal.gpkg")
#    gpkg_path = baixar_gpkg_drive(file_id, dest_path)
#    if not gpkg_path:
#        return gpd.GeoDataFrame()
#    try:
#        gdf = gpd.read_file(gpkg_path, layer="yearly_deforestation", bbox=bbox_tuple)
#        # Manter só colunas necessárias — reduz memória
#        cols_keep = ['year', 'area_km', 'class_name', 'main_class', 'state', 'geometry']
#        cols_keep = [c for c in cols_keep if c in gdf.columns]
#        gdf = gdf[cols_keep]
#        gc.collect()
#        return gdf
#    except Exception:
#        return gpd.GeoDataFrame()    
    
# =====================================
# CONFIGURAÇÃO DE CORES E ESTILOS
# =====================================

ACTIVITY_COLORS = {
    'REDD': "#f79c3c",
    'IFM': "#3b8fbf",
    'ARR': "#78cafe",
    'ACoGS': "#ffffcc",
    'ACoGS; REDD': "#c7e9b4",
    'ALM': "#7fcdbb",
    'IFM; REDD': "#225ea8",
    'Unknown': "#808080"
}

STATUS_COLORS = {
    'Registered': "#2ecc71",
    'Under Validation': "#f39c12",
    'Under Development': "#3498db",
    'Inactive': "#95a5a6",
    'Unknown': "#808080"
}

# =====================================
# SIDEBAR
# =====================================

st.sidebar.title("⚙️ Configurações")

file_id_all = st.sidebar.text_input(
    "ID - Todos os Projetos",
    value="13ijts4CMnyOV9rVdQ6tXP7-qIPfPa0Yb",
    help="ID do arquivo Parquet no Google Drive"
)

file_id_credit = st.sidebar.text_input(
    "ID - Projetos com Créditos",
    value="13ZlnQjYHsbs57A1rj92brWlGMnBGlU3P",
    help="ID do arquivo Parquet no Google Drive"
)

if st.sidebar.button("🔄 Recarregar Dados", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

st.sidebar.divider()

st.markdown("""
    <style>
        [data-testid="stSidebar"] input,
        [data-testid="stSidebar"] button { display: none !important; }
    </style>
""", unsafe_allow_html=True)

# =====================================
# CARREGAMENTO DE DADOS
# =====================================

with st.spinner("📥 Carregando dados..."):
    df_all = load_parquet_from_gdrive(file_id_all)
    df_credit = load_parquet_from_gdrive(file_id_credit)

if "Vintage" not in df_credit.columns:
    if {"vintageStart", "vintageEnd"}.issubset(df_credit.columns):
        df_credit["Vintage"] = (
            df_credit["vintageStart"].astype(str).str[:4] + " e " +
            df_credit["vintageEnd"].astype(str).str[:4]
        )
    else:
        df_credit["Vintage"] = np.nan

if df_all is None or df_credit is None:
    st.error("❌ Não foi possível carregar os dados. Verifique os IDs dos arquivos.")
    st.stop()

st.sidebar.success("✅ Dados carregados com sucesso!")
st.sidebar.metric("Total de Projetos", f"{len(df_all):,}")
st.sidebar.metric("Projetos com VCUs", f"{df_credit['resourceName_x'].nunique():,}" if 'resourceName_x' in df_credit.columns else f"{len(df_credit):,}")
st.sidebar.caption(f"Última atualização: {datetime.now().strftime('%d/%m/%Y %H:%M')}")

# =====================================
# HEADER PRINCIPAL
# =====================================

st.title("🌎 Carbon4Flow")
st.markdown("""
Dashboard interativo para estimativas sobre Projetos de Carbono. Dados: Verra.  
Dev: Edriano Souza. Reporting Issues: edriano.souza@ipam.org.br
""")

if 'selected_state_overview' not in st.session_state:
    st.session_state.selected_state_overview = None

# =====================================
# ABAS PRINCIPAIS
# =====================================

tabs = st.tabs([
    "📊 Visão Geral",
    "🌎 Mapa - Todos os Projetos BR",
    "💰 Mapa - Com lastro de Créditos/Vendas",
    "📈 Análises de Vintage",
    "📖 Storytelling",
    "📁 Dados Brutos"
])

# =====================================
# ABA 1: VISÃO GERAL
# =====================================

with tabs[0]:
    st.header("📊 Visão Geral dos Projetos")

    df_overview = df_all.copy()
    if st.session_state.selected_state_overview:
        df_overview = df_overview[df_overview["state_Recode"] == st.session_state.selected_state_overview]
        st.info(f"🔍 Filtrando por: **{st.session_state.selected_state_overview}**")
        if st.button("🔄 Limpar Filtro"):
            st.session_state.selected_state_overview = None
            st.rerun()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total de Projetos", f"{len(df_overview):,}")
    with col2:
        if "vcsProjectStatus" in df_overview.columns:
            active_count = len(df_overview[df_overview["vcsProjectStatus"] == "Registered"])
            st.metric("Projetos Registrados", f"{active_count:,}")
    with col3:
        credit_count = len(df_credit[df_credit["resourceName_x"].isin(df_overview["resourceName_x"])]) if st.session_state.selected_state_overview else len(df_credit)
        st.metric("N obs", f"{credit_count:,}")
    with col4:
        if "vcsAFOLUActivity" in df_overview.columns:
            st.metric("Projetos AFOLU", f"{df_overview['vcsAFOLUActivity'].notna().sum():,}")

    st.divider()

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("📋 Distribuição por Status")
        if "vcsProjectStatus" in df_overview.columns:
            status_counts = df_overview["vcsProjectStatus"].value_counts().reset_index()
            status_counts.columns = ["Status", "Quantidade"]
            fig_status = px.bar(status_counts, x="Status", y="Quantidade", color="Status",
                                color_discrete_map=STATUS_COLORS, text="Quantidade")
            fig_status.update_traces(textposition='outside')
            fig_status.update_layout(showlegend=False, height=600, xaxis_title="", yaxis_title="Número de Projetos")
            st.plotly_chart(fig_status, use_container_width=True)

    with col_right:
        st.subheader("🌳 Distribuição por Atividade AFOLU")
        if "vcsAFOLUActivity" in df_overview.columns:
            activity_counts = df_overview["vcsAFOLUActivity"].value_counts().reset_index()
            activity_counts.columns = ["Atividade", "Quantidade"]
            fig_activity = px.pie(activity_counts, names="Atividade", values="Quantidade",
                                  color="Atividade", color_discrete_map=ACTIVITY_COLORS, hole=0.4)
            fig_activity.update_traces(textposition='inside', textinfo='percent+label')
            fig_activity.update_layout(height=500)
            st.plotly_chart(fig_activity, use_container_width=True)

    st.divider()
    st.subheader("🗺️ Distribuição por Estado")

    if "state_Recode" in df_overview.columns:
        state_counts = df_overview["state_Recode"].value_counts().head(10).reset_index()
        state_counts.columns = ["Estado", "Quantidade"]
        fig_states = px.bar(state_counts, x="Quantidade", y="Estado", orientation='h',
                            text="Quantidade", color="Quantidade", color_continuous_scale="Viridis")
        fig_states.update_traces(textposition='outside')
        fig_states.update_layout(height=400, showlegend=False, xaxis_title="Número de Projetos", yaxis_title="")
        st.plotly_chart(fig_states, use_container_width=True, on_select="rerun", key="state_chart")

# =====================================
# FUNÇÃO PARA CRIAR MAPAS
# =====================================

def create_interactive_map(df: pd.DataFrame, title: str, map_key: str):
    st.header(title)
    df_map = prepare_map_data(df)

    col_filter1, col_filter2, col_filter3 = st.columns(3)

    with col_filter1:
        if "state_Recode" in df_map.columns:
            states = ["Todos"] + sorted(df_map["state_Recode"].dropna().unique().tolist())
            selected_state = st.selectbox("Estado:", states, key=f"state_{map_key}")
            if selected_state != "Todos":
                df_map = df_map[df_map["state_Recode"] == selected_state]

    with col_filter2:
        show_afolu = st.checkbox("Apenas AFOLU", value=True, key=f"afolu_{map_key}")
        if show_afolu and "vcsAFOLUActivity" in df_map.columns:
            df_map = df_map[df_map["vcsAFOLUActivity"].notna()]

    with col_filter3:
        map_type = st.selectbox("Tipo de Mapa:", ["Pontos", "Heatmap", "Clusters"], key=f"maptype_{map_key}")

    with st.expander("🔍 Filtros Avançados"):
        col_adv1, col_adv2 = st.columns(2)
        with col_adv1:
            if "protocolSubCategories" in df_map.columns:
                categories = sorted(df_map["protocolSubCategories"].dropna().unique().tolist())
                selected_cat = st.multiselect("Protocol Sub-Categories:", options=categories, default=categories, key=f"cat_{map_key}")
                if selected_cat:
                    df_map = df_map[df_map["protocolSubCategories"].isin(selected_cat)]
        with col_adv2:
            if "vcsProjectStatus" in df_map.columns:
                statuses = sorted(df_map["vcsProjectStatus"].dropna().unique().tolist())
                selected_status = st.multiselect("Status:", options=statuses, default=statuses, key=f"status_{map_key}")
                if selected_status:
                    df_map = df_map[df_map["vcsProjectStatus"].isin(selected_status)]

    st.info(f"📍 **{len(df_map):,}** projetos sendo exibidos")

    if len(df_map) == 0:
        st.warning("⚠️ Nenhum projeto encontrado com os filtros selecionados.")
        return

    center = [df_map["new_latitude"].mean(), df_map["new_longitude"].mean()]
    m = folium.Map(location=center, zoom_start=4, tiles="CartoDB dark_matter")

    if map_type == "Clusters":
        marker_cluster = MarkerCluster().add_to(m)
        for idx, row in df_map.iterrows():
            lat, lon = row["new_latitude"], row["new_longitude"]
            activity = row.get("vcsAFOLUActivity", "Unknown")
            color = ACTIVITY_COLORS.get(activity, "#808080")
            popup_html = f"""
            <div style="font-family: Arial; font-size: 12px; width: 250px;">
                <h4 style="margin: 0 0 10px 0;">{row.get('resourceName_x', 'N/A')}</h4>
                <b>Status:</b> {row.get('vcsProjectStatus', 'N/A')}<br>
                <b>Estado:</b> {row.get('state_Recode', 'N/A')}<br>
                <b>EAER:</b> {row.get('vcsEstimatedAnnualEmissionReductions', 'N/A')}
            </div>
            """
            folium.CircleMarker(location=[lat, lon], radius=6, color=color, fill=True,
                                fill_color=color, fill_opacity=0.7,
                                popup=folium.Popup(popup_html, max_width=300)).add_to(marker_cluster)
    elif map_type == "Heatmap":
        HeatMap([[row["new_latitude"], row["new_longitude"]] for _, row in df_map.iterrows()], radius=15).add_to(m)
    else:
        for idx, row in df_map.iterrows():
            activity = row.get("vcsAFOLUActivity", "Unknown")
            color = ACTIVITY_COLORS.get(activity, "#808080")
            folium.CircleMarker(location=[row["new_latitude"], row["new_longitude"]], radius=5,
                                color=color, fill=True, fill_color=color, fill_opacity=0.6).add_to(m)

    st_folium(m, width=None, height=600, key=f"map_{map_key}")

    if map_type != "Heatmap":
        with st.expander("🎨 Legenda"):
            for activity, color in ACTIVITY_COLORS.items():
                if activity != "Unknown":
                    st.markdown(
                        f"<span style='display:inline-block;width:20px;height:20px;"
                        f"background:{color};margin-right:10px;border:1px solid #000;'></span><b>{activity}</b>",
                        unsafe_allow_html=True
                    )

# =====================================
# ABA 2: MAPA - TODOS OS PROJETOS
# =====================================

with tabs[1]:
    create_interactive_map(df_all, "🌎 Mapa - Todos os Projetos BR", "all")

# =====================================
# ABA 3: MAPA - PROJETOS COM CRÉDITOS
# =====================================

with tabs[2]:
    st.header("💰 Mapa - Projetos com Créditos")
    df_credit_unique = df_credit.groupby('resourceName_x').first().reset_index()
    st.info(f"📊 Exibindo **{len(df_credit_unique):,}** projetos únicos (de {len(df_credit):,} registros totais)")
    create_interactive_map(df_credit_unique, "Projetos Únicos com Créditos", "credit_unique")

# =====================================
# ABA 4: ANÁLISE DE VINTAGE
# =====================================

with tabs[3]:
    st.header("📈 Análise de VCUs por Vintage")

    with st.spinner("🔄 Processando análise de vintage..."):
        estatisticas = analise_vcu_por_vintage(df_credit)

    if estatisticas.empty:
        st.warning("⚠️ Nenhum dado disponível para análise de Vintage.")
    else:
        col_f1, col_f2 = st.columns(2)

        with col_f1:
            if 'state_Recode' in estatisticas.columns:
                estados = sorted(estatisticas['state_Recode'].dropna().unique())
                if len(estados) > 0:
                    estado_sel = st.selectbox("📍 Selecione o Estado:", estados, key="vintage_state")
                    dados_estado = estatisticas[estatisticas['state_Recode'] == estado_sel]
                else:
                    st.warning("Nenhum estado disponível")
                    dados_estado = pd.DataFrame()
            else:
                dados_estado = estatisticas

        with col_f2:
            if not dados_estado.empty and 'resourceName_x' in dados_estado.columns:
                projetos = sorted(dados_estado['resourceName_x'].dropna().unique())
                if len(projetos) > 0:
                    projeto_sel = st.selectbox("🏢 Selecione o Projeto:", projetos, key="vintage_project")
                    df_proj = dados_estado[dados_estado['resourceName_x'] == projeto_sel]
                else:
                    st.warning("Nenhum projeto disponível para este estado")
                    df_proj = pd.DataFrame()
            else:
                df_proj = pd.DataFrame()

        if df_proj.empty:
            st.info("Selecione um estado e projeto para visualizar a análise")
        else:
            col_m1, col_m3, col_m4 = st.columns(3)
            with col_m1:
                if 'Mean' in df_proj.columns:
                    st.metric("Média VCUs", f"{df_proj['Mean'].iloc[0]:,.0f} ± {df_proj['IC_Mais'].iloc[0] - df_proj['Mean'].iloc[0]:,.0f}")
            with col_m3:
                if 'protocol' in df_proj.columns:
                    st.metric("Protocolo", df_proj['protocol'].iloc[0], delta_color="off")
            with col_m4:
                if 'vcsProjectStatus' in df_proj.columns:
                    st.metric("Status", df_proj['vcsProjectStatus'].iloc[0], delta_color="off")

            st.divider()

            col_graf1, col_graf2 = st.columns([3, 1.5])

            with col_graf1:
                fig = go.Figure()
                fig.add_trace(go.Bar(x=df_proj['Ano_Periodo'], y=df_proj['TotalVintageQuantity'], name='Total Vintage', marker_color='#1E800A'))
                fig.add_trace(go.Bar(x=df_proj['Ano_Periodo'], y=df_proj['SumQuantity'], name='Sum Quantity', marker_color='#6DD458'))
                fig.add_trace(go.Bar(x=df_proj['Ano_Periodo'], y=df_proj['Sum_Retired'], name='Retired', marker_color='#FFC2A3'))
                if 'Mean' in df_proj.columns:
                    fig.add_trace(go.Scatter(x=df_proj['Ano_Periodo'], y=df_proj['Mean'], mode='lines+markers',
                                             name='Média', line=dict(color='#1E800A', width=3), marker=dict(size=8)))
                if 'IC_Mais' in df_proj.columns and 'IC_Menos' in df_proj.columns:
                    fig.add_trace(go.Scatter(x=df_proj['Ano_Periodo'], y=df_proj['IC_Mais'], mode='lines',
                                             name='IC Superior', line=dict(color='gray', width=2, dash='dot')))
                    fig.add_trace(go.Scatter(x=df_proj['Ano_Periodo'], y=df_proj['IC_Menos'], mode='lines',
                                             name='IC Inferior', line=dict(color='gray', width=2, dash='dot')))
                fig.update_layout(barmode='group', title=f"Análise de VCUs - {projeto_sel}",
                                  xaxis_title="Período (Ano)", yaxis_title="Quantidade de VCUs",
                                  legend_title="Métricas", template="plotly_white", height=450,
                                  hovermode='x unified', margin=dict(t=50, b=40, l=60, r=20))
                st.plotly_chart(fig, use_container_width=True)

            with col_graf2:
                metricas_totais = {'TotalVintageQuantity': '#1E800A', 'SumQuantity': '#6DD458',
                                   'Sum_Retired': '#FFC2A3', 'Sum_Active': '#A3C4F3'}
                labels, valores, cores = [], [], []
                for col_name, cor in metricas_totais.items():
                    if col_name in df_proj.columns:
                        labels.append(col_name.replace('Sum_', '').replace('Sum', '').replace('Total', 'Total '))
                        valores.append(pd.to_numeric(df_proj[col_name], errors='coerce').sum())
                        cores.append(cor)
                fig_tot = go.Figure()
                fig_tot.add_trace(go.Bar(x=labels, y=valores, marker_color=cores,
                                         text=[f"{v:,.0f}" for v in valores], textposition='outside',
                                         textfont=dict(size=10), showlegend=False))
                fig_tot.update_layout(title="Totais Acumulados", template="plotly_white", height=450,
                                      margin=dict(t=50, b=40, l=40, r=20), yaxis=dict(showticklabels=False))
                st.plotly_chart(fig_tot, use_container_width=True)

            with st.expander("📊 Ver Tabela de Dados Detalhada"):
                display_cols = ['Ano_Periodo', 'TotalVintageQuantity', 'SumQuantity',
                                'Sum_Retired', 'Sum_Active', 'Mean', 'IC_Mais', 'IC_Menos']
                display_cols = [col for col in display_cols if col in df_proj.columns]
                st.dataframe(df_proj[display_cols].style.format(
                    {col: "{:,.0f}" for col in display_cols if col != 'Ano_Periodo'}),
                    use_container_width=True)

# =====================================
# ABA 5: STORYTELLING
# =====================================

with tabs[4]:
    st.header("📖 A História dos Projetos de Carbono no Brasil")

    story_tabs = st.tabs([
        "🌍 Panorama Geral",       # index 0
        "📍 Dados AOI",            # index 1
        "🌿 MapBiomas",            # index 2
        "📊 Evolução Temporal",    # index 3
        "🎯 Impacto Regional",     # index 4
        "💡 Insights"              # index 5
    ])

    # =====================================
    # STORYTELLING 0: PANORAMA GERAL
    # =====================================

    with story_tabs[0]:
        st.markdown("## 🌱 A Jornada do Carbono Florestal Brasileiro")

        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("""
            ### Do Desmatamento aos Créditos de Carbono
            - 💰 **Valoriza economicamente** a floresta em pé
            - 🌳 **Preserva a biodiversidade** 
            - 👥 **Beneficia comunidades** locais
            - 🌍 **Combate as mudanças climáticas** globais
            """)
        with col2:
            st.markdown("""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 30px; border-radius: 15px; color: white; text-align: center;'>
                <h1 style='margin: 0; font-size: 3em;'>🌎</h1>
                <h3 style='margin: 10px 0;'>Brasil</h3>
                <p style='margin: 5px 0; font-size: 1.2em;'><b>{:,}</b> projetos</p>
                <p style='margin: 5px 0;'><b>{:,}</b> com créditos</p>
            </div>
            """.format(len(df_all), df_credit['resourceName_x'].nunique()), unsafe_allow_html=True)

        st.divider()
        st.markdown("### 📅 Linha do Tempo dos Projetos")

        if 'vcsRegistrationDate' in df_all.columns:
            df_timeline = df_all.copy()
            df_timeline['vcsRegistrationDate'] = pd.to_datetime(df_timeline['vcsRegistrationDate'], errors='coerce')
            df_timeline = df_timeline.dropna(subset=['vcsRegistrationDate'])
            df_timeline['Ano'] = df_timeline['vcsRegistrationDate'].dt.year
            timeline_data = df_timeline.groupby(['Ano', 'vcsAFOLUActivity']).size().reset_index(name='Quantidade')
            timeline_data = timeline_data[timeline_data['Ano'] >= 2000]
            fig_timeline = px.area(timeline_data, x='Ano', y='Quantidade', color='vcsAFOLUActivity',
                                   color_discrete_map=ACTIVITY_COLORS,
                                   title='Crescimento dos Projetos de Carbono ao Longo do Tempo')
            fig_timeline.update_layout(hovermode='x unified', height=400, legend_title_text='Tipo de Atividade')
            st.plotly_chart(fig_timeline, use_container_width=True)

        st.markdown("### 🎯 Números que Contam Histórias")
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)

        with col_m1:
            if 'vcsAcresHectares' in df_all.columns:
                try:
                    df_all_temp = df_all.copy()
                    df_all_temp['area_num'] = pd.to_numeric(
                        df_all_temp['vcsAcresHectares'].astype(str)
                        .str.replace(r'[^\d.,]', '', regex=True).str.replace(',', '', regex=False), errors='coerce')
                    st.metric("Área Total Protegida", f"{df_all_temp['area_num'].sum()/1000000:,.1f}M ha")
                except Exception:
                    st.metric("Área Total Protegida", "N/A")

        with col_m2:
            if 'vcsEstimatedAnnualEmissionReductions' in df_all.columns:
                try:
                    total_eaer = pd.to_numeric(df_all['vcsEstimatedAnnualEmissionReductions'], errors='coerce').sum()
                    st.metric("Reduções Anuais (tCO2e)", f"{total_eaer/1000000:.1f}M")
                except Exception:
                    st.metric("Reduções Anuais", "N/A")

        with col_m3:
            try:
                redd_count = len(df_all[df_all['vcsAFOLUActivity'].str.contains('REDD', na=False)])
                st.metric("Projetos REDD+", f"{redd_count}")
            except Exception:
                st.metric("Projetos REDD+", "N/A")

        with col_m4:
            if 'state_Recode' in df_all.columns:
                st.metric("Estados Alcançados", f"{df_all['state_Recode'].nunique()}")

    # =====================================
    # STORYTELLING 1: DADOS AOI (GFW + PRODES + DETER)
    # =====================================

    with story_tabs[1]:
        st.markdown("## 📍 Análise Espacial por Projeto")

        GFW_API_KEY = st.secrets["GFW_API_KEY"].strip()

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        KML_DIR  = os.path.join(BASE_DIR, "kml")

        gdf_combined, erros = carregar_geometrias(df_all, KML_DIR)

        if erros:
            with st.expander("⚠️ Erros ao carregar alguns KMLs"):
                for f, e in erros:
                    st.text(f"{f}: {e}")

        if gdf_combined.empty:
            st.warning("Nenhum KML válido encontrado.")
        else:
            gdf_plot = gdf_combined[
                ~gdf_combined["geometry"].is_empty & gdf_combined["geometry"].notnull()
            ].copy()
            gdf_plot = gdf_plot[gdf_plot.is_valid]

            if gdf_plot.empty:
                st.warning("⚠️ Nenhuma geometria válida para exibir.")
            else:
                project_options = ["🌎 Visão Geral (Todos os Projetos)"] + [
                    f"{row.get('resourceName_x', 'Sem nome')} — {row.get('state_Recode', 'N/A')}"
                    for _, row in gdf_plot.iterrows()
                ]

                selected_project = st.selectbox(
                    "📍 Selecione um projeto para análise:",
                    options=project_options,
                    key="project_selector_v2"
                )

                st.info("⚠️ Pedimos compreensão: Estamos em desenvolvimento e os resultados a seguir são estimativas com base no recorte da BBox (Área Azul)")
                st.info("🧞 New features: Estimativas com base no recorte da BBox (Área Azul) e apenas para AOI - Área de Interesse")

                is_overview = selected_project == "🌎 Visão Geral (Todos os Projetos)"

                if is_overview:
                    centroid     = gdf_plot.geometry.centroid
                    center       = [centroid.y.mean(), centroid.x.mean()]
                    zoom_start   = 5
                    selected_gdf = gdf_plot
                else:
                    project_name = selected_project.split(" — ")[0]
                    selected_gdf = gdf_plot[gdf_plot["resourceName_x"] == project_name]
                    if not selected_gdf.empty:
                        bounds     = selected_gdf.total_bounds
                        center     = [(bounds[1]+bounds[3])/2, (bounds[0]+bounds[2])/2]
                        zoom_start = 10
                    else:
                        selected_gdf = gdf_plot
                        centroid     = gdf_plot.geometry.centroid
                        center       = [centroid.y.mean(), centroid.x.mean()]
                        zoom_start   = 5

                # bbox string para WFS
                if not is_overview:
                    b = selected_gdf.total_bounds
                    bbox_str = f"{b[0]},{b[1]},{b[2]},{b[3]}"
                else:
                    bbox_str = None

                st.divider()

                # ===================================
                # TABS INTERNAS: GFW | PRODES | DETER
                # ===================================
                tab_gfw, tab_prodes, tab_deter_amz, tab_deter_cer = st.tabs([
                    "🌳 GFW", "🔴 PRODES", "🟠 DETER Amazônia", "🟡 DETER Cerrado"
                ])

                # ===================================
                # TAB GFW
                # ===================================
                with tab_gfw:
                    col_mapa, col_info = st.columns([8, 2])

                    with col_mapa:
                        st.markdown("### 🗺️ Mapa")

                        c1, c2, c3 = st.columns(3)
                        with c1:
                            show_loss = st.toggle("🔴 Tree Cover Loss", value=True,  key="toggle_loss")
                        with c2:
                            show_glad = st.toggle("🟡 GLAD Alerts",     value=False, key="toggle_glad")
                        with c3:
                            show_radd = st.toggle("🟠 RADD Alerts",     value=False, key="toggle_radd")

                        m_gfw = folium.Map(location=center, zoom_start=zoom_start, tiles=None)
                        folium.TileLayer('Esri.WorldImagery', name='Satélite', control=False).add_to(m_gfw)

                        if show_loss:
                            folium.TileLayer(
                                tiles='https://tiles.globalforestwatch.org/umd_tree_cover_loss/v1.11/tcd_30/{z}/{x}/{y}.png',
                                name='Tree Cover Loss', attr='GFW', overlay=True, opacity=0.8
                            ).add_to(m_gfw)
                        if show_glad:
                            folium.TileLayer(
                                tiles='https://tiles.globalforestwatch.org/umd_glad_landsat_alerts/v20260320/default/{z}/{x}/{y}.png',
                                name='GLAD Alerts', attr='GFW', overlay=True, opacity=0.8
                            ).add_to(m_gfw)
                        if show_radd:
                            folium.TileLayer(
                                tiles='https://tiles.globalforestwatch.org/wur_radd_alerts/v20260315/default/{z}/{x}/{y}.png',
                                name='RADD Alerts', attr='GFW', overlay=True, opacity=0.8
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
                        st_folium(m_gfw, width=None, height=600, key="map_gfw")

                    with col_info:
                        if is_overview:
                            st.info("💡 Selecione um projeto.")
                            st.metric("Total de Projetos", f"{len(gdf_plot):,}")
                        else:
                            row_proj = selected_gdf.iloc[0]
                            st.markdown("### 📋 Info")
                            st.markdown(f"**Projeto:** {row_proj.get('resourceName_x', 'N/A')}")
                            st.markdown(f"**Estado:** {row_proj.get('state_Recode', 'N/A')}")
                            st.markdown(f"**ID:** {row_proj.get('resourceIdentifier', 'N/A')}")

                    if not is_overview:
                        from shapely.ops import unary_union
                        geom = selected_gdf.geometry.iloc[0]
                        if geom.geom_type == 'GeometryCollection':
                            polys = [g for g in geom.geoms if g.geom_type in ['Polygon', 'MultiPolygon']]
                            geom  = unary_union(polys) if polys else None

                        if geom and geom.geom_type in ['Polygon', 'MultiPolygon']:
                            geojson_poly = mapping(geom)
                            st.divider()
                            col_g1, col_g2, col_g3 = st.columns(3)

                            with col_g1:
                                st.markdown("### 📊 Tree Cover Loss")
                                with st.spinner("Consultando GFW..."):
                                    df_loss = gfw_tree_cover_loss(geojson_poly, GFW_API_KEY)
                                if df_loss.empty:
                                    st.warning("Sem dados.")
                                else:
                                    fig = go.Figure()
                                    fig.add_trace(go.Bar(x=df_loss['umd_tree_cover_loss__year'],
                                                         y=df_loss['loss_ha'], marker_color='#ff4444'))
                                    fig.update_layout(height=300, template="plotly_white",
                                                      margin=dict(t=10, b=40, l=40, r=10),
                                                      xaxis_title="Ano", yaxis_title="ha",
                                                      hovermode='x unified')
                                    st.plotly_chart(fig, use_container_width=True)

                            with col_g2:
                                st.markdown("### 🟡 GLAD")
                                with st.spinner("Consultando GLAD..."):
                                    df_glad = gfw_glad_alerts(geojson_poly, GFW_API_KEY)
                                if df_glad.empty:
                                    st.warning("Sem dados.")
                                else:
                                    fig = go.Figure()
                                    fig.add_trace(go.Bar(x=df_glad['alert__year'],
                                                         y=df_glad['alert_count'], marker_color='#FFC300'))
                                    fig.update_layout(height=300, template="plotly_white",
                                                      margin=dict(t=10, b=40, l=40, r=10),
                                                      xaxis_title="Ano", yaxis_title="Alertas",
                                                      hovermode='x unified')
                                    st.plotly_chart(fig, use_container_width=True)

                            with col_g3:
                                st.markdown("### 🟠 RADD")
                                with st.spinner("Consultando RADD..."):
                                    df_radd = gfw_radd_alerts(geojson_poly, GFW_API_KEY)
                                if df_radd.empty:
                                    st.warning("Sem dados.")
                                else:
                                    fig = go.Figure()
                                    fig.add_trace(go.Bar(x=df_radd['alert__year'],
                                                         y=df_radd['alert_count'], marker_color='#FF7900'))
                                    fig.update_layout(height=300, template="plotly_white",
                                                      margin=dict(t=10, b=40, l=40, r=10),
                                                      xaxis_title="Ano", yaxis_title="Alertas",
                                                      hovermode='x unified')
                                    st.plotly_chart(fig, use_container_width=True)

                # ===================================
                # TAB PRODES
                # ===================================
                with tab_prodes:
                    col_mapa_p, col_info_p = st.columns([8, 2])

                    with col_mapa_p:
                        st.markdown("### 🗺️ Mapa PRODES")

                        c1, c2 = st.columns(2)
                        with c1:
                            show_prodes_br  = st.toggle("🔴 PRODES Brasil",    value=True,  key="toggle_prodes_br")
                        with c2:
                            show_prodes_amz = st.toggle("🟥 PRODES Legal AMZ", value=False, key="toggle_prodes_amz")

                        m_prodes = folium.Map(location=center, zoom_start=zoom_start, tiles=None)
                        folium.TileLayer('Esri.WorldImagery', name='Satélite', control=False).add_to(m_prodes)

                        # Bounding box da AOI
                        if not is_overview:
                            b = selected_gdf.total_bounds  # (minx, miny, maxx, maxy)
                            bbox_coords = [
                                [b[1], b[0]],  # SW
                                [b[1], b[2]],  # SE
                                [b[3], b[2]],  # NE
                                [b[3], b[0]],  # NW
                                [b[1], b[0]],  # fecha
                            ]
                            folium.PolyLine(
                                locations=bbox_coords,
                                color="#00FFFF",
                                weight=2,
                                dash_array="6 4",
                                tooltip="Bounding Box (área de consulta WFS)",
                                opacity=0.8
                            ).add_to(m_prodes)


                        if show_prodes_br:
                            folium.WmsTileLayer(
                                url="https://terrabrasilis.dpi.inpe.br/geoserver/prodes-brasil-nb/prodes_brasil/ows",
                                layers="prodes_brasil",
                                fmt="image/png",
                                transparent=True,
                                name="PRODES Brasil",
                                overlay=True,
                                opacity=0.8
                            ).add_to(m_prodes)

                        if show_prodes_amz:
                            folium.WmsTileLayer(
                                url="https://terrabrasilis.dpi.inpe.br/geoserver/prodes-legal-amz/yearly_deforestation/ows",
                                layers="yearly_deforestation",
                                fmt="image/png",
                                transparent=True,
                                name="PRODES Legal AMZ",
                                overlay=True,
                                opacity=0.8
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
                        st_folium(m_prodes, width=None, height=600, key="map_prodes")

                    with col_info_p:
                        if not is_overview:
                            row_proj = selected_gdf.iloc[0]
                            st.markdown("### 📋 Info")
                            st.markdown(f"**Projeto:** {row_proj.get('resourceName_x', 'N/A')}")
                            st.markdown(f"**Estado:** {row_proj.get('state_Recode', 'N/A')}")
                        else:
                            st.info("💡 Selecione um projeto.")

                    if not is_overview and bbox_str:
                        st.divider()
                        #col_g1, col_g2 = st.columns(2)
                        col_g1, col_g2, col_g3 = st.columns(3)

                        with col_g1:
                            st.markdown("### 📊 PRODES Legal AMZ — Incremento Anual na AOI")
                            with st.spinner("Consultando TerraBrasilis WFS..."):
                                df_prodes = terrabrasilis_wfs(
                                    "https://terrabrasilis.dpi.inpe.br/geoserver/prodes-legal-amz/yearly_deforestation/ows",
                                    "prodes-legal-amz:yearly_deforestation",
                                    bbox_str
                                )
                            if df_prodes.empty:
                                st.warning("Sem dados PRODES para esta AOI.")
                            else:
                                if 'year' in df_prodes.columns:
                                    df_prodes['year'] = pd.to_numeric(df_prodes['year'], errors='coerce')
                                    df_prodes_year = df_prodes.groupby('year').size().reset_index(name='poligonos')
                                    fig_p = go.Figure()
                                    fig_p.add_trace(go.Bar(
                                        x=df_prodes_year['year'],
                                        y=df_prodes_year['poligonos'],
                                        marker_color='#C0392B',
                                        name='Polígonos PRODES'
                                    ))
                                    fig_p.update_layout(
                                        xaxis_title="Ano", yaxis_title="Nº Polígonos",
                                        height=300, template="plotly_white",
                                        margin=dict(t=10, b=40, l=40, r=10),
                                        hovermode='x unified'
                                    )
                                    st.plotly_chart(fig_p, use_container_width=True)

                                with st.expander("📋 Tabela PRODES"):
                                    st.dataframe(df_prodes, use_container_width=True, height=300)
                                    csv = df_prodes.to_csv(index=False).encode('utf-8')
                                    st.download_button("⬇️ Download CSV", data=csv,
                                                       file_name="prodes_aoi.csv", mime="text/csv")
                                    
                        with col_g2:
                            st.markdown("### 📊 Incremento Anual — Área (km²)")
                            if 'year' in df_prodes.columns and 'area_km' in df_prodes.columns:
                                df_prodes['area_km'] = pd.to_numeric(df_prodes['area_km'], errors='coerce')
                                df_prodes_area = df_prodes.groupby('year').agg(
                                    area_total=('area_km', 'sum')
                                ).reset_index()
                                fig_a = go.Figure()
                                fig_a.add_trace(go.Bar(
                                    x=df_prodes_area['year'],
                                    y=df_prodes_area['area_total'],
                                    marker_color='#922B21',
                                    name='Área (km²)'
                                ))
                                fig_a.update_layout(
                                    xaxis_title="Ano", yaxis_title="km²",
                                    height=300, template="plotly_white",
                                    margin=dict(t=10, b=40, l=40, r=10),
                                    hovermode='x unified'
                                )
                                st.plotly_chart(fig_a, use_container_width=True)
                            else:
                                st.warning("Coluna `area_km` não encontrada.")

                        with col_g3:
                            st.markdown("### 📊 Classe de Desmatamento")
                            if not df_prodes.empty and 'class_name' in df_prodes.columns:
                                class_counts = df_prodes['class_name'].value_counts().reset_index()
                                class_counts.columns = ['Classe', 'Count']
                                fig_cls = go.Figure()
                                fig_cls.add_trace(go.Bar(
                                    x=class_counts['Classe'],
                                    y=class_counts['Count'],
                                    marker_color='#E74C3C'
                                ))
                                fig_cls.update_layout(
                                    xaxis_title="Classe", yaxis_title="Nº Polígonos",
                                    height=300, template="plotly_white",
                                    margin=dict(t=10, b=40, l=40, r=10)
                                )
                                st.plotly_chart(fig_cls, use_container_width=True)

                # ===================================
                # TAB DETER AMAZÔNIA
                # ===================================
                with tab_deter_amz:
                    col_mapa_d, col_info_d = st.columns([8, 2])

                    with col_mapa_d:
                        st.markdown("### 🗺️ Mapa DETER Amazônia")

                        m_deter = folium.Map(location=center, zoom_start=zoom_start, tiles=None)
                        folium.TileLayer('Esri.WorldImagery', name='Satélite', control=False).add_to(m_deter)
                        
                        # Bounding box da AOI
                        if not is_overview:
                            b = selected_gdf.total_bounds  # (minx, miny, maxx, maxy)
                            bbox_coords = [
                                [b[1], b[0]],  # SW
                                [b[1], b[2]],  # SE
                                [b[3], b[2]],  # NE
                                [b[3], b[0]],  # NW
                                [b[1], b[0]],  # fecha
                            ]
                            folium.PolyLine(
                                locations=bbox_coords,
                                color="#00FFFF",
                                weight=2,
                                dash_array="6 4",
                                tooltip="Bounding Box (área de consulta WFS)",
                                opacity=0.8
                            ).add_to(m_deter)

                        folium.WmsTileLayer(
                            url="https://terrabrasilis.dpi.inpe.br/geoserver/deter-amz/deter_amz/ows",
                            layers="deter_amz",
                            fmt="image/png",
                            transparent=True,
                            name="DETER AMZ",
                            overlay=True,
                            opacity=0.8
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
                        st_folium(m_deter, width=None, height=600, key="map_deter_amz")

                    with col_info_d:
                        if not is_overview:
                            row_proj = selected_gdf.iloc[0]
                            st.markdown("### 📋 Info")
                            st.markdown(f"**Projeto:** {row_proj.get('resourceName_x', 'N/A')}")
                            st.markdown(f"**Estado:** {row_proj.get('state_Recode', 'N/A')}")
                            st.markdown("""
                            <style>
                            .deter-leg { font-size: 12px; line-height: 1.8; }
                            .deter-item { display: flex; align-items: center; margin: 3px 0; }
                            .deter-box { width: 16px; height: 16px; border-radius: 2px; margin-right: 8px; 
                                         flex-shrink: 0; border: 1px solid #aaa; }
                            </style>
                            <div class='deter-leg'>
                            <b>Avisos de Desmatamento<br>a partir de 2016</b><br><br>
                            <div class='deter-item'><div class='deter-box' style='background:#d7191c'></div>Cicatriz de queimada</div>
                            <div class='deter-item'><div class='deter-box' style='background:#868686'></div>Corte seletivo</div>
                            <div class='deter-item'><div class='deter-box' style='background:#db83ff'></div>Corte seletivo desordenado</div>
                            <div class='deter-item'><div class='deter-box' style='background:#ff7e00'></div>Corte seletivo geométrico</div>
                            <div class='deter-item'><div class='deter-box' style='background:#ffffbf'></div>Degradação</div>
                            <div class='deter-item'><div class='deter-box' style='background:#8a5f4b'></div>Desmatamento corte raso</div>
                            <div class='deter-item'><div class='deter-box' style='background:#abdda4'></div>Desmatamento vegetação</div>
                            <div class='deter-item'><div class='deter-box' style='background:#4223e5'></div>Mineração</div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                        else:
                            st.info("💡 Selecione um projeto.")
                    #st.divider()
                    
                    if not is_overview and bbox_str:
                        st.divider()
                        st.markdown("### 📊 DETER AMZ — Alertas na AOI")
                        with st.spinner("Consultando TerraBrasilis WFS..."):
                            df_deter = terrabrasilis_wfs(
                                "https://terrabrasilis.dpi.inpe.br/geoserver/deter-amz/deter_amz/ows",
                                "deter-amz:deter_amz",
                                bbox_str
                            )

                        if df_deter.empty:
                            st.warning("Sem alertas DETER para esta AOI.")
                        else:
                            #col_g1, col_g2, col_g3 = st.columns(3)
                            
                            col_g1, col_g2 = st.columns([2, 3])

                            with col_g1:
                                st.markdown("### 📊 Alertas por Ano")
                                if 'view_date' in df_deter.columns:
                                    df_deter['year'] = pd.to_datetime(
                                        df_deter['view_date'], errors='coerce').dt.year
                                    df_deter_year = df_deter.groupby('year').agg(
                                        alertas=('year', 'count'),
                                        area_km2=('areauckm', 'sum')
                                    ).reset_index()
                                    fig_d = go.Figure()
                                    fig_d.add_trace(go.Bar(
                                        x=df_deter_year['year'],
                                        y=df_deter_year['alertas'],
                                        marker_color='#E67E22', name='Alertas'
                                    ))
                                    fig_d.update_layout(
                                        xaxis_title="Ano", yaxis_title="Nº Alertas",
                                        height=300, template="plotly_white",
                                        margin=dict(t=10, b=40, l=40, r=10),
                                        hovermode='x unified'
                                    )
                                    st.plotly_chart(fig_d, use_container_width=True)

                                st.markdown("### 📊 Classe de Alerta")
                                if 'classname' in df_deter.columns:
                                    class_d = df_deter['classname'].value_counts().reset_index()
                                    class_d.columns = ['Classe', 'Count']
                                    fig_cd = go.Figure()
                                    fig_cd.add_trace(go.Bar(
                                        x=class_d['Classe'], y=class_d['Count'],
                                        marker_color='#D35400'
                                    ))
                                    fig_cd.update_layout(
                                        xaxis_title="Classe", yaxis_title="Alertas",
                                        height=300, template="plotly_white",
                                        margin=dict(t=10, b=40, l=40, r=10)
                                    )
                                    st.plotly_chart(fig_cd, use_container_width=True)

                            with col_g2:
                                st.markdown("### 📊 Área por Classe ao Longo do Tempo (ha)")
                                if 'classname' in df_deter.columns and 'areamunkm' in df_deter.columns and 'year' in df_deter.columns:
                                    df_deter['areamunkm'] = pd.to_numeric(df_deter['areamunkm'], errors='coerce')*100
                                    df_area_linha = df_deter.groupby(['year', 'classname']).agg(
                                        area_total=('areamunkm', 'sum')
                                    ).reset_index()

                                    DETER_COLORS = {
                                        'CICATRIZ_DE_QUEIMADA':       '#d7191c',
                                        'CORTE_SELETIVO':             '#868686',
                                        'CS_DESORDENADO':             '#db83ff',
                                        'CS_GEOMETRICO':              '#ff7e00',
                                        'DEGRADACAO':                 '#ffffbf',
                                        'DESMATAMENTO_CR':            '#8a5f4b',
                                        'DESMATAMENTO_VEG':           '#abdda4',
                                        'MINERACAO':                  '#4223e5',
                                    }

                                    fig_da = go.Figure()
                                    for classe in sorted(df_area_linha['classname'].unique()):
                                        df_cls = df_area_linha[df_area_linha['classname'] == classe]
                                        cor = DETER_COLORS.get(classe.upper().replace(' ', '_'), '#999999')
                                        fig_da.add_trace(go.Scatter(
                                            x=df_cls['year'],
                                            y=df_cls['area_total'],
                                            mode='lines+markers',
                                            name=classe,
                                            line=dict(width=2, color=cor),
                                            marker=dict(size=6, color=cor)
                                        ))

                                    fig_da.update_layout(
                                        xaxis_title="Ano", yaxis_title="ha",
                                        height=640, template="plotly_white",
                                        margin=dict(t=10, b=40, l=40, r=10),
                                        hovermode='x unified',
                                        #legend=dict(font=dict(size=10), orientation='v')
                                        legend=dict(font=dict(size=10), orientation='v', xanchor='auto', yanchor='auto') #Voltar aqui para alinhar tabelea
                                    )
                                    st.plotly_chart(fig_da, use_container_width=True)
                                else:
                                    st.warning("Colunas necessárias não encontradas.")

                            with st.expander("📋 Tabela DETER AMZ"):
                                st.dataframe(df_deter, use_container_width=True, height=300)
                                csv = df_deter.to_csv(index=False).encode('utf-8')
                                st.download_button("⬇️ Download CSV", data=csv,
                                                   file_name="deter_amz_aoi.csv", mime="text/csv")
                        #if df_deter.empty:
                        #    st.warning("Sem alertas DETER para esta AOI.")
                        #else:
                        #    col_g1, col_g2 = st.columns(2)
#
                        #    with col_g1:
                        #        if 'view_date' in df_deter.columns:
                        #            df_deter['year'] = pd.to_datetime(
                        #                df_deter['view_date'], errors='coerce').dt.year
                        #            df_deter_year = df_deter.groupby('year').agg(
                        #                alertas=('year', 'count'),
                        #                area_km2=('areauckm', 'sum')
                        #            ).reset_index()
                        #            fig_d = go.Figure()
                        #            fig_d.add_trace(go.Bar(
                        #                x=df_deter_year['year'],
                        #                y=df_deter_year['alertas'],
                        #                marker_color='#E67E22', name='Alertas'
                        #            ))
                        #            fig_d.update_layout(
                        #                xaxis_title="Ano", yaxis_title="Nº Alertas",
                        #                height=300, template="plotly_white",
                        #                margin=dict(t=10, b=40, l=40, r=10),
                        #                hovermode='x unified'
                        #            )
                        #            st.plotly_chart(fig_d, use_container_width=True)
#
                        #    with col_g2:
                        #        if 'classname' in df_deter.columns:
                        #            class_d = df_deter['classname'].value_counts().reset_index()
                        #            class_d.columns = ['Classe', 'Count']
                        #            fig_cd = go.Figure()
                        #            fig_cd.add_trace(go.Bar(
                        #                x=class_d['Classe'], y=class_d['Count'],
                        #                marker_color='#D35400'
                        #            ))
                        #            fig_cd.update_layout(
                        #                xaxis_title="Classe", yaxis_title="Alertas",
                        #                height=300, template="plotly_white",
                        #                margin=dict(t=10, b=40, l=40, r=10)
                        #            )
                        #            st.plotly_chart(fig_cd, use_container_width=True)
#
                        #    with st.expander("📋 Tabela DETER AMZ"):
                        #        st.dataframe(df_deter, use_container_width=True, height=300)
                        #        csv = df_deter.to_csv(index=False).encode('utf-8')
                        #        st.download_button("⬇️ Download CSV", data=csv,
                        #                           file_name="deter_amz_aoi.csv", mime="text/csv")

                # ===================================
                # TAB DETER CERRADO
                # ===================================
                with tab_deter_cer:
                    col_mapa_c, col_info_c = st.columns([8, 2])

                    with col_mapa_c:
                        st.markdown("### 🗺️ Mapa DETER Cerrado")

                        m_cer = folium.Map(location=center, zoom_start=zoom_start, tiles=None)
                        folium.TileLayer('Esri.WorldImagery', name='Satélite', control=False).add_to(m_cer)

                        # Bounding box da AOI
                        if not is_overview:
                            b = selected_gdf.total_bounds  # (minx, miny, maxx, maxy)
                            bbox_coords = [
                                [b[1], b[0]],  # SW
                                [b[1], b[2]],  # SE
                                [b[3], b[2]],  # NE
                                [b[3], b[0]],  # NW
                                [b[1], b[0]],  # fecha
                            ]
                            folium.PolyLine(
                                locations=bbox_coords,
                                color="#00FFFF",
                                weight=2,
                                dash_array="6 4",
                                tooltip="Bounding Box (área de consulta WFS)",
                                opacity=0.8
                            ).add_to(m_cer)


                        folium.WmsTileLayer(
                            url="https://terrabrasilis.dpi.inpe.br/geoserver/deter-cerrado-nb/deter_cerrado/ows",
                            layers="deter_cerrado",
                            fmt="image/png",
                            transparent=True,
                            name="DETER Cerrado",
                            overlay=True,
                            opacity=0.8
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
                        st_folium(m_cer, width=None, height=600, key="map_deter_cer")

                    with col_info_c:
                        if not is_overview:
                            row_proj = selected_gdf.iloc[0]
                            st.markdown("### 📋 Info")
                            st.markdown(f"**Projeto:** {row_proj.get('resourceName_x', 'N/A')}")
                            st.markdown(f"**Estado:** {row_proj.get('state_Recode', 'N/A')}")
                        else:
                            st.info("💡 Selecione um projeto.")

                    if not is_overview and bbox_str:
                        st.divider()
                        st.markdown("### 📊 DETER Cerrado — Alertas na AOI")
                        with st.spinner("Consultando TerraBrasilis WFS..."):
                            df_cer = terrabrasilis_wfs(
                                "https://terrabrasilis.dpi.inpe.br/geoserver/deter-cerrado-nb/deter_cerrado/ows",
                                "deter-cerrado-nb:deter_cerrado",
                                bbox_str
                            )

                        if df_cer.empty:
                            st.warning("Sem alertas DETER Cerrado para esta AOI.")
                        else:
                            st.dataframe(df_cer, use_container_width=True, height=300)
                            csv = df_cer.to_csv(index=False).encode('utf-8')
                            st.download_button("⬇️ Download CSV", data=csv,
                                               file_name="deter_cerrado_aoi.csv", mime="text/csv")

    # =====================================
    # STORYTELLING 2: MAPBIOMAS
    # =====================================

    #with story_tabs[2]:
    #    st.markdown("## 🌿 Alertas MapBiomas")
#
    #    MAPBIOMAS_EMAIL    = st.secrets["MAPBIOMAS_EMAIL"]
    #    MAPBIOMAS_PASSWORD = st.secrets["MAPBIOMAS_PASSWORD"]
#
    #    mb_token = mapbiomas_get_token(MAPBIOMAS_EMAIL, MAPBIOMAS_PASSWORD)
#
    #    if not mb_token:
    #        st.error("❌ Não foi possível autenticar no MapBiomas.")
    #    else:
    #        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    #        KML_DIR  = os.path.join(BASE_DIR, "kml")
    #        gdf_mb, _ = carregar_geometrias(df_all, KML_DIR)
#
    #        if gdf_mb.empty:
    #            st.warning("Nenhum KML válido encontrado.")
    #        else:
    #            gdf_mb_plot = gdf_mb[~gdf_mb["geometry"].is_empty & gdf_mb["geometry"].notnull()].copy()
    #            gdf_mb_plot = gdf_mb_plot[gdf_mb_plot.is_valid]
#
    #            mb_options = ["🌎 Visão Geral (Todos os Projetos)"] + [
    #                f"{row.get('resourceName_x', 'Sem nome')} — {row.get('state_Recode', 'N/A')}"
    #                for _, row in gdf_mb_plot.iterrows()
    #            ]
#
    #            mb_selected = st.selectbox("📍 Selecione um projeto:", options=mb_options, key="mb_project_selector")
    #            mb_is_overview = mb_selected == "🌎 Visão Geral (Todos os Projetos)"
#
    #            if mb_is_overview:
    #                st.info("💡 Selecione um projeto para ver os alertas MapBiomas.")
    #            else:
    #                mb_project_name = mb_selected.split(" — ")[0]
    #                mb_gdf = gdf_mb_plot[gdf_mb_plot["resourceName_x"] == mb_project_name]
#
    #                if mb_gdf.empty:
    #                    st.warning("Projeto não encontrado.")
    #                else:
    #                    bounds = mb_gdf.total_bounds
    #                    bbox   = [float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3])]
#
    #                    with st.spinner("Consultando MapBiomas Alerta..."):
    #                        mb_data = mapbiomas_alerts(bbox, mb_token)
#
    #                    if not mb_data:
    #                        st.warning("Sem dados disponíveis para este projeto.")
    #                    else:
    #                        summary    = mb_data.get('summary', {})
    #                        collection = mb_data.get('collection', [])
    #                        metadata   = mb_data.get('metadata', {})
#
    #                        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    #                        with col_m1:
    #                            st.metric("Total de Alertas", f"{summary.get('total', 0):,}")
    #                        with col_m2:
    #                            st.metric("Área Total (ha)", f"{summary.get('area', 0):,.1f}")
    #                        with col_m3:
    #                            st.metric("Total de Páginas", f"{metadata.get('totalPages', 1)}")
    #                        with col_m4:
    #                            st.metric("Anos com Alertas", f"{len(summary.get('alertsByYear', []))}")
#
    #                        st.divider()
#
    #                        col_g1, col_g2 = st.columns(2)
#
    #                        with col_g1:
    #                            st.markdown("### 📊 Alertas por Ano")
    #                            df_by_year = pd.DataFrame(summary.get('alertsByYear', []))
    #                            if not df_by_year.empty:
    #                                fig_ay = go.Figure()
    #                                fig_ay.add_trace(go.Bar(x=df_by_year['year'], y=df_by_year['value'],
    #                                                        marker_color='#E67E22', name='Alertas'))
    #                                fig_ay.update_layout(xaxis_title="Ano", yaxis_title="Alertas", height=300,
    #                                                     template="plotly_white", margin=dict(t=10, b=40, l=40, r=10),
    #                                                     hovermode='x unified')
    #                                st.plotly_chart(fig_ay, use_container_width=True)
#
    #                        with col_g2:
    #                            st.markdown("### 🌳 Área Desmatada por Ano (ha)")
    #                            df_area_year = pd.DataFrame(summary.get('deforestationAreaByYear', []))
    #                            if not df_area_year.empty:
    #                                fig_area = go.Figure()
    #                                fig_area.add_trace(go.Bar(x=df_area_year['year'], y=df_area_year['value'],
    #                                                          marker_color='#C0392B', name='Área (ha)'))
    #                                fig_area.update_layout(xaxis_title="Ano", yaxis_title="ha", height=300,
    #                                                       template="plotly_white", margin=dict(t=10, b=40, l=40, r=10),
    #                                                       hovermode='x unified')
    #                                st.plotly_chart(fig_area, use_container_width=True)
#
    #                        st.divider()
#
    #                        if collection:
    #                            st.markdown("### 📋 Lista de Alertas")
    #                            df_col = pd.DataFrame(collection)
#
    #                            for col_list in ['sources', 'deforestationClasses', 'crossedBiomes', 'crossedStates']:
    #                                if col_list in df_col.columns:
    #                                    df_col[col_list] = df_col[col_list].apply(
    #                                        lambda x: ', '.join(x) if isinstance(x, list) else x)
#
    #                            df_col = df_col.rename(columns={
    #                                'alertCode': 'Código', 'areaHa': 'Área (ha)',
    #                                'detectedAt': 'Detectado em', 'publishedAt': 'Publicado em',
    #                                'sources': 'Fontes', 'deforestationClasses': 'Classe',
    #                                'statusName': 'Status', 'crossedBiomes': 'Bioma', 'crossedStates': 'Estado'
    #                            })
#
    #                            st.dataframe(df_col.style.format({'Área (ha)': '{:,.2f}'}),
    #                                         use_container_width=True, height=400)
#
    #                            csv = df_col.to_csv(index=False).encode('utf-8')
    #                            st.download_button(
    #                                label="⬇️ Download Alertas (CSV)",
    #                                data=csv,
    #                                file_name=f"alertas_mapbiomas_{mb_project_name[:30]}.csv",
    #                                mime='text/csv'
    #                            )
#
    ## =====================================
    # STORYTELLING 3: EVOLUÇÃO TEMPORAL
    # =====================================

    with story_tabs[3]:
        st.markdown("## ⏱️ A Evolução dos Projetos no Tempo")

        pipeline_html = """
        <div style='display: flex; justify-content: space-between; margin: 30px 0;'>
            <div style='text-align: center; flex: 1;'><div style='background: #e3f2fd; padding: 20px; border-radius: 10px; margin: 5px;'><h3>📝</h3><b>Desenvolvimento</b></div></div>
            <div style='text-align: center; flex: 1;'><div style='background: #fff3e0; padding: 20px; border-radius: 10px; margin: 5px;'><h3>🔍</h3><b>Validação</b></div></div>
            <div style='text-align: center; flex: 1;'><div style='background: #e8f5e9; padding: 20px; border-radius: 10px; margin: 5px;'><h3>✅</h3><b>Registro</b></div></div>
            <div style='text-align: center; flex: 1;'><div style='background: #f3e5f5; padding: 20px; border-radius: 10px; margin: 5px;'><h3>📊</h3><b>Monitoramento</b></div></div>
            <div style='text-align: center; flex: 1;'><div style='background: #c8e6c9; padding: 20px; border-radius: 10px; margin: 5px;'><h3>💰</h3><b>Créditos</b></div></div>
        </div>
        """
        st.markdown(pipeline_html, unsafe_allow_html=True)

        if 'vcsRegistrationDate' in df_credit.columns and 'Vintage' in df_credit.columns:
            st.markdown("### ⏰ Tempo até os Primeiros Créditos")
            df_timing = df_credit.copy()
            df_timing['vcsRegistrationDate'] = pd.to_datetime(df_timing['vcsRegistrationDate'], errors='coerce')
            df_timing['Vintage_Year'] = df_timing['Vintage'].apply(
                lambda x: int(x.split(' e ')[0][:4]) if isinstance(x, str) and ' e ' in x
                else (int(str(x)[:4]) if pd.notna(x) else None))
            df_timing = df_timing.dropna(subset=['vcsRegistrationDate', 'Vintage_Year'])
            df_timing['Registration_Year'] = df_timing['vcsRegistrationDate'].dt.year
            df_timing['Years_to_Credit'] = df_timing['Vintage_Year'] - df_timing['Registration_Year']
            df_timing = df_timing[(df_timing['Years_to_Credit'] >= -5) & (df_timing['Years_to_Credit'] <= 10)]

            if len(df_timing) > 0:
                fig_timing = px.histogram(df_timing, x='Years_to_Credit', nbins=20,
                                          title='Distribuição do Tempo entre Registro e Emissão de Créditos',
                                          color_discrete_sequence=['#26a69a'])
                fig_timing.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig_timing, use_container_width=True)

                col_stat1, col_stat2, col_stat3 = st.columns(3)
                with col_stat1:
                    st.metric("Tempo Mediano", f"{df_timing['Years_to_Credit'].median():.1f} anos")
                with col_stat2:
                    st.metric("Tempo Médio", f"{df_timing['Years_to_Credit'].mean():.1f} anos")
                with col_stat3:
                    st.metric("Projetos Rápidos", f"{len(df_timing[df_timing['Years_to_Credit'] <= 1])}")

        st.divider()
        st.markdown("### 📈 Evolução dos Créditos Emitidos")

        if 'Vintage' in df_credit.columns and 'totalVintageQuantity' in df_credit.columns:
            df_credits_year = df_credit.copy()
            df_credits_year['Vintage_Year'] = df_credits_year['Vintage'].apply(
                lambda x: int(x.split(' e ')[0][:4]) if isinstance(x, str) and ' e ' in x
                else (int(str(x)[:4]) if pd.notna(x) else None))
            df_credits_year = df_credits_year.dropna(subset=['Vintage_Year'])
            df_credits_year['totalVintageQuantity'] = pd.to_numeric(df_credits_year['totalVintageQuantity'], errors='coerce')
            credits_by_year = df_credits_year.groupby('Vintage_Year').agg(
                Total_Creditos=('totalVintageQuantity', 'sum'),
                Num_Projetos=('resourceName_x', 'nunique')
            ).reset_index().rename(columns={'Vintage_Year': 'Ano'})
            credits_by_year = credits_by_year[credits_by_year['Ano'] >= 2000]

            fig_credits_evolution = go.Figure()
            fig_credits_evolution.add_trace(go.Bar(x=credits_by_year['Ano'], y=credits_by_year['Total_Creditos'],
                                                    name='Créditos Emitidos', marker_color='#26a69a', yaxis='y'))
            fig_credits_evolution.add_trace(go.Scatter(x=credits_by_year['Ano'], y=credits_by_year['Num_Projetos'],
                                                        name='Número de Projetos', marker_color='#ff6b6b',
                                                        mode='lines+markers', yaxis='y2'))
            fig_credits_evolution.update_layout(
                title='Emissão de Créditos e Número de Projetos ao Longo do Tempo',
                xaxis_title='Ano', yaxis_title='Total de Créditos (VCUs)',
                yaxis2=dict(title='Número de Projetos', overlaying='y', side='right'),
                height=500, hovermode='x unified')
            st.plotly_chart(fig_credits_evolution, use_container_width=True)

    # =====================================
    # STORYTELLING 4: IMPACTO REGIONAL
    # =====================================

    with story_tabs[4]:
        st.markdown("## 🎯 O Impacto nos Territórios")

        if 'state_Recode' in df_all.columns:
            st.markdown("### 🗺️ Densidade de Projetos por Região")
            state_summary = df_all.groupby('state_Recode').agg({
                'resourceName_x': 'count',
                'vcsAcresHectares': lambda x: pd.to_numeric(x, errors='coerce').sum(),
                'vcsEstimatedAnnualEmissionReductions': lambda x: pd.to_numeric(x, errors='coerce').sum()
            }).reset_index()
            state_summary.columns = ['Estado', 'Num_Projetos', 'Area_Total', 'EAER_Total']
            state_summary = state_summary.sort_values('Num_Projetos', ascending=False).head(15)

            fig_regional = px.scatter(state_summary, x='Area_Total', y='EAER_Total',
                                       size='Num_Projetos', color='Num_Projetos', hover_name='Estado',
                                       title='Relação entre Área, Impacto e Número de Projetos',
                                       color_continuous_scale='Viridis', size_max=60)
            fig_regional.update_layout(height=500)
            st.plotly_chart(fig_regional, use_container_width=True)

        st.divider()
        st.markdown("### 🌳 Perfil de Atividades por Estado")

        if 'state_Recode' in df_all.columns and 'vcsAFOLUActivity' in df_all.columns:
            top_states = df_all['state_Recode'].value_counts().head(10).index.tolist()
            df_activity_state = df_all[df_all['state_Recode'].isin(top_states)]
            activity_by_state = df_activity_state.groupby(['state_Recode', 'vcsAFOLUActivity']).size().reset_index(name='Count')
            fig_activity_state = px.bar(activity_by_state, x='state_Recode', y='Count',
                                         color='vcsAFOLUActivity', color_discrete_map=ACTIVITY_COLORS,
                                         title='Distribuição de Tipos de Atividade nos Principais Estados',
                                         barmode='stack')
            fig_activity_state.update_layout(height=500)
            st.plotly_chart(fig_activity_state, use_container_width=True)

    # =====================================
    # STORYTELLING 5: INSIGHTS
    # =====================================

    with story_tabs[5]:
        st.markdown("## 💡 Insights e Descobertas")

        insight_cols = st.columns(2)
        with insight_cols[0]:
            st.markdown("""
            ### 🔍 Principais Descobertas
            #### 1. Concentração Geográfica
            A maioria dos projetos se concentra em poucos estados com histórico de desmatamento elevado.
            #### 2. Predominância REDD+
            Projetos REDD+ são os mais comuns, refletindo a urgência do combate ao desmatamento.
            #### 3. Ciclo de Maturação
            Em média, projetos levam alguns anos entre registro e emissão dos primeiros créditos.
            """)
        with insight_cols[1]:
            st.markdown("""
            ### 🎯 Oportunidades
            #### 1. Expansão Geográfica
            Diversos estados ainda têm poucos projetos — oportunidades para novos investimentos.
            #### 2. Diversificação
            Além de REDD+, metodologias como ARR e IFM podem ser exploradas.
            #### 3. Escala
            Muitos projetos têm potencial para expansão em áreas adjacentes.
            """)

        st.divider()

        if 'vcsAcresHectares' in df_all.columns and 'vcsEstimatedAnnualEmissionReductions' in df_all.columns:
            df_correlation = df_all.copy()
            df_correlation['Area'] = pd.to_numeric(df_correlation['vcsAcresHectares'], errors='coerce')
            df_correlation['EAER'] = pd.to_numeric(df_correlation['vcsEstimatedAnnualEmissionReductions'], errors='coerce')
            df_correlation = df_correlation.dropna(subset=['Area', 'EAER'])
            df_correlation = df_correlation[(df_correlation['Area'] > 0) & (df_correlation['EAER'] > 0)]

            if len(df_correlation) > 0:
                fig_corr = px.scatter(df_correlation, x='Area', y='EAER', color='vcsAFOLUActivity',
                                       hover_data=['resourceName_x', 'state_Recode'],
                                       color_discrete_map=ACTIVITY_COLORS,
                                       title='Relação entre Tamanho do Projeto e Impacto Climático',
                                       log_x=True, log_y=True)
                fig_corr.update_layout(height=500)
                st.plotly_chart(fig_corr, use_container_width=True)

# =====================================
# ABA 6: DADOS BRUTOS
# =====================================

with tabs[5]:
    st.header("📁 Visualização dos Dados Brutos")

    data_option = st.radio("Selecione o conjunto de dados:",
                           ["Todos os Projetos", "Projetos com Créditos"], horizontal=True)
    df_display = df_all if data_option == "Todos os Projetos" else df_credit

    col1, col2 = st.columns([3, 1])
    with col1:
        st.info(f"Exibindo primeiras 100 linhas de {len(df_display):,} registros")
    with col2:
        if st.button("📥 Download CSV", use_container_width=True):
            csv = df_display.to_csv(index=False)
            st.download_button(
                label="Baixar arquivo", data=csv,
                file_name=f"mrv_data_{data_option.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

    with st.expander("🔍 Filtrar Dados"):
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            if 'state_Recode' in df_display.columns:
                states_raw = ["Todos"] + sorted(df_display['state_Recode'].dropna().unique().tolist())
                selected_state_raw = st.selectbox("Estado:", states_raw, key="raw_state")
                if selected_state_raw != "Todos":
                    df_display = df_display[df_display['state_Recode'] == selected_state_raw]
        with col_f2:
            if 'vcsProjectStatus' in df_display.columns:
                status_raw = ["Todos"] + sorted(df_display['vcsProjectStatus'].dropna().unique().tolist())
                selected_status_raw = st.selectbox("Status:", status_raw, key="raw_status")
                if selected_status_raw != "Todos":
                    df_display = df_display[df_display['vcsProjectStatus'] == selected_status_raw]

    st.dataframe(df_display.head(100), use_container_width=True, height=600)

    with st.expander("ℹ️ Informações sobre as Colunas"):
        col_info_a, col_info_b = st.columns(2)
        with col_info_a:
            st.write(f"**Total de colunas:** {len(df_display.columns)}")
            st.write(f"**Total de linhas (filtrado):** {len(df_display):,}")
        with col_info_b:
            st.write("**Tipos de dados:**")
            for dtype, count in df_display.dtypes.value_counts().items():
                st.text(f"• {dtype}: {count} colunas")
        st.divider()
        st.write("**Lista de Colunas:**")
        cols_list = df_display.columns.tolist()
        for i in range(0, len(cols_list), 3):
            cols = st.columns(3)
            for j, col in enumerate(cols):
                if i + j < len(cols_list):
                    col.text(f"• {cols_list[i + j]}")
