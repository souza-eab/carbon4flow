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
from carbon4flow_gcp import render_aoi_tab

# =====================================
# CONFIGURAÇÃO DA PÁGINA
# =====================================
st.set_page_config(
    page_title="Carbon4Flow",
    page_icon="🌎",
    layout="wide",
    initial_sidebar_state="auto"
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
    mask_ret = df['retiredCancelled'] == True
    df['qty_ret'] = df['quantity'].where(mask_ret, 0)
    df['qty_act'] = df['quantity'].where(~mask_ret, 0)
    estatisticas = df.groupby(group_cols, dropna=False).agg(
        TotalVintageQuantity=('totalVintageQuantity', 'first'),
        SumQuantity=('quantity', 'sum'),
        Sum_Retired=('qty_ret', 'sum'),
        Sum_Active=('qty_act', 'sum')
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
# FUNÇÕES GFW
# =====================================

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
# FUNÇÕES TERRABRASILIS
# =====================================

# [MELHORIA PERF] ttl=3600 adicionado: sem TTL os dados WFS ficavam em cache
# para sempre na sessão mas nunca expiravam entre sessões, gerando inconsistência.
#
# [MELHORIA CLIP] Agora inclui a coluna 'geometry' (dict GeoJSON) no DataFrame
# retornado, além das properties. O CRS declarado pelo servidor é preservado em
# '_wfs_crs' para que clip_and_recalculate possa reprojetar corretamente.
# Confirmado: TerraBrasilis/DETER retorna EPSG:4674 (SIRGAS 2000).
@st.cache_data(ttl=300, show_spinner=False)
def terrabrasilis_wfs(url, type_name, bbox, max_features=50000):
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
            data  = r.json()
            feats = data.get('features', [])
            if not feats:
                return pd.DataFrame()

            # Detecta CRS declarado pelo servidor (ex: "urn:ogc:def:crs:EPSG::4674")
            # GeoJSON padrão é 4326; TerraBrasilis usa 4674 (SIRGAS 2000)
            crs_raw = (
                data.get('crs', {})
                    .get('properties', {})
                    .get('name', 'EPSG:4326')
            )
            # Normaliza URN para código EPSG legível pelo GeoPandas
            # ex: "urn:ogc:def:crs:EPSG::4674" → "EPSG:4674"
            if 'EPSG::' in crs_raw:
                crs_str = 'EPSG:' + crs_raw.split('EPSG::')[-1]
            elif 'EPSG:' in crs_raw:
                crs_str = 'EPSG:' + crs_raw.split('EPSG:')[-1]
            else:
                crs_str = 'EPSG:4326'

            rows = []
            for f in feats:
                row = dict(f['properties'])
                row['geometry'] = f.get('geometry')   # dict GeoJSON ou None
                row['_wfs_crs'] = crs_str              # CRS para uso em clip
                rows.append(row)

            return pd.DataFrame(rows)
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

# =====================================
# FUNÇÕES MAPBIOMAS
# =====================================

@st.cache_data(ttl=300, show_spinner=False)
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
        metadata { totalCount totalPages currentPage limitValue }
        summary {
          total area
          alertsByYear { year value }
          deforestationAreaByYear { year value }
        }
        collection {
          alertCode areaHa detectedAt publishedAt sources
          deforestationClasses statusName crossedBiomes crossedStates
        }
      }
    }
    """
    variables = {"boundingBox": bbox, "startDate": start_date, "endDate": end_date, "limit": 1000, "page": 1}
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
# CONFIGURAÇÃO DE CORES E ESTILOS
# =====================================

ACTIVITY_COLORS = {
    'REDD':        "#f79c3c",
    'IFM':         "#3b8fbf",
    'ARR':         "#78cafe",
    'ACoGS':       "#ffffcc",
    'ACoGS; REDD': "#c7e9b4",
    'ALM':         "#7fcdbb",
    'IFM; REDD':   "#225ea8",
    'Unknown':     "#808080"
}

#STATUS_COLORS = {
#    'Registered':        "#2ecc71",
#    'Under Validation':  "#f39c12",
#    'Under Development': "#3498db",
#    'Inactive':          "#95a5a6",
#    'Unknown':           "#808080"
#}



# Dicionário de cores para cada status
STATUS_COLORS = {
    'Inactive': 'red', #OK
    'On Hold - see notification letter': 'yellow', #OK
    'Registered': '#266b2e', #OK
    'Registration and verification approval requested': '#3e8947', #OK
    'Registration requested': '#5bac78', #OK
    'Rejected by Administrator': 'red', #OK
    'Under development': '#a5d6c9', #OK
    'Under validation': '#7ac0a4', #OK
    'Verification approval requested': '#15431d', #OK
    'Withdrawn': 'red', #OK
    'Registration request denied': 'red', #?
    'Verification approval request denied': 'red', #?
    'Registration and verification approval request denied': 'red', #OK
    'Late to verify': 'red',
    'Unknown':           "#808080"
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
    # [MELHORIA] Antes: st.cache_data.clear() apagava TODOS os caches,
    # incluindo geometrias do GCS que acabam de ser baixadas.
    # Agora invalida seletivamente apenas os dados do Google Drive.
    load_parquet_from_gdrive.clear()
    st.rerun()

st.sidebar.divider()

st.markdown("""
    <style>
        /* [MELHORIA] Antes: display:none em TODOS os inputs e buttons da sidebar —
           qualquer novo controle adicionado ficaria invisível sem razão aparente.
           Agora ocultamos apenas os campos de ID de arquivo pelo atributo aria-label,
           preservando visibilidade de futuros controles adicionados à sidebar. */
        [data-testid="stSidebar"] [data-testid="stTextInput"]:has(input[aria-label="ID - Todos os Projetos"]),
        [data-testid="stSidebar"] [data-testid="stTextInput"]:has(input[aria-label="ID - Projetos com Créditos"]) {
            display: none !important;
        }
    </style>
""", unsafe_allow_html=True)

# =====================================
# CARREGAMENTO DE DADOS
# =====================================

with st.spinner("📥 Carregando dados..."):
    df_all    = load_parquet_from_gdrive(file_id_all)
    df_credit = load_parquet_from_gdrive(file_id_credit)

if df_all is None or df_credit is None:
    st.error("❌ Não foi possível carregar os dados. Verifique os IDs dos arquivos.")
    st.stop()

if "Vintage" not in df_credit.columns:
    if {"vintageStart", "vintageEnd"}.issubset(df_credit.columns):
        df_credit["Vintage"] = (
            df_credit["vintageStart"].astype(str).str[:4] + " e " +
            df_credit["vintageEnd"].astype(str).str[:4]
        )
    else:
        df_credit["Vintage"] = np.nan

st.sidebar.success("✅ Dados carregados com sucesso!")
st.sidebar.metric("Total de Projetos", f"{len(df_all):,}")
st.sidebar.metric(
    "Projetos que já emitiram VCUs",
    f"{df_credit['resourceName_x'].nunique():,}" if 'resourceName_x' in df_credit.columns else f"{len(df_credit):,}"
)
st.sidebar.caption(f"Última atualização: {datetime.now().strftime('%d/%m/%Y %H:%M')}")

# =====================================
# HEADER PRINCIPAL
# =====================================

st.title("🌎 Carbon4Flow")
st.markdown("""🎯 O protótipo que revela onde estão, quanto geram de créditos e os cruzamentos espaciais dos projetos de Carbono (AFOLU) no Brasil.""")
st.markdown("""🎲 Base de dados:  📥Verra  |  🔎Terrabrasils 🔴Prodes¹ e 🟠Deter¹ consultados via (OWS)  |  🌳 GFW¹ (api-tiles)""")
st.markdown("""
    ⚠️ **¹Necessita de validação dados espaciais**, em breve buscaremos ⭐new features de +validação, mais bases e novas funcionalidades.  \nDesenvolvedor: Edriano Souza. Reporting Issues: edriano.souza@ipam.org.br
""")
st.markdown("""🚩É um protótipo e os resultados necessitam de validações! || 😊+Recente: 📈 Análises Safras/Vintage e 📖 Storytelling > 📍Dados AOI""")

if 'selected_state_overview' not in st.session_state:
    st.session_state.selected_state_overview = None

# =====================================
# ABAS PRINCIPAIS
# =====================================

tabs = st.tabs([
    "📊 Visão Geral",
    "🌎 [POI] Projetos no Brasil",
    "💰 [POI] Projetos com lastro de créditos",
    "📈 Análises Safras/Vintage",
    "📖 Storytelling",
    "📁 Dados Brutos"
])

# =====================================
# ABA 1: VISÃO GERAL
# =====================================

with tabs[0]:
    st.header("📊 Visão Geral dos Projetos")

    df_overview = df_all
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
        credit_count = (
            len(df_credit[df_credit["resourceName_x"].isin(df_overview["resourceName_x"])])
            if st.session_state.selected_state_overview else len(df_credit)
        )
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
            fig_status = px.bar(
                status_counts, x="Status", y="Quantidade",
                color="Status", color_discrete_map=STATUS_COLORS, text="Quantidade"
            )
            fig_status.update_traces(textposition='outside')
            fig_status.update_layout(showlegend=False, height=600, xaxis_title="", yaxis_title="Número de Projetos")
            st.plotly_chart(fig_status, use_container_width=True)

    with col_right:
        st.subheader("🌳 Distribuição por Atividade AFOLU")
        if "vcsAFOLUActivity" in df_overview.columns:
            activity_counts = df_overview["vcsAFOLUActivity"].value_counts().reset_index()
            activity_counts.columns = ["Atividade", "Quantidade"]
            fig_activity = px.pie(
                activity_counts, names="Atividade", values="Quantidade",
                color="Atividade", color_discrete_map=ACTIVITY_COLORS, hole=0.4
            )
            fig_activity.update_traces(textposition='inside', textinfo='percent+label')
            fig_activity.update_layout(height=500)
            st.plotly_chart(fig_activity, use_container_width=True)

    st.divider()
    st.subheader("🗺️ Distribuição por Estado")

    if "state_Recode" in df_overview.columns:
        state_counts = df_overview["state_Recode"].value_counts().head(10).reset_index()
        state_counts.columns = ["Estado", "Quantidade"]
        fig_states = px.bar(
            state_counts, x="Quantidade", y="Estado", orientation='h',
            text="Quantidade", color="Quantidade", color_continuous_scale="Viridis"
        )
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

    # [MELHORIA] Função auxiliar deduplica o HTML de popup que antes era
    # copiado identicamente nos blocos "Clusters" e "Pontos".
    def _popup_html(row) -> str:
        return f"""
        <div style="font-family: Arial; font-size: 11px; width: 250px;">
            <h4 style="margin: 0 0 15px 0;">{row.get('resourceName_x', 'N/A')}</h4>
            <b>ID:</b> {row.get('resourceIdentifier', 'N/A')}<br>
            <b>Status:</b> {row.get('vcsProjectStatus', 'N/A')}<br>
            <b>Estado:</b> {row.get('state_Recode', 'N/A')}<br>
            <b>Protocolos:</b> {row.get('vcsMethodology', 'N/A')}<br>
            <b>Tipo:</b> {row.get('vcsAFOLUActivity', 'N/A')}<br>
            <b>Acreditação:</b> {row.get('vcsCreditingPeriodTerm', 'N/A')}<br>
            <b>Area:</b> {row.get('vcsAcresHectares', 'N/A')}<br>
            <b>EAER:</b> {row.get('vcsEstimatedAnnualEmissionReductions', 'N/A')}
        </div>
        """

    center = [df_map["new_latitude"].mean(), df_map["new_longitude"].mean()]
    m = folium.Map(location=center, zoom_start=4, tiles="CartoDB dark_matter")

    # [MELHORIA] Cap de 300 marcadores no modo Pontos para evitar HTML pesado
    # que pode travar o browser com datasets grandes.
    _PONTOS_CAP = 300

    if map_type == "Clusters":
        marker_cluster = MarkerCluster().add_to(m)
        for _, row in df_map.iterrows():
            lat, lon = row["new_latitude"], row["new_longitude"]
            activity = row.get("vcsAFOLUActivity", "Unknown")
            color    = ACTIVITY_COLORS.get(activity, "#808080")
            folium.CircleMarker(
                location=[lat, lon], radius=6, color=color, fill=True,
                fill_color=color, fill_opacity=0.7,
                popup=folium.Popup(_popup_html(row), max_width=300)
            ).add_to(marker_cluster)

    elif map_type == "Heatmap":
        HeatMap(
            [[row["new_latitude"], row["new_longitude"]] for _, row in df_map.iterrows()],
            radius=15
        ).add_to(m)

    else:
        df_pontos = df_map
        if len(df_map) > _PONTOS_CAP:
            st.warning(
                f"⚠️ Modo **Pontos** limitado a {_PONTOS_CAP} projetos para evitar lentidão no browser "
                f"({len(df_map):,} projetos no filtro atual). Use **Clusters** para visualizar todos."
            )
            df_pontos = df_map.head(_PONTOS_CAP)
        for _, row in df_pontos.iterrows():
            activity = row.get("vcsAFOLUActivity", "Unknown")
            color    = ACTIVITY_COLORS.get(activity, "#808080")
            folium.CircleMarker(
                location=[row["new_latitude"], row["new_longitude"]],
                radius=5, color=color, fill=True, fill_color=color, fill_opacity=0.6,
                popup=folium.Popup(_popup_html(row), max_width=300)
            ).add_to(m)

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

@st.cache_data(show_spinner=False)
def get_credit_unique(df):
    return df.groupby('resourceName_x').first().reset_index()

df_credit_unique = get_credit_unique(df_credit)

with tabs[2]:
    st.header("💰 Mapa - Projetos já emitiram Créditos")
    st.info(f"📊 Exibindo **{len(df_credit_unique):,}** projetos únicos (de {len(df_credit):,} registros totais)")
    create_interactive_map(df_credit_unique, "Projetos com lastro de Créditos", "credit_unique")

# =====================================
# ABA 4: ANÁLISE DE VINTAGE
# =====================================

with tabs[3]:
    st.header("📈 Análise VCUs (Unidades de Carbono Verificadas) por Safra/Vintage")

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
                    estado_sel  = st.selectbox("📍 Selecione o Estado:", estados, key="vintage_state")
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
                    df_proj     = dados_estado[dados_estado['resourceName_x'] == projeto_sel]
                else:
                    st.warning("Nenhum projeto disponível para este estado")
                    df_proj = pd.DataFrame()
            else:
                df_proj = pd.DataFrame()

        if df_proj.empty:
            st.info("Selecione um estado e projeto para visualizar a análise")
        else:
            col_m1, col_m3, col_m4, col_m5, col_m6 = st.columns(5)
            with col_m1:
                if 'Mean' in df_proj.columns:
                    st.metric("Média VCUs", f"{df_proj['Mean'].iloc[0]:,.0f} ± {df_proj['IC_Mais'].iloc[0] - df_proj['Mean'].iloc[0]:,.0f}")
            with col_m3:
                if 'protocol' in df_proj.columns:
                    st.metric("Protocolo", df_proj['protocol'].iloc[0], delta_color="off")
            with col_m4:
                if 'vcsProjectStatus' in df_proj.columns:
                    st.metric("Status", df_proj['vcsProjectStatus'].iloc[0], delta_color="off")
            with col_m5:
                if 'resourceIdentifier' in df_proj.columns:
                    st.metric("ID", df_proj['resourceIdentifier'].iloc[0], delta_color="off")
            with col_m6:
                if 'vcsAFOLUActivity' in df_proj.columns:
                    st.metric("Tipo", df_proj['vcsAFOLUActivity'].iloc[0], delta_color="off")

            st.divider()

            col_graf1, col_graf2 = st.columns([3, 1.5])

            with col_graf1:
                fig = go.Figure()
                fig.add_trace(go.Bar(x=df_proj['Ano_Periodo'], y=df_proj['TotalVintageQuantity'], name='Total Vintage', marker_color='#1E800A'))
                fig.add_trace(go.Bar(x=df_proj['Ano_Periodo'], y=df_proj['SumQuantity'],          name='Sum Quantity',  marker_color='#6DD458'))
                fig.add_trace(go.Bar(x=df_proj['Ano_Periodo'], y=df_proj['Sum_Retired'],           name='Retired',       marker_color='#FFC2A3'))
                if 'Mean' in df_proj.columns:
                    fig.add_trace(go.Scatter(
                        x=df_proj['Ano_Periodo'], y=df_proj['Mean'], mode='lines+markers',
                        name='Média', line=dict(color='#1E800A', width=3), marker=dict(size=8)
                    ))
                if 'IC_Mais' in df_proj.columns and 'IC_Menos' in df_proj.columns:
                    fig.add_trace(go.Scatter(x=df_proj['Ano_Periodo'], y=df_proj['IC_Mais'], mode='lines', name='IC Superior', line=dict(color='gray', width=2, dash='dot')))
                    fig.add_trace(go.Scatter(x=df_proj['Ano_Periodo'], y=df_proj['IC_Menos'], mode='lines', name='IC Inferior', line=dict(color='gray', width=2, dash='dot')))
                fig.update_layout(
                    barmode='group', title=f"Análise de VCUs - {projeto_sel}",
                    xaxis_title="Período (Ano)", yaxis_title="Quantidade de VCUs",
                    legend_title="Métricas", template="plotly_white", height=450,
                    legend=dict(font=dict(size=10), orientation='v', xanchor='auto', yanchor='auto'),
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)

            with col_graf2:
                metricas_totais = {
                    'TotalVintageQuantity': '#1E800A',
                    'SumQuantity':          '#6DD458',
                    'Sum_Retired':          '#FFC2A3',
                    'Sum_Active':           '#A3C4F3'
                }
                labels, valores, cores = [], [], []
                for col_name, cor in metricas_totais.items():
                    if col_name in df_proj.columns:
                        labels.append(col_name.replace('Sum_','').replace('Sum','').replace('Total','Total '))
                        valores.append(pd.to_numeric(df_proj[col_name], errors='coerce').sum())
                        cores.append(cor)
                fig_tot = go.Figure()
                fig_tot.add_trace(go.Bar(
                    x=labels, y=valores, marker_color=cores,
                    text=[f"{v:,.0f}" for v in valores], textposition='outside',
                    textfont=dict(size=10), showlegend=False
                ))
                fig_tot.update_layout(
                    title="Totais Acumulados", template="plotly_white", height=450,
                    margin=dict(t=50, b=40, l=40, r=20), yaxis=dict(showticklabels=False)
                )
                st.plotly_chart(fig_tot, use_container_width=True)

            with st.expander("📊 Ver Tabela de Dados Detalhada"):
                display_cols = ['Ano_Periodo', 'TotalVintageQuantity', 'SumQuantity',
                                'Sum_Retired', 'Sum_Active', 'Mean', 'IC_Mais', 'IC_Menos']
                display_cols = [col for col in display_cols if col in df_proj.columns]
                st.dataframe(
                    df_proj[display_cols].style.format({col: "{:,.0f}" for col in display_cols if col != 'Ano_Periodo'}),
                    use_container_width=True
                )

            # =====================================
            # SEÇÃO: TRANSAÇÕES | FLOW
            # =====================================

            st.divider()
            st.subheader("💸 Transações | Flow")
            st.caption(f"Transações granulares de `df_credit` filtradas por: **{estado_sel}** → **{projeto_sel}**")

            df_flow = df_credit[df_credit['resourceName_x'] == projeto_sel].copy()

            if df_flow.empty:
                st.warning("⚠️ Nenhuma transação encontrada para este projeto.")
            else:
                col_fl1, col_fl2, col_fl3, col_fl4 = st.columns(4)
                total_qty = pd.to_numeric(df_flow['quantity'], errors='coerce').sum()

                if 'retiredCancelled' in df_flow.columns:
                    mask_ret    = df_flow['retiredCancelled'] == True
                    qty_retired = pd.to_numeric(df_flow.loc[mask_ret,  'quantity'], errors='coerce').sum()
                    qty_active  = pd.to_numeric(df_flow.loc[~mask_ret, 'quantity'], errors='coerce').sum()
                else:
                    qty_retired = 0
                    qty_active  = total_qty

                pct_retired = (qty_retired / total_qty * 100) if total_qty > 0 else 0

                with col_fl1:
                    st.metric("📦 Total Transações", f"{len(df_flow):,}")
                with col_fl2:
                    st.metric("🔢 Quantidade Total (VCUs)", f"{total_qty:,.0f}")
                with col_fl3:
                    st.metric("✅ Ativos", f"{qty_active:,.0f}", delta=f"{100 - pct_retired:.1f}%", delta_color="normal")
                with col_fl4:
                    st.metric("🔴 Retired / Cancelled", f"{qty_retired:,.0f}", delta=f"{pct_retired:.1f}%", delta_color="inverse")

                st.divider()

                if 'Vintage' in df_flow.columns and 'retiredCancelled' in df_flow.columns:
                    df_flow['quantity_num'] = pd.to_numeric(df_flow['quantity'], errors='coerce')
                    df_flow['Status_Label'] = df_flow['retiredCancelled'].apply(
                        lambda x: '🔴 Retired/Cancelled' if x == True else '✅ Ativo'
                    )
                    df_flow['Ano_Periodo'] = df_flow['Vintage'].apply(
                        lambda x: (
                            f"{x.split(' e ')[0][:4]}-{x.split(' e ')[1][:4]}"
                            if isinstance(x, str) and ' e ' in x
                            else str(x)[:4] if pd.notna(x) else 'N/A'
                        )
                    )
                    df_bar = df_flow.groupby(['Ano_Periodo', 'Status_Label']).agg(
                        Quantidade=('quantity_num', 'sum')
                    ).reset_index()

                    fig_flow = px.bar(
                        df_bar, x='Ano_Periodo', y='Quantidade', color='Status_Label',
                        color_discrete_map={'✅ Ativo': '#2ecc71', '🔴 Retired/Cancelled': '#e74c3c'},
                        barmode='stack',
                        title=f"Distribuição de VCUs por Vintage — {projeto_sel}",
                        labels={'Ano_Periodo': 'Período (Vintage)', 'Quantidade': 'VCUs', 'Status_Label': 'Status'}
                    )
                    fig_flow.update_layout(
                        template='plotly_white', height=380, hovermode='x unified',
                        legend=dict(font=dict(size=10), orientation='v', xanchor='auto', yanchor='auto'),
                        margin=dict(t=60, b=40, l=60, r=20)
                    )
                    st.plotly_chart(fig_flow, use_container_width=True)

                st.divider()

                with st.expander("📋 Ver Tabela de Transações (Granular)", expanded=True):
                    cols_flow = [
                        'issuanceDate', 'tbl_type', 'resourceName_y', 'quantity',
                        'retiredCancelled', 'additionalCertifications',
                        'retirementBeneficiary', 'retirementReason', 'retirementDetails', 'Vintage'
                    ]
                    cols_flow_exist   = [c for c in cols_flow if c in df_flow.columns]
                    df_flow_display   = df_flow[cols_flow_exist]
                    st.dataframe(df_flow_display, use_container_width=True, height=420)

                    csv_flow = df_flow_display.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="⬇️ Download Transações (CSV)", data=csv_flow,
                        file_name=f"transacoes_{projeto_sel[:40].replace(' ', '_')}.csv",
                        mime="text/csv"
                    )

                    st.divider()
                    st.markdown("### 🛒 Quem Comprou? — Top 15 Beneficiários")

                    if 'retirementBeneficiary' in df_flow.columns and 'quantity' in df_flow.columns:
                        df_buyers = df_flow.copy()
                        df_buyers['quantity_num'] = pd.to_numeric(df_buyers['quantity'], errors='coerce')
                        if 'retiredCancelled' in df_buyers.columns:
                            df_buyers = df_buyers[df_buyers['retiredCancelled'] == True]
                        df_buyers = df_buyers.dropna(subset=['retirementBeneficiary', 'quantity_num'])
                        df_buyers = df_buyers[df_buyers['retirementBeneficiary'].str.strip() != '']

                        top_buyers = (
                            df_buyers.groupby('retirementBeneficiary')['quantity_num']
                            .sum().sort_values(ascending=True).tail(15).reset_index()
                        )
                        top_buyers.columns = ['Beneficiário', 'Total_VCUs']

                        if top_buyers.empty:
                            st.info("Nenhum dado de beneficiário disponível para este projeto.")
                        else:
                            total_vcus      = top_buyers['Total_VCUs'].sum()
                            top_buyers['Pct']   = (top_buyers['Total_VCUs'] / total_vcus * 100).round(1)
                            top_buyers['Label'] = top_buyers.apply(
                                lambda r: f"{r['Total_VCUs']:,.0f} VCUs ({r['Pct']}%)", axis=1
                            )

                            fig_buyers = go.Figure()
                            fig_buyers.add_trace(go.Bar(
                                x=top_buyers['Total_VCUs'], y=top_buyers['Beneficiário'],
                                orientation='h', text=top_buyers['Label'], textposition='outside',
                                marker=dict(color=top_buyers['Total_VCUs'], colorscale='Greens', showscale=False),
                                hovertemplate='<b>%{y}</b><br>VCUs: %{x:,.0f}<extra></extra>'
                            ))
                            fig_buyers.update_layout(
                                template='plotly_white',
                                height=max(350, len(top_buyers) * 42),
                                margin=dict(t=20, b=40, l=260, r=160),
                                xaxis=dict(title='Quantidade de VCUs', showgrid=True, gridcolor='#eeeeee'),
                                yaxis=dict(title='', tickfont=dict(size=12)),
                                bargap=0.3
                            )
                            st.plotly_chart(fig_buyers, use_container_width=True)

                            with st.expander("📊 Ver ranking completo"):
                                st.dataframe(
                                    top_buyers[['Beneficiário', 'Total_VCUs', 'Pct']]
                                    .sort_values('Total_VCUs', ascending=False)
                                    .rename(columns={'Total_VCUs': 'VCUs Comprados', 'Pct': '% do Total'})
                                    .style.format({'VCUs Comprados': '{:,.0f}', '% do Total': '{:.1f}%'})
                                    .background_gradient(subset=['VCUs Comprados'], cmap='Greens'),
                                    use_container_width=True, hide_index=True
                                )
                    else:
                        st.info("Colunas `retirementBeneficiary` ou `quantity` não encontradas no dataset.")

# =====================================
# ABA 5: STORYTELLING
# =====================================

with tabs[4]:
    st.header("📖 A História dos Projetos de Carbono no Brasil")

    story_tabs = st.tabs([
        "🌎 Panorama Geral",
        "📍 Dados AOI",
        "🌿 MapBiomas",
        "📊 Evolução Temporal",
        "🎯 Impacto Regional",
        "💡 Insights"
    ])

    # ── STORYTELLING 0: PANORAMA GERAL ──────────────────────────────────
    with story_tabs[0]:
        st.markdown("## 🌱 A Jornada do Carbono Florestal Brasileiro")

        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("""
            ### Do Desmatamento aos Créditos de Carbono
            - ⬇️ **Reduções** e suas Unidades de Carbono Verificadas (VCUs)            
            - 💰 **Valoriza economicamente** a floresta em pé
            - 🌳 **Preserva a biodiversidade**
            - 👥 **Beneficia comunidades** locais
            - 🗺️ **Combate as mudanças climáticas** globais
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
            fig_timeline = px.area(
                timeline_data, x='Ano', y='Quantidade', color='vcsAFOLUActivity',
                color_discrete_map=ACTIVITY_COLORS,
                title='Crescimento dos Projetos de Carbono ao Longo do Tempo'
            )
            fig_timeline.update_layout(hovermode='x unified', height=400, legend_title_text='Tipo de Atividade')
            st.plotly_chart(fig_timeline, use_container_width=True)

        st.markdown("### 🎯 Números que Contam Histórias")
        col_m1, col_m3 = st.columns(2)

        with col_m1:
            if 'vcsAcresHectares' in df_all.columns:
                try:
                    df_all_temp = df_all.copy()
                    df_all_temp['area_num'] = pd.to_numeric(
                        df_all_temp['vcsAcresHectares'].astype(str)
                        .str.replace(r'[^\d.,]', '', regex=True).str.replace(',', '', regex=False),
                        errors='coerce'
                    )
                    st.metric("Área Total Protegida", f"{df_all_temp['area_num'].sum()/1_000_000:,.1f}M ha")
                except Exception:
                    st.metric("Área Total Protegida", "N/A")

        with col_m3:
            try:
                redd_count = len(df_all[df_all['vcsAFOLUActivity'].str.contains('REDD', na=False)])
                st.metric("Projetos REDD+", f"{redd_count}")
            except Exception:
                st.metric("Projetos REDD+", "N/A")

    # ── STORYTELLING 1: DADOS AOI ────────────────────────────────────────
    with story_tabs[1]:
        render_aoi_tab(
            df_all              = df_all,
            GFW_API_KEY         = st.secrets["GFW_API_KEY"].strip(),
            gfw_tree_cover_loss = gfw_tree_cover_loss,
            gfw_glad_alerts     = gfw_glad_alerts,
            gfw_radd_alerts     = gfw_radd_alerts,
            terrabrasilis_wfs   = terrabrasilis_wfs,
            ACTIVITY_COLORS     = ACTIVITY_COLORS,
        )

    # ── STORYTELLING 2: MAPBIOMAS (em desenvolvimento) ───────────────────
    with story_tabs[2]:
        st.markdown("## 🌿 Alertas MapBiomas")
        st.info("🚧 Em desenvolvimento. Em breve disponível.")

    # ── STORYTELLING 3: EVOLUÇÃO TEMPORAL ───────────────────────────────
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
                else (int(str(x)[:4]) if pd.notna(x) else None)
            )
            df_timing = df_timing.dropna(subset=['vcsRegistrationDate', 'Vintage_Year'])
            df_timing['Registration_Year']  = df_timing['vcsRegistrationDate'].dt.year
            df_timing['Years_to_Credit']    = df_timing['Vintage_Year'] - df_timing['Registration_Year']
            df_timing = df_timing[(df_timing['Years_to_Credit'] >= -5) & (df_timing['Years_to_Credit'] <= 10)]

            if len(df_timing) > 0:
                fig_timing = px.histogram(
                    df_timing, x='Years_to_Credit', nbins=20,
                    title='Distribuição do Tempo entre Registro e Emissão de Créditos',
                    color_discrete_sequence=['#26a69a']
                )
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
                else (int(str(x)[:4]) if pd.notna(x) else None)
            )
            df_credits_year = df_credits_year.dropna(subset=['Vintage_Year'])
            df_credits_year['totalVintageQuantity'] = pd.to_numeric(df_credits_year['totalVintageQuantity'], errors='coerce')
            credits_by_year = df_credits_year.groupby('Vintage_Year').agg(
                Total_Creditos=('totalVintageQuantity', 'sum'),
                Num_Projetos=('resourceName_x', 'nunique')
            ).reset_index().rename(columns={'Vintage_Year': 'Ano'})
            credits_by_year = credits_by_year[credits_by_year['Ano'] >= 2000]

            fig_credits_evolution = go.Figure()
            fig_credits_evolution.add_trace(go.Bar(
                x=credits_by_year['Ano'], y=credits_by_year['Total_Creditos'],
                name='Créditos Emitidos', marker_color='#26a69a', yaxis='y'
            ))
            fig_credits_evolution.add_trace(go.Scatter(
                x=credits_by_year['Ano'], y=credits_by_year['Num_Projetos'],
                name='Número de Projetos', marker_color='#ff6b6b',
                mode='lines+markers', yaxis='y2'
            ))
            fig_credits_evolution.update_layout(
                title='Emissão de Créditos e Número de Projetos ao Longo do Tempo',
                xaxis_title='Ano', yaxis_title='Total de Créditos (VCUs)',
                yaxis2=dict(title='Número de Projetos', overlaying='y', side='right'),
                height=500, hovermode='x unified'
            )
            st.plotly_chart(fig_credits_evolution, use_container_width=True)

    # ── STORYTELLING 4: IMPACTO REGIONAL ────────────────────────────────
    with story_tabs[4]:
        st.markdown("## 🎯 O Impacto nos Territórios")

        if 'state_Recode' in df_all.columns:
            st.markdown("### 🗺️ Densidade de Projetos por Região")
            state_summary = df_all.groupby('state_Recode').agg({
                'resourceName_x':                      'count',
                'vcsAcresHectares':                    lambda x: pd.to_numeric(x, errors='coerce').sum(),
                'vcsEstimatedAnnualEmissionReductions': lambda x: pd.to_numeric(x, errors='coerce').sum()
            }).reset_index()
            state_summary.columns = ['Estado', 'Num_Projetos', 'Area_Total', 'EAER_Total']
            state_summary = state_summary.sort_values('Num_Projetos', ascending=False).head(15)

            fig_regional = px.scatter(
                state_summary, x='Area_Total', y='EAER_Total',
                size='Num_Projetos', color='Num_Projetos', hover_name='Estado',
                title='Relação entre Área, Impacto e Número de Projetos',
                color_continuous_scale='Viridis', size_max=60
            )
            fig_regional.update_layout(height=500)
            st.plotly_chart(fig_regional, use_container_width=True)

        st.divider()
        st.markdown("### 🌳 Perfil de Atividades por Estado")

        if 'state_Recode' in df_all.columns and 'vcsAFOLUActivity' in df_all.columns:
            top_states        = df_all['state_Recode'].value_counts().head(10).index.tolist()
            df_activity_state = df_all[df_all['state_Recode'].isin(top_states)]
            activity_by_state = df_activity_state.groupby(['state_Recode', 'vcsAFOLUActivity']).size().reset_index(name='Count')
            fig_activity_state = px.bar(
                activity_by_state, x='state_Recode', y='Count',
                color='vcsAFOLUActivity', color_discrete_map=ACTIVITY_COLORS,
                title='Distribuição de Tipos de Atividade nos Principais Estados',
                barmode='stack'
            )
            fig_activity_state.update_layout(height=500)
            st.plotly_chart(fig_activity_state, use_container_width=True)

    # ── STORYTELLING 5: INSIGHTS ─────────────────────────────────────────
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
                fig_corr = px.scatter(
                    df_correlation, x='Area', y='EAER', color='vcsAFOLUActivity',
                    hover_data=['resourceName_x', 'state_Recode'],
                    color_discrete_map=ACTIVITY_COLORS,
                    title='Relação entre Tamanho do Projeto e Impacto Climático',
                    log_x=True, log_y=True
                )
                fig_corr.update_layout(height=500)
                st.plotly_chart(fig_corr, use_container_width=True)

# =====================================
# ABA 6: DADOS BRUTOS
# =====================================

with tabs[5]:
    st.header("📁 Visualização dos Dados Brutos")

    data_option = st.radio(
        "Selecione o conjunto de dados:",
        ["Todos os Projetos", "Projetos com Créditos"], horizontal=True
    )
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
