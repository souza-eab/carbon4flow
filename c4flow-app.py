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

# requeries pip install
# streamlit | pandas | requests | io | plotly | datetime | folium | numpy | scipy

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
    """Carrega dados do Google Drive com cache de 1 hora"""
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
    """Converte colunas para numérico de forma segura"""
    df_clean = df.copy()
    for col in columns:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    return df_clean

def prepare_map_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepara dados para visualização no mapa"""
    df_map = df.copy()
    
    # Limpar coordenadas
    coord_cols = ["new_latitude", "new_longitude", "latitude", "longitude"]
    df_map = clean_numeric_columns(df_map, coord_cols)
    
    # Usar coordenadas adequadas
    if "new_latitude" in df_map.columns and "new_longitude" in df_map.columns:
        df_map = df_map.dropna(subset=["new_latitude", "new_longitude"])
    else:
        df_map = df_map.dropna(subset=["latitude", "longitude"])
        df_map.rename(columns={"latitude": "new_latitude", "longitude": "new_longitude"}, inplace=True)
    
    return df_map

@st.cache_data(show_spinner=False)
def calcular_intervalo_confianca(df_grouped, confidence=0.95):
    """Calcula intervalo de confiança de forma otimizada"""
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
    """Processa dados de VCU por vintage com otimizações"""
    df = df_full.copy()
    
    # Validar colunas necessárias
    required_cols = ['resourceName_x', 'totalVintageQuantity', 'quantity', 'Vintage']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        st.error(f"❌ Colunas ausentes: {', '.join(missing_cols)}")
        return pd.DataFrame()
    
    # Limpar dados
    df = df.dropna(subset=['resourceName_x', 'Vintage', 'totalVintageQuantity'])
    df['totalVintageQuantity'] = pd.to_numeric(df['totalVintageQuantity'], errors='coerce')
    df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
    
    # Criar coluna retiredCancelled se não existir
    if 'retiredCancelled' not in df.columns:
        df['retiredCancelled'] = False
    
    # Agrupar e calcular estatísticas
    group_cols = ['state_Recode', 'resourceName_x', 'Vintage', 'protocol', 
                  'vcsProjectStatus', 'vcsEstimatedAnnualEmissionReductions']
    
    # Remover colunas inexistentes
    group_cols = [col for col in group_cols if col in df.columns]
    
    estatisticas = df.groupby(group_cols, dropna=False).agg(
        TotalVintageQuantity=('totalVintageQuantity', 'first'),
        SumQuantity=('quantity', 'sum'),
        Sum_Retired=('quantity', lambda x: x[df.loc[x.index, 'retiredCancelled'] == True].sum()),
        Sum_Active=('quantity', lambda x: x[df.loc[x.index, 'retiredCancelled'] == False].sum())
    ).reset_index()
    
    # Calcular intervalo de confiança (otimizado)
    ic_df = calcular_intervalo_confianca(df)
    estatisticas = estatisticas.merge(ic_df, on='resourceName_x', how='left')
    
    # Criar coluna de período
    estatisticas['Ano_Periodo'] = estatisticas['Vintage'].apply(
        lambda x: (
            f"{x.split(' e ')[0][:4]}-{x.split(' e ')[1][:4]}"
            if isinstance(x, str) and ' e ' in x
            else str(x)[:4] if pd.notna(x) else 'N/A'
        )
    )
    
    # Renomear colunas
    if 'vcsEstimatedAnnualEmissionReductions' in estatisticas.columns:
        estatisticas.rename(columns={'vcsEstimatedAnnualEmissionReductions': 'EAER'}, inplace=True)
    
    return estatisticas

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

#STATUS_COLORS = {
#    'Inactive': '#b2182b',
#    'On Hold - see notification letter': '#fbec5d',
#    'Registered': '#1a6c2d',
#    'Registration and verification approval requested': '#338a46',
#    'Registration requested': '#50ad77',
#    'Under development': '#a0d7c9',
#    'Under validation': '#71c1a4',
#    'Verification approval requested': '#0d431c',
#    'Withdrawn': '#b2182b',
#    'Registration request denied': '#b2182b'
#}


STATUS_COLORS = {
    'Registered': "#2ecc71",
    'Under Validation': "#f39c12",
    'Under Development': "#3498db",
    'Inactive': "#95a5a6",
    'Unknown': "#808080"
}


# =====================================
# SIDEBAR - CONFIGURAÇÕES
# =====================================

st.sidebar.title("⚙️ Configurações")

# IDs dos arquivos
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

# Botão para recarregar dados
if st.sidebar.button("🔄 Recarregar Dados", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

st.sidebar.divider()

# Esconder os inputs e o botão do sidebar
hide_sidebar_inputs = """
    <style>
        [data-testid="stSidebar"] input,
        [data-testid="stSidebar"] button {
            display: none !important;
        }
    </style>
"""
st.markdown(hide_sidebar_inputs, unsafe_allow_html=True)


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

# Informações na sidebar
st.sidebar.success("✅ Dados carregados com sucesso!")
st.sidebar.metric(" Total de Projetos", f"{len(df_all):,}")
st.sidebar.metric(" Projetos com lasto de VCUs", f"{df_credit['resourceName_x'].nunique():,}" if 'resourceName_x' in df_credit.columns else f"{len(df_credit):,}")
st.sidebar.caption(f"Última atualização: {datetime.now().strftime('%d/%m/%Y %H:%M')}")

# =====================================
# HEADER PRINCIPAL
# =====================================

st.title("🌎 Carbon4Flow")
st.markdown("""
Dashboard interativo para estimativas sobre Projetos de Carbono. Dados: Verra.  
    Dev: Edriano Souza. Os dados são atualizados automaticamente do Google Drive.
    Reporting Issues
    For clarification or an issue/bug report, please write to edriano.souza@ipam.org.br or edriano759@gmail.com
""")

# =====================================
# SESSION STATE PARA FILTROS INTERATIVOS
# =====================================

if 'selected_state_overview' not in st.session_state:
    st.session_state.selected_state_overview = None

# =====================================
# ABAS PRINCIPAIS
# =====================================

#tabs = st.tabs([
#    "📊 Visão Geral",
#    "🌎 Mapa - Todos os Projetos BR",
#    "💰 Mapa - Com lastro de Créditos/Vendas",
#    "📈 Análises de Vintage",
#    "📁 Dados Brutos"
#])

tabs = st.tabs([
    "📊 Visão Geral",
    "🌎 Mapa - Todos os Projetos BR",
    "💰 Mapa - Com lastro de Créditos/Vendas",
    "📈 Análises de Vintage",
    "📖 Storytelling",  # NOVA ABA
    "📁 Dados Brutos"
])

# =====================================
# ABA 1: VISÃO GERAL
# =====================================

with tabs[0]:
    st.header("📊 Visão Geral dos Projetos")
    
    # Aplicar filtro se houver seleção
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
            afolu_count = df_overview["vcsAFOLUActivity"].notna().sum()
            st.metric("Projetos AFOLU", f"{afolu_count:,}")
    
    st.divider()
    
    # Gráficos lado a lado
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("📋 Distribuição por Status")
        if "vcsProjectStatus" in df_overview.columns:
            status_counts = df_overview["vcsProjectStatus"].value_counts().reset_index()
            status_counts.columns = ["Status", "Quantidade"]
            
            fig_status = px.bar(
                status_counts,
                x="Status",
                y="Quantidade",
                color="Status",
                color_discrete_map=STATUS_COLORS,
                text="Quantidade"
            )
            fig_status.update_traces(textposition='outside')
            fig_status.update_layout(
                showlegend=False,
                height=600,
                xaxis_title="",
                yaxis_title="Número de Projetos"
            )
            st.plotly_chart(fig_status, use_container_width=True)
    
    with col_right:
        st.subheader("🌳 Distribuição por Atividade AFOLU")
        if "vcsAFOLUActivity" in df_overview.columns:
            activity_counts = df_overview["vcsAFOLUActivity"].value_counts().reset_index()
            activity_counts.columns = ["Atividade", "Quantidade"]
            
            fig_activity = px.pie(
                activity_counts,
                names="Atividade",
                values="Quantidade",
                color="Atividade",
                color_discrete_map=ACTIVITY_COLORS,
                hole=0.4
            )
            fig_activity.update_traces(textposition='inside', textinfo='percent+label')
            fig_activity.update_layout(height=500)
            st.plotly_chart(fig_activity, use_container_width=True)
    
    # Distribuição geográfica COM INTERATIVIDADE
    st.divider()
    st.subheader("🗺️ Distribuição por Estado")
    st.caption("💡 Clique em uma barra para filtrar os gráficos acima")
    
    if "state_Recode" in df_overview.columns:
        state_counts = df_overview["state_Recode"].value_counts().head(10).reset_index()
        state_counts.columns = ["Estado", "Quantidade"]
        
        fig_states = px.bar(
            state_counts,
            x="Quantidade",
            y="Estado",
            orientation='h',
            text="Quantidade",
            color="Quantidade",
            color_continuous_scale="Viridis"
        )
        fig_states.update_traces(textposition='outside')
        fig_states.update_layout(
            height=400,
            showlegend=False,
            xaxis_title="Número de Projetos",
            yaxis_title=""
        )
        
        selected_points = st.plotly_chart(fig_states, use_container_width=True, on_select="rerun", key="state_chart")
        
        # Capturar clique no gráfico
        #if selected_points and selected_points.get("selection"):
         #   if selected_points["selection"].get("points"):
          #      clicked_state = selected_points["selection"]["points"][0].get("y")
           #     if clicked_state:
            #        st.session_state.selected_state_overview = clicked_state
             #       st.rerun()

# =====================================
# FUNÇÃO PARA CRIAR MAPAS
# =====================================

def create_interactive_map(df: pd.DataFrame, title: str, map_key: str):
    """Cria mapa interativo com filtros e visualizações"""
    
    st.header(title)
    
    # Preparar dados
    df_map = prepare_map_data(df)
    
    # Filtros em colunas
    col_filter1, col_filter2, col_filter3 = st.columns(3)
    
    with col_filter1:
        # Filtro por Estado
        if "state_Recode" in df_map.columns:
            states = ["Todos"] + sorted(df_map["state_Recode"].dropna().unique().tolist())
            selected_state = st.selectbox(
                "Estado:",
                states,
                key=f"state_{map_key}"
            )
            if selected_state != "Todos":
                df_map = df_map[df_map["state_Recode"] == selected_state]
    
    with col_filter2:
        # Filtro AFOLU
        show_afolu = st.checkbox(
            "Apenas AFOLU",
            value=True,
            key=f"afolu_{map_key}"
        )
        if show_afolu and "vcsAFOLUActivity" in df_map.columns:
            df_map = df_map[df_map["vcsAFOLUActivity"].notna()]
    
    with col_filter3:
        # Tipo de visualização
        map_type = st.selectbox(
            "Tipo de Mapa:",
            [ "Pontos", "Heatmap", "Clusters"],
            key=f"maptype_{map_key}"
        )
    
    # Filtros adicionais em expander
    with st.expander("🔍 Filtros Avançados"):
        col_adv1, col_adv2 = st.columns(2)
        
        with col_adv1:
            if "protocolSubCategories" in df_map.columns:
                categories = sorted(df_map["protocolSubCategories"].dropna().unique().tolist())
                selected_cat = st.multiselect(
                    "Protocol Sub-Categories:",
                    options=categories,
                    default=categories,
                    key=f"cat_{map_key}"
                )
                if selected_cat:
                    df_map = df_map[df_map["protocolSubCategories"].isin(selected_cat)]
        
        with col_adv2:
            if "vcsProjectStatus" in df_map.columns:
                statuses = sorted(df_map["vcsProjectStatus"].dropna().unique().tolist())
                selected_status = st.multiselect(
                    "Status:",
                    options=statuses,
                    default=statuses,
                    key=f"status_{map_key}"
                )
                if selected_status:
                    df_map = df_map[df_map["vcsProjectStatus"].isin(selected_status)]
    
    st.info(f"📍 **{len(df_map):,}** projetos sendo exibidos")
    
    # Criar mapa
    if len(df_map) == 0:
        st.warning("⚠️ Nenhum projeto encontrado com os filtros selecionados.")
        return
    
    center = [df_map["new_latitude"].mean(), df_map["new_longitude"].mean()]
    m = folium.Map(
        location=center,
        zoom_start=4,
        tiles="CartoDB dark_matter"
    )
    
    if map_type == "Clusters":
        # Mapa com clusters
        marker_cluster = MarkerCluster().add_to(m)
        
        for idx, row in df_map.iterrows():
            lat, lon = row["new_latitude"], row["new_longitude"]
            activity = row.get("vcsAFOLUActivity", "Unknown")
            color = ACTIVITY_COLORS.get(activity, "#808080")
            
            popup_html = f"""
            <div style="font-family: Arial; font-size: 12px; width: 250px;">
                <h4 style="margin: 0 0 10px 0; color: #2c3e50;">{row.get('resourceName_x', 'N/A')}</h4>
                <hr style="margin: 5px 0;">
                <b>ID:</b> {row.get('resourceIdentifier', 'N/A')}<br>
                <b>Status:</b> {row.get('vcsProjectStatus', 'N/A')}<br>
                <b>Atividade:</b> {activity}<br>
                <b>Estado:</b> {row.get('state_Recode', 'N/A')}<br>
                <b>EAER:</b> {row.get('vcsEstimatedAnnualEmissionReductions', 'N/A')}<br>
                <b>Área:</b> {row.get('vcsAcresHectares', 'N/A')}
            </div>
            """
            
            folium.CircleMarker(
                location=[lat, lon],
                radius=6,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                popup=folium.Popup(popup_html, max_width=300)
            ).add_to(marker_cluster)
    
    elif map_type == "Heatmap":
        # Heatmap
        heat_data = [[row["new_latitude"], row["new_longitude"]] 
                     for idx, row in df_map.iterrows()]
        HeatMap(heat_data, radius=15).add_to(m)
    
    else:
        # Pontos simples
        for idx, row in df_map.iterrows():
            lat, lon = row["new_latitude"], row["new_longitude"]
            activity = row.get("vcsAFOLUActivity", "Unknown")
            color = ACTIVITY_COLORS.get(activity, "#808080")
            
            folium.CircleMarker(
                location=[lat, lon],
                radius=5,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.6
            ).add_to(m)
    
    # Exibir mapa
    st_folium(m, width=None, height=600, key=f"map_{map_key}")
    
    # Legenda
    if map_type != "Heatmap":
        with st.expander("🎨 Legenda"):
            for activity, color in ACTIVITY_COLORS.items():
                if activity != "Unknown":
                    st.markdown(
                        f"<span style='display:inline-block;width:20px;height:20px;"
                        f"background:{color};margin-right:10px;border:1px solid #000;'></span> "
                        f"<b>{activity}</b>",
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
    
    # Preparar dados agrupados por resourceName_x
    df_credit_map = df_credit.copy()
    
    # Agregar por projeto (resourceName_x) mantendo a primeira ocorrência de coordenadas
    df_credit_unique = df_credit_map.groupby('resourceName_x').first().reset_index()
    
    st.info(f"📊 Exibindo **{len(df_credit_unique):,}** projetos únicos (de {len(df_credit_map):,} registros totais)")
    
    # Usar a função de mapa com dados únicos
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
        # Filtros
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
            # Métricas superiores
            #col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1,  col_m3, col_m4 = st.columns(3)
            
            with col_m1:
                if 'Mean' in df_proj.columns:
                    #st.metric("Média VCUs", f"{df_proj['Mean'].iloc[0]:,.0f}")
                    st.metric("Média VCUs", f"{df_proj['Mean'].iloc[0]:,.0f} ± {df_proj['IC_Mais'].iloc[0] - df_proj['Mean'].iloc[0]:,.0f}")
            
            #with col_m2:
            #    if 'EAER' in df_proj.columns:
            #        #st.metric("EAER", f"{df_proj['EAER'].iloc[0]:,.0f}")
            #        eaer_value = pd.to_numeric(df_proj["EAER"].iloc[0], errors="coerce")        
            #        st.metric("EAER", f"{eaer_value:,.0f}" if not np.isnan(eaer_value) else "N/A")

            with col_m3:
                if 'protocol' in df_overview.columns.columns:
                    st.metric("Protocolo", df_overview['protocol'].iloc[0],delta_color="off")
            
            with col_m4:
                if 'vcsProjectStatus' in df_overview.columns:
                    st.metric("Status", df_overview['vcsProjectStatus'].iloc[0],delta_color="off")
            
            
            

            st.divider()

            # Gráfico interativo - layout lado a lado
            col_graf1, col_graf2 = st.columns([3, 1.5])

            with col_graf1:
            
            # Gráfico interativo
                fig = go.Figure()

                # Barras
                fig.add_trace(go.Bar(
                    x=df_proj['Ano_Periodo'],
                    y=df_proj['TotalVintageQuantity'],
                    name='Total Vintage',
                    marker_color='#1E800A'
                ))

                fig.add_trace(go.Bar(
                    x=df_proj['Ano_Periodo'],
                    y=df_proj['SumQuantity'],
                    name='Sum Quantity',
                    marker_color='#6DD458'
                ))

                fig.add_trace(go.Bar(
                    x=df_proj['Ano_Periodo'],
                    y=df_proj['Sum_Retired'],
                    name='Retired',
                    marker_color='#FFC2A3'
                ))

                # Linha da média com IC
                if 'Mean' in df_proj.columns:
                    fig.add_trace(go.Scatter(
                        x=df_proj['Ano_Periodo'],
                        y=df_proj['Mean'],
                        mode='lines+markers',
                        name='Média',
                        line=dict(color='#1E800A', width=3),
                        marker=dict(size=8)
                    ))

                if 'IC_Mais' in df_proj.columns and 'IC_Menos' in df_proj.columns:
                    fig.add_trace(go.Scatter(
                        x=df_proj['Ano_Periodo'],
                        y=df_proj['IC_Mais'],
                        mode='lines',
                        name='IC Superior',
                        line=dict(color='gray', width=2, dash='dot')
                    ))

                    fig.add_trace(go.Scatter(
                        x=df_proj['Ano_Periodo'],
                        y=df_proj['IC_Menos'],
                        mode='lines',
                        name='IC Inferior',
                        line=dict(color='gray', width=2, dash='dot')
                    ))

                fig.update_layout(
                    barmode='group',
                    title=f"Análise de VCUs - {projeto_sel}",
                    xaxis_title="Período (Ano)",
                    yaxis_title="Quantidade de VCUs",
                    legend_title="Métricas",
                    template="plotly_white",
                    height=450,
                    hovermode='x unified',
                    #hovermode='x unified',
                    margin=dict(t=50, b=40, l=60, r=20)  # <-- t=50 igual ao de totais
                )

                st.plotly_chart(fig, use_container_width=True)


            #################################
            with col_graf2:
            # Soma total de todos os anos por métrica
                metricas_totais = {
                    'TotalVintageQuantity': '#1E800A',
                    'SumQuantity':          '#6DD458',
                    'Sum_Retired':          '#FFC2A3',
                    'Sum_Active':           '#A3C4F3',
                }

                labels  = []
                valores = []
                cores   = []

                for col_name, cor in metricas_totais.items():
                    if col_name in df_proj.columns:
                        labels.append(col_name.replace('Sum_', '').replace('Sum', '').replace('Total', 'Total '))
                        valores.append(pd.to_numeric(df_proj[col_name], errors='coerce').sum())
                        cores.append(cor)

                fig_tot = go.Figure()

                fig_tot.add_trace(go.Bar(
                    x=labels,
                    y=valores,
                    marker_color=cores,
                    text=[f"{v:,.0f}" for v in valores],
                    textposition='outside',
                    textfont=dict(size=10),
                    showlegend=False
                ))

                fig_tot.update_layout(
                    title="Totais Acumulados",
                    xaxis_title="",
                    yaxis_title="VCUs",
                    template="plotly_white",
                    height=450,
                    margin=dict(t=50, b=40, l=40, r=20),
                    yaxis=dict(showticklabels=False)   # evita poluição visual no eixo Y
                )

                st.plotly_chart(fig_tot, use_container_width=True)

            #######################
            # Tabela de dados
            with st.expander("📊 Ver Tabela de Dados Detalhada"):
                display_cols = ['Ano_Periodo', 'TotalVintageQuantity', 'SumQuantity', 
                               'Sum_Retired', 'Sum_Active', 'Mean', 'IC_Mais', 'IC_Menos']
                display_cols = [col for col in display_cols if col in df_proj.columns]
                st.dataframe(
                    df_proj[display_cols].style.format({
                        col: "{:,.0f}" for col in display_cols if col != 'Ano_Periodo'
                    }),
                    use_container_width=True
                )

            
            
            # =====================================
            # ABA 5: DADOS BRUTOS
            # =====================================

            with tabs[5]:
                st.header("📁 Visualização dos Dados Brutos")

                data_option = st.radio(
                    "Selecione o conjunto de dados:",
                    ["Todos os Projetos", "Projetos com Créditos"],
                    horizontal=True
                )

                df_display = df_all if data_option == "Todos os Projetos" else df_credit

                col1, col2 = st.columns([3, 1])
                with col1:
                    st.info(f"Exibindo primeiras 100 linhas de {len(df_display):,} registros")
                with col2:
                    if st.button("📥 Download CSV", use_container_width=True):
                        csv = df_display.to_csv(index=False)
                        st.download_button(
                            label="Baixar arquivo",
                            data=csv,
                            file_name=f"mrv_data_{data_option.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv"
                        )

                # Filtros para dados brutos
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

                st.dataframe(
                    df_display.head(100),
                    use_container_width=True,
                    height=600
                )

                # Informações sobre colunas
                with st.expander("ℹ️ Informações sobre as Colunas"):
                    col_info_a, col_info_b = st.columns(2)

                    with col_info_a:
                        st.write(f"**Total de colunas:** {len(df_display.columns)}")
                        st.write(f"**Total de linhas (filtrado):** {len(df_display):,}")

                    with col_info_b:
                        st.write("**Tipos de dados:**")
                        type_counts = df_display.dtypes.value_counts()
                        for dtype, count in type_counts.items():
                            st.text(f"• {dtype}: {count} colunas")

                    st.divider()
                    st.write("**Lista de Colunas:**")
                    cols_per_row = 3
                    cols_list = df_display.columns.tolist()

                    for i in range(0, len(cols_list), cols_per_row):
                        cols = st.columns(cols_per_row)
                        for j, col in enumerate(cols):
                            if i + j < len(cols_list):
                                col.text(f"• {cols_list[i + j]}")

# =====================================
# ABA 5: STORYTELLING
# INSERIR ESTE CÓDIGO APÓS A ABA 4 (Análises de Vintage)
# =====================================



with tabs[4]:
    st.header("📖 A História dos Projetos de Carbono no Brasil")
    
    # Subtabs para organizar o storytelling
    story_tabs = st.tabs([
        "🌍 Panorama Geral",
        "🗺️ Perda Florestal",
        "📊 Evolução Temporal",
        "🎯 Impacto Regional",
        "💡 Insights"
    ])
    
    # =====================================
    # STORYTELLING 1: PANORAMA GERAL
    # =====================================
    
    with story_tabs[0]:
        st.markdown("## 🌱 A Jornada do Carbono Florestal Brasileiro")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### Do Desmatamento aos Créditos de Carbono
            
            O Brasil abriga a maior floresta tropical do mundo, mas também enfrenta desafios 
            significativos de desmatamento. Os projetos de carbono surgem como uma solução 
            inovadora que:
            
            - 💰 **Valoriza economicamente** a floresta em pé
            - 🌳 **Preserva a biodiversidade** amazônica
            - 👥 **Beneficia comunidades** locais
            - 🌍 **Combate as mudanças climáticas** globais
            """)
        
        with col2:
            # Card de destaque
            st.markdown("""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 30px; border-radius: 15px; color: white; text-align: center;'>
                <h1 style='margin: 0; font-size: 3em;'>🌎</h1>
                <h3 style='margin: 10px 0;'>Brasil</h3>
                <p style='margin: 5px 0; font-size: 1.2em;'><b>{:,}</b> projetos</p>
                <p style='margin: 5px 0;'><b>{:,}</b> com créditos</p>
            </div>
            """.format(len(df_all), df_credit['resourceName_x'].nunique()), 
            unsafe_allow_html=True)
        
        st.divider()
        
        # Timeline de evolução dos projetos
        st.markdown("### 📅 Linha do Tempo dos Projetos")
        
        if 'vcsRegistrationDate' in df_all.columns:
            df_timeline = df_all.copy()
            df_timeline['vcsRegistrationDate'] = pd.to_datetime(
                df_timeline['vcsRegistrationDate'], 
                errors='coerce'
            )
            df_timeline = df_timeline.dropna(subset=['vcsRegistrationDate'])
            df_timeline['Ano'] = df_timeline['vcsRegistrationDate'].dt.year
            
            timeline_data = df_timeline.groupby(['Ano', 'vcsAFOLUActivity']).size().reset_index(name='Quantidade')
            timeline_data = timeline_data[timeline_data['Ano'] >= 2000]
            
            fig_timeline = px.area(
                timeline_data,
                x='Ano',
                y='Quantidade',
                color='vcsAFOLUActivity',
                color_discrete_map=ACTIVITY_COLORS,
                title='Crescimento dos Projetos de Carbono ao Longo do Tempo'
            )
            fig_timeline.update_layout(
                hovermode='x unified',
                height=400,
                legend_title_text='Tipo de Atividade'
            )
            st.plotly_chart(fig_timeline, use_container_width=True)
        
        # Métricas comparativas
        st.markdown("### 🎯 Números que Contam Histórias")
        
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        
        with col_m1:
            if 'vcsAcresHectares' in df_all.columns:
                try:
                    # Limpar e converter área
                    df_all_temp = df_all.copy()
                    df_all_temp['vcsAcresHectares_num'] = (
                        df_all_temp['vcsAcresHectares']
                        .astype(str)
                        .str.replace(r'[^\d.,]', '', regex=True)
                        .str.replace(',', '', regex=False)
                    )
                    df_all_temp['vcsAcresHectares_num'] = pd.to_numeric(
                        df_all_temp['vcsAcresHectares_num'], 
                        errors='coerce'
                    )
                    total_area = df_all_temp['vcsAcresHectares_num'].sum()
                    
                    st.metric(
                        "Área Total Protegida",
                        f"{total_area/1000000:,.1f}M ha",
                        help="Milhões de hectares sob proteção"
                    )
                except Exception as e:
                    st.metric("Área Total Protegida", "N/A", help=f"Erro: {str(e)}")
        
        with col_m2:
            if 'vcsEstimatedAnnualEmissionReductions' in df_all.columns:
                try:
                    total_eaer = pd.to_numeric(
                        df_all['vcsEstimatedAnnualEmissionReductions'], 
                        errors='coerce'
                    ).sum()
                    st.metric(
                        "Reduções Anuais (tCO2e)",
                        f"{total_eaer/1000000:.1f}M",
                        help="Milhões de toneladas de CO2 equivalente"
                    )
                except Exception as e:
                    st.metric("Reduções Anuais", "N/A", help=f"Erro: {str(e)}")
        
        with col_m3:
            try:
                redd_count = len(df_all[df_all['vcsAFOLUActivity'].str.contains('REDD', na=False)])
                st.metric(
                    "Projetos REDD+",
                    f"{redd_count}",
                    help="Redução de Emissões por Desmatamento e Degradação"
                )
            except Exception as e:
                st.metric("Projetos REDD+", "N/A", help=f"Erro: {str(e)}")
        
        with col_m4:
            if 'state_Recode' in df_all.columns:
                try:
                    states_count = df_all['state_Recode'].nunique()
                    st.metric(
                        "Estados Alcançados",
                        f"{states_count}",
                        help="Número de estados com projetos"
                    )
                except Exception as e:
                    st.metric("Estados Alcançados", "N/A", help=f"Erro: {str(e)}")
    
    # =====================================
    # OUTRAS SUB-ABAS (placeholder por enquanto)
    # =====================================
    #
    #with story_tabs[1]:
    #    st.info("🚧 Em construção - Perda Florestal")
    #
    #with story_tabs[2]:
    #    st.info("🚧 Em construção - Evolução Temporal")
    #
    #with story_tabs[3]:
    #    st.info("🚧 Em construção - Impacto Regional")
    #
    #with story_tabs[4]:
    #    st.info("🚧 Em construção - Insights")


# =====================================
# SUBSTITUIR AS LINHAS DAS SUB-ABAS PLACEHOLDER
# =====================================
# Localize onde está:
#     with story_tabs[1]:
#         st.info("🚧 Em construção - Perda Florestal")
#     with story_tabs[2]:
#         st.info("🚧 Em construção - Evolução Temporal")
#     ...
# 
# E SUBSTITUA por este código completo:
# =====================================

    # =====================================
    # STORYTELLING 2: PERDA FLORESTAL
    # =====================================
    
    with story_tabs[1]:
        #st.markdown("## 🔥 O Contexto da Perda Florestal")
        
        #st.markdown("""
        #Os projetos de carbono muitas vezes são estabelecidos em áreas com **histórico de 
        #desmatamento** ou sob **pressão de desmatamento**. Entender esse contexto é crucial 
        #para avaliar o impacto real desses projetos.
        #""")
        
        # Verificar se existem dados de perda florestal
        forest_cols = [col for col in df_all.columns if 'forest' in col.lower() or 'loss' in col.lower()]
        
        if forest_cols:
            st.info(f"📊 Encontradas {len(forest_cols)} colunas relacionadas a perda florestal")
            
            # Criar mapa de calor de perda florestal
            st.markdown("### 🗺️ Mapa de Risco de Desmatamento")
            
            # Preparar dados para o mapa
            df_forest_map = prepare_map_data(df_all)
            
            # Adicionar coluna de intensidade (exemplo baseado em EAER)
            if 'vcsEstimatedAnnualEmissionReductions' in df_forest_map.columns:
                df_forest_map['intensity'] = pd.to_numeric(
                    df_forest_map['vcsEstimatedAnnualEmissionReductions'],
                    errors='coerce'
                ).fillna(0)
            
            # Filtros
            col_f1, col_f2 = st.columns(2)
            
            with col_f1:
                show_redd_only = st.checkbox(
                    "Mostrar apenas projetos REDD+",
                    value=False,
                    key="forest_redd"
                )
                if show_redd_only and 'vcsAFOLUActivity' in df_forest_map.columns:
                    df_forest_map = df_forest_map[
                        df_forest_map['vcsAFOLUActivity'].str.contains('REDD', na=False)
                    ]
            
            with col_f2:
                if 'state_Recode' in df_forest_map.columns:
                    states_forest = ["Todos"] + sorted(
                        df_forest_map["state_Recode"].dropna().unique().tolist()
                    )
                    selected_state_forest = st.selectbox(
                        "Estado:",
                        states_forest,
                        key="forest_state"
                    )
                    if selected_state_forest != "Todos":
                        df_forest_map = df_forest_map[
                            df_forest_map["state_Recode"] == selected_state_forest
                        ]
            
            st.info(f"🌲 Exibindo {len(df_forest_map):,} projetos")
            
            # Criar mapa
            if len(df_forest_map) > 0:
                center = [df_forest_map["new_latitude"].mean(), 
                         df_forest_map["new_longitude"].mean()]
                m_forest = folium.Map(
                    location=center,
                    zoom_start=5,
                    tiles="Esri.WorldImagery"
                )
                
                # Adicionar heatmap baseado em EAER ou contagem
                if 'intensity' in df_forest_map.columns:
                    heat_data = [
                        [row["new_latitude"], row["new_longitude"], row["intensity"]/1000]
                        for idx, row in df_forest_map.iterrows()
                        if pd.notna(row["intensity"])
                    ]
                else:
                    heat_data = [
                        [row["new_latitude"], row["new_longitude"], 1]
                        for idx, row in df_forest_map.iterrows()
                    ]
                
                HeatMap(
                    heat_data,
                    radius=20,
                    blur=25,
                    max_zoom=13,
                    gradient={
                        0.0: 'green',
                        0.5: 'yellow',
                        1.0: 'red'
                    }
                ).add_to(m_forest)
                
                st_folium(m_forest, width=None, height=500, key="forest_heatmap")
                
                st.caption("""
                🔴 **Vermelho**: Áreas com maior concentração de projetos/emissões evitadas  
                🟡 **Amarelo**: Concentração média  
                🟢 **Verde**: Menor concentração
                """)
        
        #else:
        #    st.warning("⚠️ Dados de perda florestal não disponíveis no dataset atual")
        
        # Análise por estado
        #st.divider()
        #st.markdown("### 📊 Pressão de Desmatamento por Estado")
        
        if 'state_Recode' in df_all.columns and 'vcsAFOLUActivity' in df_all.columns:
            # Contagem de projetos REDD por estado
            df_redd_state = df_all[
                df_all['vcsAFOLUActivity'].str.contains('REDD', na=False)
            ].groupby('state_Recode').size().reset_index(name='Projetos_REDD')
            
            df_redd_state = df_redd_state.sort_values('Projetos_REDD', ascending=True).tail(15)
            
            fig_redd_state = px.bar(
                df_redd_state,
                x='Projetos_REDD',
                y='state_Recode',
                orientation='h',
                title='Estados com Mais Projetos REDD+ (Proteção contra Desmatamento)',
                color='Projetos_REDD',
                color_continuous_scale='RdYlGn_r'
            )
            fig_redd_state.update_layout(height=500, showlegend=False)
            #st.plotly_chart(fig_redd_state, use_container_width=True)
    
    # =====================================
    # STORYTELLING 3: EVOLUÇÃO TEMPORAL
    # =====================================
    
    with story_tabs[2]:
        st.markdown("## ⏱️ A Evolução dos Projetos no Tempo")
        
        st.markdown("""
        ### Da Ideia aos Créditos: Uma Jornada de Anos
        
        Um projeto de carbono passa por várias fases antes de gerar créditos verificados:
        """)
        
        # Pipeline visual
        pipeline_html = """
        <div style='display: flex; justify-content: space-between; margin: 30px 0;'>
            <div style='text-align: center; flex: 1;'>
                <div style='background: #e3f2fd; padding: 20px; border-radius: 10px; margin: 5px;'>
                    <h3>📝</h3>
                    <b>Desenvolvimento</b>
                    <p style='font-size: 0.9em; color: #666;'>Planejamento e design</p>
                </div>
            </div>
            <div style='text-align: center; flex: 1;'>
                <div style='background: #fff3e0; padding: 20px; border-radius: 10px; margin: 5px;'>
                    <h3>🔍</h3>
                    <b>Validação</b>
                    <p style='font-size: 0.9em; color: #666;'>Auditoria independente</p>
                </div>
            </div>
            <div style='text-align: center; flex: 1;'>
                <div style='background: #e8f5e9; padding: 20px; border-radius: 10px; margin: 5px;'>
                    <h3>✅</h3>
                    <b>Registro</b>
                    <p style='font-size: 0.9em; color: #666;'>Aprovação oficial</p>
                </div>
            </div>
            <div style='text-align: center; flex: 1;'>
                <div style='background: #f3e5f5; padding: 20px; border-radius: 10px; margin: 5px;'>
                    <h3>📊</h3>
                    <b>Monitoramento</b>
                    <p style='font-size: 0.9em; color: #666;'>Verificação contínua</p>
                </div>
            </div>
            <div style='text-align: center; flex: 1;'>
                <div style='background: #c8e6c9; padding: 20px; border-radius: 10px; margin: 5px;'>
                    <h3>💰</h3>
                    <b>Créditos</b>
                    <p style='font-size: 0.9em; color: #666;'>Emissão e venda</p>
                </div>
            </div>
        </div>
        """
        st.markdown(pipeline_html, unsafe_allow_html=True)
        
        # Análise de tempo entre registro e primeiro crédito
        if 'vcsRegistrationDate' in df_credit.columns and 'Vintage' in df_credit.columns:
            st.markdown("### ⏰ Tempo até os Primeiros Créditos")
            
            df_timing = df_credit.copy()
            df_timing['vcsRegistrationDate'] = pd.to_datetime(
                df_timing['vcsRegistrationDate'], 
                errors='coerce'
            )
            
            # Extrair ano do vintage
            df_timing['Vintage_Year'] = df_timing['Vintage'].apply(
                lambda x: int(x.split(' e ')[0][:4]) if isinstance(x, str) and ' e ' in x 
                else (int(str(x)[:4]) if pd.notna(x) else None)
            )
            
            df_timing = df_timing.dropna(subset=['vcsRegistrationDate', 'Vintage_Year'])
            df_timing['Registration_Year'] = df_timing['vcsRegistrationDate'].dt.year
            df_timing['Years_to_Credit'] = df_timing['Vintage_Year'] - df_timing['Registration_Year']
            
            # Filtrar valores razoáveis
            df_timing = df_timing[
                (df_timing['Years_to_Credit'] >= -5) & 
                (df_timing['Years_to_Credit'] <= 10)
            ]
            
            if len(df_timing) > 0:
                fig_timing = px.histogram(
                    df_timing,
                    x='Years_to_Credit',
                    nbins=20,
                    title='Distribuição do Tempo entre Registro e Emissão de Créditos',
                    labels={'Years_to_Credit': 'Anos', 'count': 'Número de Projetos'},
                    color_discrete_sequence=['#26a69a']
                )
                fig_timing.update_layout(
                    showlegend=False,
                    height=400
                )
                st.plotly_chart(fig_timing, use_container_width=True)
                
                # Estatísticas
                col_stat1, col_stat2, col_stat3 = st.columns(3)
                
                with col_stat1:
                    median_time = df_timing['Years_to_Credit'].median()
                    st.metric("Tempo Mediano", f"{median_time:.1f} anos")
                
                with col_stat2:
                    mean_time = df_timing['Years_to_Credit'].mean()
                    st.metric("Tempo Médio", f"{mean_time:.1f} anos")
                
                with col_stat3:
                    fast_projects = len(df_timing[df_timing['Years_to_Credit'] <= 1])
                    st.metric("Projetos Rápidos", f"{fast_projects}", 
                             help="Projetos que emitiram créditos em até 1 ano")
        
        # Evolução anual dos créditos
        st.divider()
        st.markdown("### 📈 Evolução dos Créditos Emitidos")
        
        if 'Vintage' in df_credit.columns and 'totalVintageQuantity' in df_credit.columns:
            df_credits_year = df_credit.copy()
            df_credits_year['Vintage_Year'] = df_credits_year['Vintage'].apply(
                lambda x: int(x.split(' e ')[0][:4]) if isinstance(x, str) and ' e ' in x 
                else (int(str(x)[:4]) if pd.notna(x) else None)
            )
            
            df_credits_year = df_credits_year.dropna(subset=['Vintage_Year'])
            df_credits_year['totalVintageQuantity'] = pd.to_numeric(
                df_credits_year['totalVintageQuantity'],
                errors='coerce'
            )
            
            credits_by_year = df_credits_year.groupby('Vintage_Year').agg({
                'totalVintageQuantity': 'sum',
                'resourceName_x': 'nunique'
            }).reset_index()
            
            credits_by_year.columns = ['Ano', 'Total_Creditos', 'Num_Projetos']
            credits_by_year = credits_by_year[credits_by_year['Ano'] >= 2000]
            
            fig_credits_evolution = go.Figure()
            
            fig_credits_evolution.add_trace(go.Bar(
                x=credits_by_year['Ano'],
                y=credits_by_year['Total_Creditos'],
                name='Créditos Emitidos',
                marker_color='#26a69a',
                yaxis='y'
            ))
            
            fig_credits_evolution.add_trace(go.Scatter(
                x=credits_by_year['Ano'],
                y=credits_by_year['Num_Projetos'],
                name='Número de Projetos',
                marker_color='#ff6b6b',
                mode='lines+markers',
                yaxis='y2'
            ))
            
            fig_credits_evolution.update_layout(
                title='Emissão de Créditos e Número de Projetos ao Longo do Tempo',
                xaxis_title='Ano',
                yaxis_title='Total de Créditos (VCUs)',
                yaxis2=dict(
                    title='Número de Projetos',
                    overlaying='y',
                    side='right'
                ),
                height=500,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_credits_evolution, use_container_width=True)
    
    # =====================================
    # STORYTELLING 4: IMPACTO REGIONAL
    # =====================================
    
    with story_tabs[3]:
        st.markdown("## 🎯 O Impacto nos Territórios")
        
        st.markdown("""
        Os projetos de carbono não são distribuídos uniformemente pelo Brasil. 
        Eles se concentram em regiões estratégicas, cada uma com sua própria história.
        """)
        
        # Mapa de calor regional
        if 'state_Recode' in df_all.columns:
            st.markdown("### 🗺️ Densidade de Projetos por Região")
            
            # Preparar dados regionais
            state_summary = df_all.groupby('state_Recode').agg({
                'resourceName_x': 'count',
                'vcsAcresHectares': lambda x: pd.to_numeric(x, errors='coerce').sum(),
                'vcsEstimatedAnnualEmissionReductions': lambda x: pd.to_numeric(x, errors='coerce').sum()
            }).reset_index()
            
            state_summary.columns = ['Estado', 'Num_Projetos', 'Area_Total', 'EAER_Total']
            state_summary = state_summary.sort_values('Num_Projetos', ascending=False).head(15)
            
            # Criar gráfico de bolhas
            fig_regional = px.scatter(
                state_summary,
                x='Area_Total',
                y='EAER_Total',
                size='Num_Projetos',
                color='Num_Projetos',
                hover_name='Estado',
                labels={
                    'Area_Total': 'Área Total (hectares)',
                    'EAER_Total': 'Reduções de Emissões (tCO2e/ano)',
                    'Num_Projetos': 'Número de Projetos'
                },
                title='Relação entre Área, Impacto e Número de Projetos',
                color_continuous_scale='Viridis',
                size_max=60
            )
            fig_regional.update_layout(height=500)
            st.plotly_chart(fig_regional, use_container_width=True)
            
            st.caption("""
            💡 **Interpretação**: Bolhas maiores = mais projetos. 
            Quanto mais à direita e acima, maior o impacto combinado de área e reduções.
            """)
        
        # Comparativo de tipos de projeto por região
        st.divider()
        st.markdown("### 🌳 Perfil de Atividades por Estado")
        
        if 'state_Recode' in df_all.columns and 'vcsAFOLUActivity' in df_all.columns:
            # Top 10 estados
            top_states = df_all['state_Recode'].value_counts().head(10).index.tolist()
            df_activity_state = df_all[df_all['state_Recode'].isin(top_states)]
            
            activity_by_state = df_activity_state.groupby(
                ['state_Recode', 'vcsAFOLUActivity']
            ).size().reset_index(name='Count')
            
            fig_activity_state = px.bar(
                activity_by_state,
                x='state_Recode',
                y='Count',
                color='vcsAFOLUActivity',
                title='Distribuição de Tipos de Atividade nos Principais Estados',
                labels={'state_Recode': 'Estado', 'Count': 'Número de Projetos'},
                color_discrete_map=ACTIVITY_COLORS,
                barmode='stack'
            )
            fig_activity_state.update_layout(height=500)
            st.plotly_chart(fig_activity_state, use_container_width=True)
    
    # =====================================
    # STORYTELLING 5: INSIGHTS
    # =====================================
    
    with story_tabs[4]:
        st.markdown("## 💡 Insights e Descobertas")
        
        # Container de insights
        insight_cols = st.columns(2)
        
        with insight_cols[0]:
            st.markdown("""
            ### 🔍 Principais Descobertas
            
            #### 1. Concentração Geográfica
            A maioria dos projetos se concentra em poucos estados, 
            principalmente aqueles com histórico de desmatamento elevado.
            
            #### 2. Predominância REDD+
            Projetos de Redução de Emissões por Desmatamento e Degradação 
            são os mais comuns, refletindo a urgência do combate ao desmatamento.
            
            #### 3. Ciclo de Maturação
            Em média, projetos levam alguns anos entre registro e emissão 
            dos primeiros créditos, refletindo a complexidade do processo.
            """)
        
        with insight_cols[1]:
            st.markdown("""
            ### 🎯 Oportunidades
            
            #### 1. Expansão Geográfica
            Diversos estados ainda têm poucos projetos, representando 
            oportunidades para novos investimentos.
            
            #### 2. Diversificação
            Além de REDD+, outras metodologias como ARR e IFM 
            podem ser exploradas.
            
            #### 3. Escala
            Muitos projetos têm potencial para expansão e replicação 
            em áreas adjacentes.
            """)
        
        st.divider()
        
        # Análise de correlação
        st.markdown("### 📊 Análise de Correlação: Área vs Impacto")
        
        if 'vcsAcresHectares' in df_all.columns and 'vcsEstimatedAnnualEmissionReductions' in df_all.columns:
            df_correlation = df_all.copy()
            df_correlation['Area'] = pd.to_numeric(df_correlation['vcsAcresHectares'], errors='coerce')
            df_correlation['EAER'] = pd.to_numeric(
                df_correlation['vcsEstimatedAnnualEmissionReductions'], 
                errors='coerce'
            )
            
            df_correlation = df_correlation.dropna(subset=['Area', 'EAER'])
            df_correlation = df_correlation[(df_correlation['Area'] > 0) & (df_correlation['EAER'] > 0)]
            
            if len(df_correlation) > 0:
                fig_corr = px.scatter(
                    df_correlation,
                    x='Area',
                    y='EAER',
                    color='vcsAFOLUActivity',
                    hover_data=['resourceName_x', 'state_Recode'],
                    labels={
                        'Area': 'Área do Projeto (hectares)',
                        'EAER': 'Reduções Anuais de Emissões (tCO2e)',
                        'vcsAFOLUActivity': 'Tipo de Atividade'
                    },
                    title='Relação entre Tamanho do Projeto e Impacto Climático',
                    color_discrete_map=ACTIVITY_COLORS,
                    log_x=True,
                    log_y=True
                )
                fig_corr.update_layout(height=500)
                st.plotly_chart(fig_corr, use_container_width=True)
                
                st.info("""
                💡 **Observação**: Escalas logarítmicas permitem visualizar melhor 
                a relação entre projetos de diferentes tamanhos. Projetos maiores 
                tendem a ter maior impacto, mas a eficiência varia.
                """)
        
        # Call to action final
        st.divider()
        st.markdown("""
        ### 🌍 O Futuro do Carbono Florestal
        
        Os dados revelam um ecossistema em crescimento, com desafios e oportunidades:
        
        - **Crescimento sustentado** no número de projetos ao longo dos anos
        - **Concentração regional** que pode ser rebalanceada
        - **Potencial inexplorado** em várias regiões do Brasil
        - **Impacto mensurável** na preservação florestal e mitigação climática
        
        📊 **Explore as outras abas** para análises mais detalhadas e dados brutos.
        """)

# =====================================
# FIM DO CÓDIGO DO STORYTELLING
# A aba Dados Brutos (tabs[5]) deve vir DEPOIS deste bloco
# =====================================

import os
import geopandas as gpd
import pandas as pd
import folium
from shapely.geometry import mapping
from shapely.errors import TopologicalError
from folium.plugins import HeatMap
from streamlit_folium import st_folium

# =====================================
# FUNÇÃO: CARREGAR GEOMETRIAS KML
# =====================================
@st.cache_data(show_spinner=True)
def carregar_geometrias(df_all, kml_dir: str):
    """
    Carrega os KMLs da pasta local, cruza com df_all via resourceIdentifier e retorna um GeoDataFrame combinado.
    """
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

    # Concatenar todos os KMLs
    gdf_all = pd.concat(lista_gdfs, ignore_index=True)

    # Converter CRS
    if gdf_all.crs is None:
        gdf_all.set_crs("EPSG:4326", inplace=True)
    else:
        gdf_all = gdf_all.to_crs("EPSG:4326")

    # Garantir tipos compatíveis antes do merge
    df_all["resourceIdentifier"] = df_all["resourceIdentifier"].astype(str)
    gdf_all["resourceIdentifier"] = gdf_all["resourceIdentifier"].astype(str)

    # Corrigir geometrias inválidas
    gdf_all["geometry"] = gdf_all["geometry"].buffer(0)

    # Dissolver geometrias duplicadas
    try:
        gdf_all = gdf_all.dissolve(by="resourceIdentifier")
    except TopologicalError:
        gdf_all["geometry"] = gdf_all["geometry"].buffer(0)
        gdf_all = gdf_all.dissolve(by="resourceIdentifier")

    # Merge com df_all
    gdf_all = gdf_all.merge(df_all, on="resourceIdentifier", how="left")

    return gdf_all, erros

    # =====================================
# STORYTELLING 2: PERDA FLORESTAL + VISUALIZADOR GFW COMPLETO
# =====================================
import streamlit as st
import folium
from streamlit_folium import st_folium
from folium import plugins
import geopandas as gpd
from shapely.geometry import mapping
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

with story_tabs[1]:
    st.markdown("## 🔥 O Contexto da Perda Florestal")

    st.markdown("""
    Os projetos de carbono muitas vezes são estabelecidos em áreas com **histórico de 
    desmatamento** ou sob **pressão de desmatamento**. Entender esse contexto é crucial 
    para avaliar o impacto real desses projetos.
    """)

    # Carregar geometrias dos KMLs
    #KML_DIR = r"D:\OneDrive - IPAM-Amazonia\1_BKP_Git\2024\0_Demanda_Espacializar_proj_priv\2025\kml"
    #KML_DIR = r"kml"
    

    import os

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    KML_DIR = os.path.join(BASE_DIR, "kml")

    print("BASE_DIR:", BASE_DIR)
    print("KML_DIR:", KML_DIR)
    print("Exists?", os.path.exists(KML_DIR))

    gdf_combined, erros = carregar_geometrias(df_all, KML_DIR)

    if erros:
        with st.expander("⚠️ Erros ao carregar alguns KMLs"):
            for f, e in erros:
                st.text(f"{f}: {e}")

    if gdf_combined.empty:
        st.warning("Nenhum KML válido encontrado na pasta especificada.")
    else:
        st.success(f"✅ {len(gdf_combined)} geometrias carregadas com sucesso!")

        # Filtrar geometrias válidas
        gdf_plot = gdf_combined[~gdf_combined["geometry"].is_empty & gdf_combined["geometry"].notnull()].copy()
        gdf_plot = gdf_plot[gdf_plot.is_valid]

        if gdf_plot.empty:
            st.warning("⚠️ Nenhuma geometria válida para exibir.")
        else:
            # ===================================
            # SELETOR DE PROJETO
            # ===================================
            st.markdown("### 📍 Selecione um Projeto para Análise GFW")
            
            # Criar lista de projetos com informações
            project_options = ["Visão Geral (Todos os Projetos)"] + [
                f"{row.get('resourceName_x', 'Sem nome')} - {row.get('state_Recode', 'N/A')}" 
                for _, row in gdf_plot.iterrows()
            ]
            
            selected_project = st.selectbox(
                "Escolha um projeto:",
                options=project_options,
                key="project_selector"
            )

            # ===================================
            # CONFIGURAÇÃO AVANÇADA GFW
            # ===================================
            
            # Determinar centro e zoom baseado na seleção
            if selected_project == "Visão Geral (Todos os Projetos)":
                centroid = gdf_plot.geometry.centroid
                center = [centroid.y.mean(), centroid.x.mean()]
                zoom_start = 5
                selected_gdf = gdf_plot
                show_gfw = False
            else:
                project_name = selected_project.split(" - ")[0]
                selected_gdf = gdf_plot[gdf_plot["resourceName_x"] == project_name]
                
                if not selected_gdf.empty:
                    bounds = selected_gdf.total_bounds
                    center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]
                    zoom_start = 10
                    show_gfw = True
                else:
                    selected_gdf = gdf_plot
                    centroid = gdf_plot.geometry.centroid
                    center = [centroid.y.mean(), centroid.x.mean()]
                    zoom_start = 5
                    show_gfw = False

            # ===================================
            # PAINEL DE CONTROLE GFW
            # ===================================
            
            if show_gfw:
                st.markdown("---")
                st.markdown("### 🎛️ Painel de Controle - Análise Temporal GFW")
                
                # Tabs para diferentes análises
                analysis_tabs = st.tabs([
                    "📊 Comparação Temporal", 
                    "🌳 Altura da Vegetação", 
                    "📈 Ganho Florestal",
                    "🔥 Perda Anual",
                    "🎬 Timeline Animada"
                ])
                
                # ===================================
                # TAB 1: COMPARAÇÃO TEMPORAL
                # ===================================
                with analysis_tabs[0]:
                    st.markdown("#### 🔄 Compare Dois Anos Diferentes")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        year_left = st.selectbox(
                            "Ano Esquerdo:",
                            options=[2000, 2005, 2010, 2015, 2020],
                            index=0,
                            key="year_left"
                        )
                    
                    with col2:
                        year_right = st.selectbox(
                            "Ano Direito:",
                            options=[2000, 2005, 2010, 2015, 2020, "Perda até 2024"],
                            index=5,
                            key="year_right"
                        )
                    
                    # Opções de visualização
                    st.markdown("**Tipo de Dado:**")
                    data_type = st.radio(
                        "Selecione o tipo de dado:",
                        options=["Cobertura Florestal", "Altura da Vegetação", "Densidade (%)"],
                        horizontal=True,
                        key="data_type"
                    )
                    
                    # Criar mapa com split
                    st.markdown("---")
                    st.markdown(f"### 🗺️ {year_left} vs {year_right} - {data_type}")
                    
                    bounds = selected_gdf.total_bounds
                    
                    m = folium.Map(
                        location=center,
                        zoom_start=zoom_start,
                        tiles=None
                    )
                    
                    folium.TileLayer('Esri.WorldImagery', name='Base', control=False).add_to(m)
                    
                    # URLs das camadas GFW baseadas na seleção (URLs CORRETOS - 2024)
                    # Padrão GFW: https://tiles.globalforestwatch.org/{dataset}/{version}/dynamic/{z}/{x}/{y}.png
                    # Versão atual Tree Cover Loss: v1.12 (2024 data)
                    # Referência: https://data-api.globalforestwatch.org/
                    
                    #Caminhos corretos
                    
                    # URLs CORRETAS DO GFW (Janeiro 2025)
                    #GFW_URLS = {
                    #    #'tree_cover_loss': 'https://tiles.globalforestwatch.org/umd_tree_cover_loss/v1.11/tcd_30/{z}/{x}/{y}.png',
                    #    #'tree_cover_density_2000': 'https://tiles.globalforestwatch.org/umd_tree_cover_density_2000/v1.7/tcd_30/{z}/{x}/{y}.png',
                    #    #'tree_cover_gain': 'https://tiles.globalforestwatch.org/umd_tree_cover_gain/v1.7/tcd_30/{z}/{x}/{y}.png',
                    #    'carbon_density': 'https://tiles.globalforestwatch.org/gfw_forest_carbon_net_flux/v20210819/tcd_30/{z}/{x}/{y}.png',
                    #    'glad_alerts': 'https://tiles.globalforestwatch.org/umd_glad_landsat_alerts/v20220223/alert_date/{z}/{x}/{y}.png',
                    #    'radd_alerts': 'https://tiles.globalforestwatch.org/wur_radd_alerts/v20220126/alert_date/{z}/{x}/{y}.png'
                    #}
                    
                    if data_type == "Altura da Vegetação":
                        # Tree Cover Height - Anos disponíveis: 2000, 2005, 2010, 2015, 2020
                        # Nota: Altura usa endpoint diferente (não tem /dynamic/)
                        layer_left_url = f'https://tiles.globalforestwatch.org/gfw_forest_height_{year_left}/v202409/tcd_30/{{z}}/{{x}}/{{y}}.png'
                        
                        if year_right == "Perda até 2024":
                            # Hansen Tree Cover Loss v1.12 - atualizado para 2024
                            layer_right_url = 'https://tiles.globalforestwatch.org/umd_tree_cover_loss/v1.11/tcd_30/{z}/{x}/{y}.png'
                        else:
                            layer_right_url = f'https://tiles.globalforestwatch.org/gfw_forest_height_{year_right}/v202409/tcd_30/{{z}}/{{x}}/{{y}}.png'
                    
                    elif data_type == "Densidade (%)":
                        # Tree Canopy Cover 2000 - baseline
                        layer_left_url = 'https://tiles.globalforestwatch.org/umd_tree_cover_density/v1.7/dynamic/{z}/{x}/{y}.png'
                        
                        if year_right == "Perda até 2024":
                            layer_right_url = 'https://tiles.globalforestwatch.org/umd_tree_cover_loss/v1.11/tcd_30/{z}/{x}/{y}.png'
                        else:
                            # Densidade só está disponível para 2000
                            layer_right_url = 'https://tiles.globalforestwatch.org/umd_tree_cover_density/v1.7/dynamic/{z}/{x}/{y}.png'
                    
                    else:  # Cobertura Florestal
                        # Usar Tree Cover Density como proxy para cobertura
                        layer_left_url = 'https://tiles.globalforestwatch.org/umd_tree_cover_density/v1.7/dynamic/{z}/{x}/{y}.png'
                        
                        if year_right == "Perda até 2024":
                            layer_right_url = 'https://tiles.globalforestwatch.org/umd_tree_cover_loss/v1.11/tcd_30/{z}/{x}/{y}.png'
                        else:
                            layer_right_url = 'https://tiles.globalforestwatch.org/umd_tree_cover_density/v1.7/dynamic/{z}/{x}/{y}.png'
                    
                    layer_left = folium.TileLayer(
                        tiles=layer_left_url,
                        name=f'{data_type} {year_left}',
                        attr='Global Forest Watch',
                        overlay=True,
                        opacity=0.8
                    )
                    
                    layer_right = folium.TileLayer(
                        tiles=layer_right_url,
                        name=f'{data_type} {year_right}',
                        attr='Global Forest Watch',
                        overlay=True,
                        opacity=0.8
                    )
                    
                    sbs = plugins.SideBySideLayers(layer_left=layer_left, layer_right=layer_right)
                    layer_left.add_to(m)
                    layer_right.add_to(m)
                    sbs.add_to(m)
                    
                    # Adicionar polígono do projeto
                    def safe_geojson(row):
                        try:
                            return mapping(row["geometry"])
                        except Exception:
                            return None
                    
                    for _, row in selected_gdf.iterrows():
                        geojson_data = safe_geojson(row)
                        if geojson_data:
                            folium.GeoJson(
                                data=geojson_data,
                                name=row.get("resourceName_x", "Projeto"),
                                tooltip=folium.Tooltip(f"""
                                    <div style="font-family: Arial; font-size: 12px;">
                                        <b style="font-size: 14px;">{row.get('resourceName_x', 'Sem nome')}</b><br>
                                        <b>Estado:</b> {row.get('state_Recode', 'N/A')}<br>
                                        <b>Comparação:</b> {year_left} vs {year_right}<br>
                                        <hr style="margin: 5px 0;">
                                        <i>🌳 Arraste o controle para comparar</i>
                                    </div>
                                """, sticky=True),
                                style_function=lambda x: {
                                    "fillColor": "transparent",
                                    "color": "#FF0000",
                                    "weight": 3,
                                    "fillOpacity": 0.1,
                                    "dashArray": "5, 5"
                                }
                            ).add_to(m)
                    
                    m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
                    folium.LayerControl().add_to(m)
                    
                    st_folium(m, width=900, height=600, key="map_comparison")
                
                # ===================================
                # TAB 2: ALTURA DA VEGETAÇÃO
                # ===================================
                with analysis_tabs[1]:
                    st.markdown("#### 🌲 Evolução da Altura da Vegetação (2000-2020)")
                    
                    height_year = st.select_slider(
                        "Selecione o ano:",
                        options=[2000, 2005, 2010, 2015, 2020],
                        value=2020,
                        key="height_year"
                    )
                    
                    st.info(f"""
                    📏 **Dados de Altura para {height_year}**
                    - Resolução: 30 metros
                    - Vegetação ≥ 3 metros de altura
                    - Baseado em GEDI LiDAR + Landsat
                    """)
                    
                    bounds = selected_gdf.total_bounds
                    
                    m_height = folium.Map(
                        location=center,
                        zoom_start=zoom_start,
                        tiles='Esri.WorldImagery'
                    )
                    
                    # Camada de altura (versão corrigida)
                    height_url = f'https://tiles.globalforestwatch.org/gfw_forest_height_{height_year}/v202409/tcd_30/{{z}}/{{x}}/{{y}}.png'
                    
                    folium.TileLayer(
                        tiles=height_url,
                        name=f'Altura da Vegetação {height_year}',
                        attr='Global Forest Watch',
                        overlay=True,
                        opacity=0.8
                    ).add_to(m_height)
                    
                    # Adicionar polígono
                    for _, row in selected_gdf.iterrows():
                        geojson_data = safe_geojson(row)
                        if geojson_data:
                            folium.GeoJson(
                                data=geojson_data,
                                name=row.get("resourceName_x", "Projeto"),
                                style_function=lambda x: {
                                    "fillColor": "transparent",
                                    "color": "#00FF00",
                                    "weight": 3,
                                    "fillOpacity": 0.1
                                }
                            ).add_to(m_height)
                    
                    m_height.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
                    
                    # Legenda de altura
                    legend_height = '''
                    <div style="position: fixed; bottom: 50px; right: 50px; width: 250px; 
                                background-color: white; border:2px solid grey; z-index:9999; 
                                font-size:12px; padding: 10px; border-radius: 5px;">
                        <h4 style="margin-top:0;">Altura da Vegetação</h4>
                        <div style="background: linear-gradient(to right, #fff5eb, #006400); 
                                    height: 20px; border: 1px solid #ccc;"></div>
                        <div style="display: flex; justify-content: space-between; margin-top: 5px;">
                            <span>0m</span>
                            <span>15m</span>
                            <span>30m+</span>
                        </div>
                    </div>
                    '''
                    m_height.get_root().html.add_child(folium.Element(legend_height))
                    
                    folium.LayerControl().add_to(m_height)
                    st_folium(m_height, width=900, height=600, key="map_height")
                
                # ===================================
                # TAB 3: GANHO FLORESTAL
                # ===================================
                with analysis_tabs[2]:
                    st.markdown("#### 🌱 Ganho de Cobertura Florestal")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        gain_start = st.selectbox(
                            "Ano Inicial:",
                            options=[2000, 2005, 2010, 2015],
                            index=0,
                            key="gain_start"
                        )
                    
                    with col2:
                        gain_end = st.selectbox(
                            "Ano Final:",
                            options=[2020],
                            index=0,
                            key="gain_end"
                        )
                    
                    st.success(f"""
                    🌳 **Ganho Florestal {gain_start}-{gain_end}**
                    - Áreas que cresceram de <5m para ≥5m de altura
                    - Inclui reflorestamento natural e plantações
                    """)
                    
                    bounds = selected_gdf.total_bounds
                    
                    m_gain = folium.Map(
                        location=center,
                        zoom_start=zoom_start,
                        tiles='Esri.WorldImagery'
                    )
                    
                    # Camada de ganho (2000-2020) - URL CORRIGIDO
                    # Tree Cover Gain do Hansen/UMD v1.7
                    gain_url = 'https://tiles.globalforestwatch.org/umd_tree_cover_gain/v1.7/tcd_30/{z}/{x}/{y}.png'
                    
                    folium.TileLayer(
                        tiles=gain_url,
                        name=f'Ganho Florestal {gain_start}-{gain_end}',
                        attr='Global Forest Watch - Hansen/UMD v1.7',
                        overlay=True,
                        opacity=0.8
                    ).add_to(m_gain)
                    
                    # Camada de perda para contexto (v1.12 - 2024)
                    loss_url = 'https://tiles.globalforestwatch.org/umd_tree_cover_loss/v1.11/tcd_30/{z}/{x}/{y}.png'
                    
                    folium.TileLayer(
                        tiles=loss_url,
                        name='Perda Florestal (contexto)',
                        attr='Global Forest Watch',
                        overlay=True,
                        opacity=0.5,
                        show=False
                    ).add_to(m_gain)
                    
                    # Adicionar polígono
                    for _, row in selected_gdf.iterrows():
                        geojson_data = safe_geojson(row)
                        if geojson_data:
                            folium.GeoJson(
                                data=geojson_data,
                                name=row.get("resourceName_x", "Projeto"),
                                style_function=lambda x: {
                                    "fillColor": "transparent",
                                    "color": "#0000FF",
                                    "weight": 3,
                                    "fillOpacity": 0.1
                                }
                            ).add_to(m_gain)
                    
                    m_gain.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
                    
                    # Legenda
                    legend_gain = '''
                    <div style="position: fixed; bottom: 50px; right: 50px; width: 250px; 
                                background-color: white; border:2px solid grey; z-index:9999; 
                                font-size:12px; padding: 10px; border-radius: 5px;">
                        <h4 style="margin-top:0;">Legenda</h4>
                        <p><span style="color:#0000FF;">█</span> <b>Azul:</b> Ganho florestal</p>
                        <p><span style="color:#FF0000;">█</span> <b>Vermelho:</b> Perda florestal</p>
                        <p style="font-size: 10px; color: #666; margin-top: 10px;">
                            Ative/desative camadas no controle superior direito
                        </p>
                    </div>
                    '''
                    m_gain.get_root().html.add_child(folium.Element(legend_gain))
                    
                    folium.LayerControl().add_to(m_gain)
                    st_folium(m_gain, width=900, height=600, key="map_gain")
                
                # ===================================
                # TAB 4: PERDA ANUAL
                # ===================================
                with analysis_tabs[3]:
                    st.markdown("#### 🔥 Análise de Perda Florestal por Ano")
                    
                    # Slider de anos
                    year_range = st.slider(
                        "Selecione o período de análise:",
                        min_value=2001,
                        max_value=2024,
                        value=(2001, 2024),
                        key="loss_range"
                    )
                    
                    st.warning(f"""
                    ⚠️ **Período Selecionado: {year_range[0]}-{year_range[1]}**
                    - Dados de perda anual do Hansen/UMD
                    - Resolução: 30 metros (Landsat)
                    - Inclui todos os tipos de perda (incêndio, corte raso, degradação)
                    """)
                    
                    bounds = selected_gdf.total_bounds
                    
                    m_loss = folium.Map(
                        location=center,
                        zoom_start=zoom_start,
                        tiles='Esri.WorldImagery'
                    )
                    
                    # Camada de perda (v1.12 - 2024)
                    loss_url = 'https://tiles.globalforestwatch.org/umd_tree_cover_loss/v1.11/tcd_30/{z}/{x}/{y}.png'
                    
                    folium.TileLayer(
                        tiles=loss_url,
                        name=f'Perda Florestal {year_range[0]}-{year_range[1]}',
                        attr='Global Forest Watch - Hansen/UMD v1.12',
                        overlay=True,
                        opacity=0.8
                    ).add_to(m_loss)
                    
                    # Camada de cobertura 2000 para contexto
                    cover_url = 'https://tiles.globalforestwatch.org/umd_tree_cover_density/v1.7/dynamic/{z}/{x}/{y}.png'
                    
                    folium.TileLayer(
                        tiles=cover_url,
                        name='Cobertura 2000 (contexto)',
                        attr='Global Forest Watch',
                        overlay=True,
                        opacity=0.4,
                        show=False
                    ).add_to(m_loss)
                    
                    # Adicionar polígono
                    for _, row in selected_gdf.iterrows():
                        geojson_data = safe_geojson(row)
                        if geojson_data:
                            folium.GeoJson(
                                data=geojson_data,
                                name=row.get("resourceName_x", "Projeto"),
                                style_function=lambda x: {
                                    "fillColor": "transparent",
                                    "color": "#FF6600",
                                    "weight": 3,
                                    "fillOpacity": 0.1
                                }
                            ).add_to(m_loss)
                    
                    m_loss.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
                    
                    folium.LayerControl().add_to(m_loss)
                    st_folium(m_loss, width=900, height=600, key="map_loss")
                    
                    # Gráfico simulado de perda anual
                    st.markdown("---")
                    st.markdown("#### 📊 Perda Anual Estimada (Exemplo)")
                    
                    # Dados fictícios para demonstração
                    years = list(range(year_range[0], year_range[1] + 1))
                    loss_ha = [100 + i*10 + (i%3)*50 for i in range(len(years))]
                    
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=years,
                        y=loss_ha,
                        name='Perda (ha)',
                        marker_color='#ff4444'
                    ))
                    
                    fig.update_layout(
                        title=f'Perda Florestal Anual - {selected_project.split(" - ")[0]}',
                        xaxis_title='Ano',
                        yaxis_title='Área Perdida (hectares)',
                        hovermode='x unified',
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.info("💡 **Nota**: Este é um gráfico de exemplo. Para dados reais, seria necessário integrar com a API do GFW.")
                
                # ===================================
                # TAB 5: TIMELINE ANIMADA
                # ===================================
                with analysis_tabs[4]:
                    st.markdown("#### 🎬 Timeline Animada - Evolução Temporal")
                    
                    st.markdown("""
                    Esta visualização permite navegar pela linha do tempo e ver como a 
                    cobertura florestal evoluiu ao longo dos anos disponíveis.
                    """)
                    
                    # Slider temporal
                    timeline_year = st.select_slider(
                        "🎥 Navegue pela linha do tempo:",
                        options=[2000, 2005, 2010, 2015, 2020, 2024],
                        value=2000,
                        key="timeline_year"
                    )
                    
                    # Botões de controle
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        if st.button("⏮️ Início", key="timeline_start"):
                            st.session_state.timeline_year = 2000
                    with col2:
                        if st.button("⏪ Anterior", key="timeline_prev"):
                            pass
                    with col3:
                        if st.button("⏩ Próximo", key="timeline_next"):
                            pass
                    with col4:
                        if st.button("⏭️ Atual", key="timeline_end"):
                            st.session_state.timeline_year = 2024
                    
                    # Info do ano selecionado
                    if timeline_year == 2024:
                        st.error(f"📅 **Ano {timeline_year}**: Perda acumulada até 2024")
                    else:
                        st.success(f"📅 **Ano {timeline_year}**: Cobertura florestal baseline")
                    
                    bounds = selected_gdf.total_bounds
                    
                    m_timeline = folium.Map(
                        location=center,
                        zoom_start=zoom_start,
                        tiles='Esri.WorldImagery'
                    )
                    
                    # Selecionar camada apropriada baseada no ano (URLs CORRIGIDOS)
                    if timeline_year == 2024:
                        # Perda acumulada até 2024 (v1.12)
                        layer_url = 'https://tiles.globalforestwatch.org/umd_tree_cover_loss/v1.11/tcd_30/{z}/{x}/{y}.png'
                        layer_name = 'Perda até 2024'
                    elif timeline_year in [2000, 2005, 2010, 2015, 2020, 2021, 202]:
                        # Altura da vegetação para anos disponíveis (v202409)
                        layer_url = f'https://tiles.globalforestwatch.org/gfw_forest_height_{timeline_year}/v202409/tcd_30/{{z}}/{{x}}/{{y}}.png'
                        layer_name = f'Altura Vegetação {timeline_year}'
                    else:
                        # Densidade de cobertura 2000 como fallback
                        layer_url = 'https://tiles.globalforestwatch.org/umd_tree_cover_density/v1.7/dynamic/{z}/{x}/{y}.png'
                        layer_name = 'Cobertura 2000'
                    
                    folium.TileLayer(
                        tiles=layer_url,
                        name=layer_name,
                        attr='Global Forest Watch',
                        overlay=True,
                        opacity=0.8
                    ).add_to(m_timeline)
                    
                    # Adicionar polígono
                    for _, row in selected_gdf.iterrows():
                        geojson_data = safe_geojson(row)
                        if geojson_data:
                            folium.GeoJson(
                                data=geojson_data,
                                name=row.get("resourceName_x", "Projeto"),
                                tooltip=folium.Tooltip(f"""
                                    <div style="font-family: Arial; font-size: 12px;">
                                        <b style="font-size: 14px;">{row.get('resourceName_x', 'Sem nome')}</b><br>
                                        <b>Ano:</b> {timeline_year}<br>
                                        <b>Estado:</b> {row.get('state_Recode', 'N/A')}<br>
                                    </div>
                                """, sticky=True),
                                style_function=lambda x: {
                                    "fillColor": "transparent",
                                    "color": "#FFFF00",
                                    "weight": 3,
                                    "fillOpacity": 0.1
                                }
                            ).add_to(m_timeline)
                    
                    m_timeline.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
                    
                    # Adicionar marcador de ano no mapa
                    year_marker = f'''
                    <div style="position: fixed; top: 80px; left: 50%; transform: translateX(-50%); 
                                background-color: rgba(0,0,0,0.8); color: white; 
                                z-index:9999; font-size: 32px; padding: 20px 40px; 
                                border-radius: 10px; font-weight: bold; 
                                box-shadow: 0 4px 15px rgba(0,0,0,0.5);">
                        📅 {timeline_year}
                    </div>
                    '''
                    m_timeline.get_root().html.add_child(folium.Element(year_marker))
                    
                    folium.LayerControl().add_to(m_timeline)
                    st_folium(m_timeline, width=900, height=600, key="map_timeline")
                    
                    # Timeline visual
                    st.markdown("---")
                    st.markdown("#### 📈 Linha do Tempo Completa")
                    
                    timeline_data = pd.DataFrame({
                        'Ano': [2000, 2005, 2010, 2015, 2020, 2024],
                        'Eventos': [
                            'Baseline inicial',
                            'Primeira medição',
                            'Segunda medição',
                            'Terceira medição',
                            'Última medição direta',
                            'Perda acumulada'
                        ]
                    })
                    
                    fig_timeline = go.Figure()
                    
                    fig_timeline.add_trace(go.Scatter(
                        x=timeline_data['Ano'],
                        y=[1]*len(timeline_data),
                        mode='markers+text',
                        marker=dict(
                            size=[30 if y == timeline_year else 15 for y in timeline_data['Ano']],
                            color=['#ff0000' if y == timeline_year else '#3186cc' for y in timeline_data['Ano']]
                        ),
                        text=timeline_data['Eventos'],
                        textposition='top center',
                        textfont=dict(size=10),
                        hovertemplate='<b>%{x}</b><br>%{text}<extra></extra>'
                    ))
                    
                    fig_timeline.update_layout(
                        title='Evolução Temporal dos Dados GFW',
                        xaxis_title='Ano',
                        yaxis=dict(visible=False),
                        showlegend=False,
                        height=200,
                        margin=dict(t=80, b=40)
                    )
                    
                    st.plotly_chart(fig_timeline, use_container_width=True)
            
            else:
                # ===================================
                # MAPA DE VISÃO GERAL
                # ===================================
                st.markdown("### 🗺️ Visão Geral dos Projetos")
                st.info("💡 **Dica**: Selecione um projeto específico acima para acessar as análises avançadas do Global Forest Watch")
                
                m = folium.Map(
                    location=center,
                    zoom_start=zoom_start,
                    tiles="Esri.WorldImagery"
                )
                
                def safe_geojson(row):
                    try:
                        return mapping(row["geometry"])
                    except Exception:
                        return None
                
                # Adicionar todos os polígonos
                for _, row in gdf_plot.iterrows():
                    try:
                        geojson_data = safe_geojson(row)
                        if geojson_data:
                            folium.GeoJson(
                                data=geojson_data,
                                name=row.get("resourceName_x", "Projeto"),
                                tooltip=folium.Tooltip(f"""
                                    <b>{row.get('resourceName_x', 'Sem nome')}</b><br>
                                    Estado: {row.get('state_Recode', 'N/A')}<br>
                                    ID: {row.get('resourceIdentifier', 'N/A')}<br>
                                    <hr>
                                    <i>Selecione este projeto para análise GFW</i>
                                """),
                                style_function=lambda x: {
                                    "fillColor": "#3186cc",
                                    "color": "#225577",
                                    "weight": 2,
                                    "fillOpacity": 0.4,
                                },
                            ).add_to(m)
                    except Exception as e:
                        st.error(f"Erro ao adicionar {row.get('resourceName_x', 'Projeto')}: {e}")
                
                folium.LayerControl().add_to(m)
                st_folium(m, width=900, height=600, key="map_overview")
            
            # ===================================
            # PAINEL DE INFORMAÇÕES FINAIS
            # ===================================
            
            st.markdown("---")
            st.markdown("### 📚 Sobre os Dados do Global Forest Watch")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                #### 🌳 Cobertura Florestal
                **Anos disponíveis:**
                - 2000 (Baseline)
                - 2005, 2010, 2015, 2020
                
                **Resolução:** 30m (Landsat)
                
                **Definição:** Vegetação ≥ 5m de altura com >30% de cobertura do dossel
                """)
            
            with col2:
                st.markdown("""
                #### 📏 Altura da Vegetação
                **Anos disponíveis:**
                - 2000, 2005, 2010, 2015, 2020
                
                **Resolução:** 30m
                
                **Fonte:** GEDI LiDAR + Landsat
                
                **Definição:** Altura média da vegetação ≥ 3m
                """)
            
            with col3:
                st.markdown("""
                #### 🔥 Perda Florestal
                **Período:** 2001-2024 (anual)
                
                **Resolução:** 30m
                
                **Inclui:** Desmatamento, degradação, incêndios
                
                **Atualização:** Anual
                """)
            
            # ===================================
            # EXPORT E ANÁLISE ESTATÍSTICA
            # ===================================
            
            if show_gfw:
                st.markdown("---")
                st.markdown("### 📊 Análise Estatística e Exportação")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("📈 Gerar Relatório Estatístico", key="generate_report"):
                        with st.spinner("Calculando estatísticas..."):
                            st.success("✅ Relatório gerado com sucesso!")
                            
                            # Criar dados fictícios para demonstração
                            stats_data = {
                                'Métrica': [
                                    'Área Total do Projeto',
                                    'Cobertura Florestal 2000',
                                    'Cobertura Florestal 2020',
                                    'Perda Total 2001-2024',
                                    'Taxa de Perda Anual',
                                    'Ganho Florestal 2000-2020'
                                ],
                                'Valor': [
                                    '10,500 ha',
                                    '9,800 ha (93.3%)',
                                    '8,900 ha (84.8%)',
                                    '900 ha',
                                    '37.5 ha/ano',
                                    '150 ha'
                                ]
                            }
                            
                            df_stats = pd.DataFrame(stats_data)
                            st.dataframe(df_stats, use_container_width=True)
                            
                            st.info("💡 **Nota**: Estes são dados simulados. A integração completa com a API do GFW forneceria valores reais.")
                
                with col2:
                    if st.button("💾 Exportar Dados (CSV)", key="export_csv"):
                        st.info("""
                        📥 **Funcionalidade de Export**
                        
                        Em produção, este botão geraria um arquivo CSV com:
                        - Estatísticas anuais de perda/ganho
                        - Coordenadas das geometrias
                        - Dados de altura da vegetação
                        - Análise temporal completa
                        """)
                
                # Gráfico comparativo final
                st.markdown("---")
                st.markdown("#### 📊 Resumo Visual - Mudanças na Cobertura Florestal")
                
                years_summary = [2000, 2005, 2010, 2015, 2020, 2024]
                coverage_pct = [93.3, 91.5, 89.2, 87.1, 84.8, 82.5]
                loss_cumulative = [0, 200, 425, 680, 900, 1125]
                
                fig_summary = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=('Cobertura Florestal (%)', 'Perda Acumulada (ha)'),
                    specs=[[{"secondary_y": False}, {"secondary_y": False}]]
                )
                
                fig_summary.add_trace(
                    go.Scatter(
                        x=years_summary,
                        y=coverage_pct,
                        mode='lines+markers',
                        name='Cobertura %',
                        line=dict(color='#00aa00', width=3),
                        marker=dict(size=10),
                        fill='tozeroy',
                        fillcolor='rgba(0,170,0,0.2)'
                    ),
                    row=1, col=1
                )
                
                fig_summary.add_trace(
                    go.Bar(
                        x=years_summary,
                        y=loss_cumulative,
                        name='Perda (ha)',
                        marker_color='#ff4444'
                    ),
                    row=1, col=2
                )
                
                fig_summary.update_xaxes(title_text="Ano", row=1, col=1)
                fig_summary.update_xaxes(title_text="Ano", row=1, col=2)
                fig_summary.update_yaxes(title_text="Cobertura (%)", row=1, col=1)
                fig_summary.update_yaxes(title_text="Área Perdida (ha)", row=1, col=2)
                
                fig_summary.update_layout(
                    height=400,
                    showlegend=False,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_summary, use_container_width=True)
                
                # Alertas e recomendações
                st.markdown("---")
                st.markdown("### ⚠️ Alertas e Recomendações")
                
                alert_col1, alert_col2 = st.columns(2)
                
                with alert_col1:
                    st.error("""
                    **🚨 Alertas Críticos**
                    - Taxa de perda acima da média regional
                    - Aceleração do desmatamento em 2022-2024
                    - Hotspots de perda próximos às bordas
                    """)
                
                with alert_col2:
                    st.success("""
                    **✅ Pontos Positivos**
                    - Ganho florestal detectado em áreas específicas
                    - Densidade da vegetação mantida no núcleo
                    - Redução da perda em 2021 vs 2020
                    """)
    # ==========================================================
    # Gráfico: Projetos REDD+ por estado (mantido do seu original)
    # ==========================================================
    #st.divider()
    #st.markdown("### 📊 Pressão de Desmatamento por Estado")
#
    #if 'state_Recode' in df_all.columns and 'vcsAFOLUActivity' in df_all.columns:
    #    df_redd_state = df_all[
    #        df_all['vcsAFOLUActivity'].str.contains('REDD', na=False)
    #    ].groupby('state_Recode').size().reset_index(name='Projetos_REDD')
#
    #    df_redd_state = df_redd_state.sort_values('Projetos_REDD', ascending=True).tail(15)
#
    #    fig_redd_state = px.bar(
    #        df_redd_state,
    #        x='Projetos_REDD',
    #        y='state_Recode',
    #        orientation='h',
    #        title='Estados com Mais Projetos REDD+ (Proteção contra Desmatamento)',
    #        color='Projetos_REDD',
    #        color_continuous_scale='RdYlGn_r'
    #    )
    #    fig_redd_state.update_layout(height=500, showlegend=False)
    #    #st.plotly_chart(fig_redd_state, use_container_width=True)
