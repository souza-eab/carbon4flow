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
    value="1XhuQ1hnqD_ac7Ki_ICq3rGYEX34uWrNA",
    help="ID do arquivo Parquet no Google Drive"
)

file_id_credit = st.sidebar.text_input(
    "ID - Projetos com Créditos",
    value="1fMkQAucgj9b8xZNr7Iv-jQnuIbiVw8bG",
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
st.sidebar.metric(" Projetos com lasto de Créditos Aposentados", f"{df_credit['resourceName_x'].nunique():,}" if 'resourceName_x' in df_credit.columns else f"{len(df_credit):,}")
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

tabs = st.tabs([
    "📊 Visão Geral",
    "🌎 Mapa - Todos os Projetos BR",
    "💰 Mapa - Com lastro de Créditos/Vendas",
    "📈 Análises de Vintage",
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
        state_counts = df_overview["state_Recode"].value_counts().head(20).reset_index()
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
        if selected_points and selected_points.get("selection"):
            if selected_points["selection"].get("points"):
                clicked_state = selected_points["selection"]["points"][0].get("y")
                if clicked_state:
                    st.session_state.selected_state_overview = clicked_state
                    st.rerun()

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
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            #col_m1,  col_m3, col_m4 = st.columns(3)
            
            with col_m1:
                if 'Mean' in df_proj.columns:
                    st.metric("Média VCUs", f"{df_proj['Mean'].iloc[0]:,.0f}")
            
            #with col_m2:
            #    if 'EAER' in df_proj.columns:
            #        #st.metric("EAER", f"{df_proj['EAER'].iloc[0]:,.0f}")
            #        eaer_value = pd.to_numeric(df_proj["EAER"].iloc[0], errors="coerce")        
            #        st.metric("EAER", f"{eaer_value:,.0f}" if not np.isnan(eaer_value) else "N/A")

            with col_m3:
                if 'protocol' in df_proj.columns:
                    st.metric("Protocolo", df_proj['protocol'].iloc[0])
            
            with col_m4:
                if 'vcsProjectStatus' in df_proj.columns:
                    st.metric("Status", df_proj['vcsProjectStatus'].iloc[0])
            
            st.divider()
            
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
                height=600,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
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

            with tabs[4]:
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
            # FOOTER
            # =====================================

            st.divider()

            footer_cols = st.columns([2, 1])
            with footer_cols[0]:
                st.caption("🌎 MRV Data & Map Dashboard | Desenvolvido com Streamlit | Dados atualizados automaticamente do Google Drive")
            with footer_cols[1]:
                st.caption(f"Versão 0.0.2 | {datetime.now().strftime('%Y')}")
                
