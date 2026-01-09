import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests

# --- CONFIGURAÇÕES TÉCNICAS ---
NOME_SISTEMA = "BOVESPA Viewer"
VERSAO = "v1.0.20260109"
ICONE_APP = "📈"
PROGRAMADOR = "PRINCE, K.B"
COR_CRITICO, COR_ALTO, COR_MEDIO, COR_BAIXO = "#FF0000", "#FF4B4B", "#FFA500", "#00FF00"

st.set_page_config(page_title=NOME_SISTEMA, page_icon=ICONE_APP, layout="wide")

# --- ESTILIZAÇÃO CSS AVANÇADA (UX REFINADA) ---
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');

    html, body, [data-testid="stSidebar"] {{font-family: 'JetBrains Mono', monospace;}}

    .main-title {{
        text-align: center; color: {COR_ALTO}; font-weight: bold; 
        letter-spacing: 5px; border-bottom: 2px solid {COR_ALTO};
        padding: 20px; margin-bottom: 30px; background: rgba(255, 75, 75, 0.05);
    }}

    .stock-card {{
        background: rgba(30, 30, 30, 0.6);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.1);
        padding: 20px; border-radius: 12px;
        border-left: 8px solid; margin-bottom: 15px;
        transition: transform 0.3s ease, border 0.3s ease;
    }}
    .stock-card:hover {{
        transform: scale(1.01);
        border-right: 1px solid {COR_ALTO};
    }}

    .positivo {{border-left-color: {COR_BAIXO}; box-shadow: -5px 0 15px rgba(0,255,0,0.2);}}
    .negativo {{border-left-color: {COR_CRITICO}; box-shadow: -5px 0 15px rgba(255,0,0,0.2);}}
    .neutro {{border-left-color: {COR_MEDIO};}}

    .metric-box {{
        background: rgba(0, 150, 255, 0.1);
        border-radius: 8px; padding: 15px;
        margin: 10px 0; font-size: 0.9em;
        border: 1px solid rgba(0, 150, 255, 0.3);
        text-align: center;
    }}
    </style>
    """, unsafe_allow_html=True)


# --- FUNÇÕES AUXILIARES ---
@st.cache_data(ttl=3600)
def obter_acoes_bovespa():
    """Retorna lista das principais ações da Bovespa"""
    # Lista das principais ações da B3
    acoes_principais = [
        "PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBDC4.SA", "ABEV3.SA",
        "BBAS3.SA", "WEGE3.SA", "RENT3.SA", "GGBR4.SA", "SUZB3.SA",
        "RAIL3.SA", "JBSS3.SA", "MGLU3.SA", "B3SA3.SA", "HAPV3.SA",
        "RADL3.SA", "CSAN3.SA", "ELET3.SA", "CMIG4.SA", "CPLE6.SA",
        "EMBR3.SA", "LREN3.SA", "UGPA3.SA", "VIVT3.SA", "TOTS3.SA",
        "PRIO3.SA", "BPAC11.SA", "SANB11.SA", "CRFB3.SA", "ASAI3.SA"
    ]
    return acoes_principais


@st.cache_data(ttl=300)
def obter_dados_acao(ticker, periodo="1mo"):
    """Obtém dados históricos de uma ação"""
    try:
        acao = yf.Ticker(ticker)
        hist = acao.history(period=periodo)
        info = acao.info
        return hist, info
    except:
        return None, None


def calcular_variacao(hist):
    """Calcula a variação percentual"""
    if hist is None or len(hist) < 2:
        return 0
    preco_inicial = hist['Close'].iloc[0]
    preco_final = hist['Close'].iloc[-1]
    return ((preco_final - preco_inicial) / preco_inicial) * 100


def classificar_variacao(variacao):
    """Classifica a variação e retorna classe CSS"""
    if variacao > 0:
        return "positivo", "🟢"
    elif variacao < 0:
        return "negativo", "🔴"
    else:
        return "neutro", "⚪"


# --- UI PRINCIPAL ---
st.markdown(f"<h1 class='main-title'>📈 {NOME_SISTEMA}</h1>", unsafe_allow_html=True)

# SIDEBAR
st.sidebar.header("🕹️ CONTROL CENTER")
periodo_selecionado = st.sidebar.selectbox(
    "📅 Período de Análise",
    ["1d", "5d", "1mo", "3mo", "6mo", "1y", "5y"],
    index=2
)

filtro_setor = st.sidebar.multiselect(
    "🏢 Filtrar por Setor",
    ["Todos", "Financeiro", "Energia", "Varejo", "Mineração", "Tecnologia"],
    default=["Todos"]
)

mostrar_detalhes = st.sidebar.checkbox("📊 Mostrar Detalhes Expandidos", value=False)

# --- TABS PRINCIPAIS ---
tabs = st.tabs(["📡 Dashboard Geral", "📊 Análise Individual", "📚 Ranking"])

with tabs[0]:
    st.subheader("📊 Visão Geral do Mercado")

    # Obter lista de ações
    acoes = obter_acoes_bovespa()

    # Criar métricas gerais
    col1, col2, col3, col4 = st.columns(4)

    dados_resumo = []

    # Barra de progresso
    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, ticker in enumerate(acoes):
        status_text.text(f"Carregando {ticker}... ({idx + 1}/{len(acoes)})")
        progress_bar.progress((idx + 1) / len(acoes))

        hist, info = obter_dados_acao(ticker, periodo_selecionado)

        if hist is not None and len(hist) > 0:
            variacao = calcular_variacao(hist)
            preco_atual = hist['Close'].iloc[-1]
            nome = info.get('shortName', ticker.replace('.SA', '')) if info else ticker.replace('.SA', '')

            dados_resumo.append({
                'Ticker': ticker,
                'Nome': nome,
                'Preço': preco_atual,
                'Variação (%)': variacao,
                'Volume': hist['Volume'].iloc[-1] if 'Volume' in hist.columns else 0
            })

    progress_bar.empty()
    status_text.empty()

    # Criar DataFrame
    df_resumo = pd.DataFrame(dados_resumo)

    # Métricas gerais
    with col1:
        st.metric("📈 Total de Ações", len(df_resumo))
    with col2:
        acoes_alta = len(df_resumo[df_resumo['Variação (%)'] > 0])
        st.metric("🟢 Em Alta", acoes_alta)
    with col3:
        acoes_baixa = len(df_resumo[df_resumo['Variação (%)'] < 0])
        st.metric("🔴 Em Baixa", acoes_baixa)
    with col4:
        variacao_media = df_resumo['Variação (%)'].mean()
        st.metric("📊 Variação Média", f"{variacao_media:.2f}%")

    # Gráfico de dispersão
    st.subheader("🎯 Mapa de Desempenho")
    fig = px.scatter(df_resumo,
                     x='Volume',
                     y='Variação (%)',
                     size='Preço',
                     color='Variação (%)',
                     hover_data=['Nome', 'Ticker'],
                     text='Ticker',
                     color_continuous_scale=['red', 'yellow', 'green'],
                     title="Volume vs Variação (tamanho = preço)")

    fig.update_traces(textposition='top center', textfont_size=8)
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    # Tabela de ações
    st.subheader("📋 Todas as Ações")
    df_display = df_resumo.sort_values('Variação (%)', ascending=False)
    df_display['Variação (%)'] = df_display['Variação (%)'].apply(lambda x: f"{x:.2f}%")
    df_display['Preço'] = df_display['Preço'].apply(lambda x: f"R$ {x:.2f}")
    st.dataframe(df_display, use_container_width=True, height=400)

with tabs[1]:
    st.subheader("🔍 Análise Detalhada de Ação")

    ticker_selecionado = st.selectbox("Selecione uma ação:", acoes)

    if ticker_selecionado:
        hist, info = obter_dados_acao(ticker_selecionado, periodo_selecionado)

        if hist is not None and info:
            # Informações básicas
            col1, col2, col3 = st.columns(3)

            with col1:
                nome_empresa = info.get('shortName', ticker_selecionado)
                st.markdown(f"### {nome_empresa}")
                st.markdown(f"**Ticker:** {ticker_selecionado}")

            with col2:
                preco_atual = hist['Close'].iloc[-1]
                variacao = calcular_variacao(hist)
                st.metric("💰 Preço Atual", f"R$ {preco_atual:.2f}", f"{variacao:.2f}%")

            with col3:
                volume_medio = hist['Volume'].mean()
                st.metric("📊 Volume Médio", f"{volume_medio:,.0f}")

            # Gráfico de candlestick
            st.subheader("📈 Gráfico de Candlestick")
            fig = go.Figure(data=[go.Candlestick(
                x=hist.index,
                open=hist['Open'],
                high=hist['High'],
                low=hist['Low'],
                close=hist['Close']
            )])

            fig.update_layout(
                title=f"{ticker_selecionado} - Período: {periodo_selecionado}",
                yaxis_title="Preço (R$)",
                xaxis_title="Data",
                height=500
            )

            st.plotly_chart(fig, use_container_width=True)

            # Gráfico de volume
            st.subheader("📊 Volume de Negociação")
            fig_volume = px.bar(hist, y='Volume', title="Volume ao longo do tempo")
            fig_volume.update_layout(height=300)
            st.plotly_chart(fig_volume, use_container_width=True)

            if mostrar_detalhes:
                st.subheader("ℹ️ Informações Adicionais")
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown(f"**Setor:** {info.get('sector', 'N/A')}")
                    st.markdown(f"**Indústria:** {info.get('industry', 'N/A')}")
                    st.markdown(f"**Market Cap:** {info.get('marketCap', 'N/A')}")

                with col2:
                    st.markdown(f"**P/E Ratio:** {info.get('trailingPE', 'N/A')}")
                    st.markdown(f"**Dividend Yield:** {info.get('dividendYield', 'N/A')}")
                    st.markdown(f"**52W High:** R$ {info.get('fiftyTwoWeekHigh', 'N/A')}")

with tabs[2]:
    st.subheader("🏆 Ranking de Performance")

    if len(dados_resumo) > 0:
        df_ranking = pd.DataFrame(dados_resumo).sort_values('Variação (%)', ascending=False)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🟢 Top 10 Maiores Altas")
            top_altas = df_ranking.head(10)

            for idx, row in top_altas.iterrows():
                classe, emoji = classificar_variacao(row['Variação (%)'])
                st.markdown(f"""
                <div class='stock-card {classe}'>
                    <h4>{emoji} {row['Nome']}</h4>
                    <p><strong>Ticker:</strong> {row['Ticker']}</p>
                    <p><strong>Preço:</strong> R$ {row['Preço']:.2f}</p>
                    <p><strong>Variação:</strong> {row['Variação (%)']:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)

        with col2:
            st.markdown("### 🔴 Top 10 Maiores Baixas")
            top_baixas = df_ranking.tail(10)

            for idx, row in top_baixas.iterrows():
                classe, emoji = classificar_variacao(row['Variação (%)'])
                st.markdown(f"""
                <div class='stock-card {classe}'>
                    <h4>{emoji} {row['Nome']}</h4>
                    <p><strong>Ticker:</strong> {row['Ticker']}</p>
                    <p><strong>Preço:</strong> R$ {row['Preço']:.2f}</p>
                    <p><strong>Variação:</strong> {row['Variação (%)']:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)

# Footer
st.markdown(
    f'<div style="text-align:center; color:#555; font-size:10px; margin-top:50px;">{NOME_SISTEMA} | {VERSAO} | BY {PROGRAMADOR}</div>',
    unsafe_allow_html=True
)