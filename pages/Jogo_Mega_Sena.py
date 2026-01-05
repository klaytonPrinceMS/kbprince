import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
import plotly.express as px
import plotly.graph_objects as go

from app import LINK_PESSOAL

# --- 1. CONFIGURAÇÕES TÉCNICAS E ESTÉTICAS ---
ARQUIVO_CAIXA = "resultadoJogoMegaSena.xlsx"
CAIXA_URL = "https://servicebus2.caixa.gov.br/portaldeloterias/api/resultados/download?modalidade=Mega-Sena"
COR_QUENTE = "#FF4B4B"
COR_FRIO = "#007BFF"
NOME_SISTEMA = "Mega Sena"
VERSAO = "SIPP v1.0.20260104"
PROGRAMADOR = "PRINCE, K.B"


st.set_page_config(
    page_title=NOME_SISTEMA,
    page_icon="🍀",
    layout="wide",
    initial_sidebar_state='expanded',
    menu_items={"About": LINK_PESSOAL}
)

st.markdown(f"""
    <style>
    .main-title {{text-align: center; color: {COR_QUENTE}; font-weight: bold; margin-bottom: 20px;}}
    .stButton>button {{width: 100%; font-weight: bold; border-radius: 10px; height: 45px;}}
    .footer-text {{text-align: center; padding: 20px; color: #888; font-size: 14px;}}
    </style>
    """, unsafe_allow_html=True)


# --- 2. MOTOR DE DADOS (LOGICA SIPP) ---

def baixar_dados():
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(CAIXA_URL, headers=headers, timeout=30)
        if res.status_code == 200:
            with open(ARQUIVO_CAIXA, "wb") as f: f.write(res.content)
            return True
        return False
    except:
        return False
@st.cache_data
def processar_base_completa():
    if not os.path.exists(ARQUIVO_CAIXA): baixar_dados()
    try:
        df = pd.read_excel(ARQUIVO_CAIXA, engine='openpyxl')
        df.columns = [str(c).strip() for c in df.columns]
        df = df.dropna(subset=["Concurso"])
        df["Concurso"] = df["Concurso"].astype(int)

        # Gerar Matriz Binária
        cols_bolas = ['Bola1', 'Bola2', 'Bola3', 'Bola4', 'Bola5', 'Bola6']
        df_melt = df.melt(id_vars=["Concurso"], value_vars=cols_bolas, value_name='N').dropna()
        df_bin = pd.crosstab(df_melt["Concurso"], df_melt['N'])
        for i in range(1, 61):
            if i not in df_bin.columns: df_bin[i] = 0
        df_bin = df_bin.reindex(columns=sorted(df_bin.columns)).reset_index()
        return df, df_bin
    except:
        return None, None




# --- 3. INTERFACE E CONFIGURAÇÕES REFINADAS ---
st.markdown(f"<h1 class='main-title'>🍀 {NOME_SISTEMA}</h1>", unsafe_allow_html=True)

df_bruto, df_binario = processar_base_completa()

if df_bruto is not None:
    # --- SIDEBAR: CONFIGURAÇÃO COMPLETA ---
    st.sidebar.header("⚙️ Mineração")
    min_c, max_c = int(df_bruto["Concurso"].min()), int(df_bruto["Concurso"].max())

    # Atalhos rápidos
    st.sidebar.write("**Seleção de Ciclo:**")
    c1, c2 = st.sidebar.columns(2)
    if c1.button("Últimos 100"): st.session_state.ini, st.session_state.fim = max_c - 100, max_c
    if c2.button("Base Toda"): st.session_state.ini, st.session_state.fim = min_c, max_c

    # Slider e Input para Precisão 100%
    val_ini = st.session_state.get('ini', max_c - 500)
    val_fim = st.session_state.get('fim', max_c)
    range_c = st.sidebar.slider("Janela Temporal:", min_c, max_c, (val_ini, val_fim))

    col_i, col_f = st.sidebar.columns(2)
    c_ini = col_i.number_input("Início:", min_c, max_c, range_c[0])
    c_fim = col_f.number_input("Fim:", min_c, max_c, range_c[1])

    qtd_top = st.sidebar.slider("Amostra SIPP (Top N):", 4, 30, 15)
    metodo = st.sidebar.selectbox("Peso de Recência:", ["Linear", "Exponencial"])

    if st.sidebar.button("🔄 ATUALIZAR DB"):
        if baixar_dados(): st.cache_data.clear(); st.rerun()

    # --- PROCESSAMENTO DOS DADOS FILTRADOS ---
    df_bruto_f = df_bruto[(df_bruto["Concurso"] >= c_ini) & (df_bruto["Concurso"] <= c_fim)].copy()
    df_bin_f = df_binario[
        (df_binario["Concurso"].astype(int) >= c_ini) & (df_binario["Concurso"].astype(int) <= c_fim)].copy()

    # Cálculo do Score SIPP
    weights = np.exp(np.linspace(0, 3, len(df_bin_f))) if metodo == "Exponencial" else np.linspace(0.1, 1.0,
                                                                                                   len(df_bin_f))
    scores = df_bin_f.drop(columns=["Concurso"]).astype(float).multiply(weights, axis=0).sum()
    df_ranking = pd.DataFrame({'Número': scores.index.astype(int), 'Score': scores.values}).sort_values('Score',
                                                                                                        ascending=False)

    # --- IMPLEMENTAÇÃO DAS TABS SOLICITADAS ---
    t_bruto, t_bin, t_tend, t_perc, t_sipp, t_irmaos, t_inimigos, t_gov = st.tabs(
        ["📊 Dados", "🔢 Binário", "📈 Tendências", "🎯 Percentis", "🚀 SIPP", "👨🏻‍🤝‍👨🏿 Irmãos", "⚔️ Inimigos","🛡️ Governança"]
    )

    with t_bruto:
        st.subheader("📋 Histórico Oficial de Concursos")
        st.dataframe(df_bruto_f.iloc[::-1], use_container_width=True, hide_index=True)

    with t_bin:
        st.subheader("🔢 Matriz de Identificação (0 e 1)")
        st.dataframe(df_bin_f.iloc[::-1], use_container_width=True, hide_index=True)

    with t_tend:
        st.subheader("📈 Análise de Paridades e Somas")
        c1, c2 = st.columns(2)
        cols_b = ['Bola1', 'Bola2', 'Bola3', 'Bola4', 'Bola5', 'Bola6']

        # Cálculo de Paridade
        paridade = df_bruto_f[cols_b].apply(lambda x: sum(1 for n in x if n % 2 == 0), axis=1)
        df_par = pd.DataFrame({'Concurso': df_bruto_f["Concurso"], 'Pares': paridade, 'Ímpares': 6 - paridade, 'Soma': df_bruto_f[cols_b].sum(axis=1)})

        with c1:
            st.write("**Distribuição Par vs Ímpar**")
            st.bar_chart(df_par.set_index('Concurso')[['Pares', 'Ímpares']])
        with c2:
            st.write("**Evolução da Soma das Dezenas**")
            st.line_chart(df_par.set_index('Concurso')['Soma'])

        st.write("**Tabela de Tendências Analíticas**")
        st.dataframe(df_par.iloc[::-1], use_container_width=True, hide_index=True)

    with t_perc:
        st.subheader("🎯 Análise de Percentis (Alta e Baixa)")
        p_alta = df_ranking['Score'].quantile(0.80)
        p_baixa = df_ranking['Score'].quantile(0.20)

        c1, c2 = st.columns(2)
        c1.metric("Corte Percentil 80 (Quentes)", f"{p_alta:.2f}")
        c2.metric("Corte Percentil 20 (Frios)", f"{p_baixa:.2f}")

        st.write("**Distribuição Acumulada de Scores**")
        fig_p = px.ecdf(df_ranking, x="Score", title="Curva de Probabilidade Acumulada")
        st.plotly_chart(fig_p, use_container_width=True)

    with t_sipp:
        st.subheader("🚀 Motor SIPP Dual (Fogo e Gelo)")
        top = df_ranking.head(qtd_top)
        bottom = df_ranking.tail(qtd_top)

        fig = go.Figure()
        fig.add_trace(go.Bar(x=top['Número'].astype(str), y=top['Score'], name='Quentes', marker_color=COR_QUENTE))
        fig.add_trace(go.Bar(x=bottom['Número'].astype(str), y=bottom['Score'], name='Frios', marker_color=COR_FRIO))
        fig.update_layout(title="Extremos de Tendência", barmode='group')
        st.plotly_chart(fig, use_container_width=True)

        st.write("**Ranking Geral (1 a 60)**")
        st.dataframe(df_ranking, use_container_width=True, hide_index=True)

    with t_irmaos:
        st.subheader("👯 Números Irmãos (Alta Afinidade)")
        matriz_co = pd.DataFrame(
            df_bin_f.drop(columns=["Concurso"]).T.astype(int) @ df_bin_f.drop(columns=["Concurso"]).astype(int))
        irmaos = []
        for n in top['Número']:
            parc = matriz_co[n].drop(labels=[n]).sort_values(ascending=False).head(3).index.tolist()
            irmaos.append({"Base": n, "Irmão 1": parc[0], "Irmão 2": parc[1], "Irmão 3": parc[2]})
        st.table(pd.DataFrame(irmaos))

    with t_inimigos:
        st.subheader("⚔️ Números Inimigos (Baixa Afinidade)")
        inimigos = []
        for n in top['Número']:
            parc = matriz_co[n].drop(labels=[n]).sort_values(ascending=True).head(3).index.tolist()
            inimigos.append({"Base": n, "Inimigo 1": parc[0], "Inimigo 2": parc[1], "Inimigo 3": parc[2]})
        st.table(pd.DataFrame(inimigos))

    with t_gov:
        st.subheader("🛡️ Governança, Ética e Transparência")
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Integridade dos Dados**")
            integridade = (len(df_bruto_f) / (c_fim - c_ini + 1)) * 100 if (c_fim - c_ini) > 0 else 100
            st.success(f"Nível de Integridade: {integridade:.2f}%")
            st.info(f"Amostra analisadas: {len(df_bruto_f)}")
        with c2:
            st.write("**Fonte e Ética**")
            st.write("- **Fonte:** Loterias Caixa (Dados Oficiais)")
            st.write("- **Uso Ético:** Este sistema é para fins de estudo estatístico. Não garante ganhos financeiros.")
            st.write("- **Transparência:** Algoritmos baseados na metodologia SIPP (Silva et al., 2016).")

st.markdown(f'<div class="footer-text notranslate">© {NOME_SISTEMA} {VERSAO} | 2026 | By: {PROGRAMADOR} | T! SOS Sistemas</div>', unsafe_allow_html=True)





