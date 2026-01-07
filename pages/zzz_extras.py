import streamlit as st

PROGRAMADOR = "PRINCE, K.B"
NOME_SISTEMA = "T! SOS Sistemas"
VERSAO = "TSistemas v1.0.20250104"
LINK_PESSOAL = "https://klaytonprincems.github.io/site/"
COR_QUENTE = "#FF4B4B"

# 1. Configuração ÚNICA da Página
st.set_page_config(
    page_title=NOME_SISTEMA,
    page_icon="📱",
    layout="centered",
    initial_sidebar_state='expanded',
    menu_items={"About": LINK_PESSOAL}
)

# 2. Estilo CSS Unificado e Organizado

st.markdown(f"""
    <style>
    #MainMenu {{visibility: hidden;}} #esconde o menu
    #header {{visibility: hidden;}} #esconde header
    #footer {{visibility: hidden;}} #esconde o footer

    .main-title {{text-align: center; color: {COR_QUENTE}; font-weight: bold; margin-bottom: 20px;}}
    .subtitle {{text-align: center; color: #666; margin-bottom: 20px;}}
    .stButton>button {{width: 100%; font-weight: bold; border-radius: 15px; border: 2px solid transparent; transition: all 0.3s ease;}}
    .stButton>button:hover {{border-color: #ff4b4b;color: #ff4b4b;transform: translateY(-5px);box-shadow: 0 4px 15px rgba(255, 75, 75, 0.2);}}
    .footer-text {{text-align: center; padding: 20px; color: #888; font-size: 14px;}}

    .custom-footer {{text-align: center;padding: 20px;margin-top: 50px;border-top: 1px solid #333;color: #288;font-size: 14px;line-height: 1.6;}}
    .custom-footer a {{color: #ff4b4b;text-decoration: none;font-weight: bold;}}
    div.stButton > button {{height: 100px;font-size: 20px !important;}}
    </style>
    """, unsafe_allow_html=True)

# 3. Cabeçalho
st.markdown("<h1 class='main-title'>T! SOS Sistemas</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Selecione uma ferramenta abaixo para iniciar</p>", unsafe_allow_html=True)
st.divider()

# 5. Rodapé Final Unificado
st.markdown(
    f'<div class="footer-text notranslate">© {NOME_SISTEMA} {VERSAO} | 2026 | By: {PROGRAMADOR} | T! SOS Sistemas</div>',
    unsafe_allow_html=True)

import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
import plotly.express as px
import plotly.graph_objects as go
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import datetime

# --- 1. CONFIGURAÇÕES TÉCNICAS E ESTÉTICAS ---
ARQUIVO_CAIXA = "resultadoJogoMegaSena.xlsx"
CAIXA_URL = "https://servicebus2.caixa.gov.br/portaldeloterias/api/resultados/download?modalidade=Mega-Sena"
COR_QUENTE = "#FF4B4B"
COR_FRIO = "#007BFF"
NOME_SISTEMA = "Mega Sena SIPP Advanced"
VERSAO = "SIPP v2.0.20260105"
PROGRAMADOR = "PRINCE, K.B & Manus AI"
LINK_PESSOAL = "https://manus.im"  # Placeholder para o link original

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
    .crisp-box {{background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid {COR_FRIO}; margin-bottom: 10px;}}
    </style>
    """, unsafe_allow_html=True)


# --- 2. MOTOR DE DADOS ---

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


# --- 3. INTERFACE E LÓGICA ---
st.markdown(f"<h1 class='main-title'>🍀 {NOME_SISTEMA}</h1>", unsafe_allow_html=True)

df_bruto, df_binario = processar_base_completa()

if df_bruto is not None:
    # --- SIDEBAR ---
    st.sidebar.header("⚙️ Mineração Avançada")
    min_c, max_c = int(df_bruto["Concurso"].min()), int(df_bruto["Concurso"].max())

    st.sidebar.write("**Seleção de Ciclo:**")
    c1, c2 = st.sidebar.columns(2)
    if c1.button("Últimos 100"): st.session_state.ini, st.session_state.fim = max_c - 100, max_c
    if c2.button("Base Toda"): st.session_state.ini, st.session_state.fim = min_c, max_c

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

    # Filtro de dados
    df_bruto_f = df_bruto[(df_bruto["Concurso"] >= c_ini) & (df_bruto["Concurso"] <= c_fim)].copy()
    df_bin_f = df_binario[(df_binario["Concurso"] >= c_ini) & (df_binario["Concurso"] <= c_fim)].copy()

    # Score SIPP
    weights = np.exp(np.linspace(0, 3, len(df_bin_f))) if metodo == "Exponencial" else np.linspace(0.1, 1.0,
                                                                                                   len(df_bin_f))
    scores = df_bin_f.drop(columns=["Concurso"]).astype(float).multiply(weights, axis=0).sum()
    df_ranking = pd.DataFrame({'Número': scores.index.astype(int), 'Score': scores.values}).sort_values('Score',
                                                                                                        ascending=False)

    # --- TABS ---
    tabs = st.tabs([
        "📊 Dados", "🔢 Binário", "📈 Tendências", "🎯 Percentis", "🚀 SIPP",
        "🔗 Associação", "🤖 Predição ML", "🧩 Clustering", "🔍 Analisador",
        "🎲 Gerador", "🛡️ Governança", "📖 CRISP-DM"
    ])

    with tabs[0]:  # Dados
        st.subheader("📋 Histórico Oficial de Concursos")
        st.dataframe(df_bruto_f.iloc[::-1], use_container_width=True, hide_index=True)

    with tabs[1]:  # Binário
        st.subheader("🔢 Matriz de Identificação (0 e 1)")
        st.dataframe(df_bin_f.iloc[::-1], use_container_width=True, hide_index=True)

    with tabs[2]:  # Tendências
        st.subheader("📈 Análise de Paridades e Somas")
        c1, c2 = st.columns(2)
        cols_b = ['Bola1', 'Bola2', 'Bola3', 'Bola4', 'Bola5', 'Bola6']
        paridade = df_bruto_f[cols_b].apply(lambda x: sum(1 for n in x if n % 2 == 0), axis=1)
        df_par = pd.DataFrame({'Concurso': df_bruto_f["Concurso"], 'Pares': paridade, 'Ímpares': 6 - paridade,
                               'Soma': df_bruto_f[cols_b].sum(axis=1)})
        with c1:
            st.write("**Distribuição Par vs Ímpar**")
            st.bar_chart(df_par.set_index('Concurso')[['Pares', 'Ímpares']])
        with c2:
            st.write("**Evolução da Soma das Dezenas**")
            st.line_chart(df_par.set_index('Concurso')['Soma'])
        st.dataframe(df_par.iloc[::-1], use_container_width=True, hide_index=True)

    with tabs[3]:  # Percentis
        st.subheader("🎯 Análise de Percentis")
        p_alta = df_ranking['Score'].quantile(0.80)
        p_baixa = df_ranking['Score'].quantile(0.20)
        c1, c2 = st.columns(2)
        c1.metric("Corte Percentil 80 (Quentes)", f"{p_alta:.2f}")
        c2.metric("Corte Percentil 20 (Frios)", f"{p_baixa:.2f}")
        fig_p = px.ecdf(df_ranking, x="Score", title="Curva de Probabilidade Acumulada")
        st.plotly_chart(fig_p, use_container_width=True)

    with tabs[4]:  # SIPP
        st.subheader("🚀 Motor SIPP Dual (Fogo e Gelo)")
        top = df_ranking.head(qtd_top)
        bottom = df_ranking.tail(qtd_top)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=top['Número'].astype(str), y=top['Score'], name='Quentes', marker_color=COR_QUENTE))
        fig.add_trace(go.Bar(x=bottom['Número'].astype(str), y=bottom['Score'], name='Frios', marker_color=COR_FRIO))
        fig.update_layout(title="Extremos de Tendência", barmode='group')
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df_ranking, use_container_width=True, hide_index=True)

    with tabs[5]:  # Associação (Apriori / FP-Growth)
        st.subheader("🔗 Mineração de Regras de Associação")
        col_alg, col_sup, col_conf = st.columns(3)
        alg_assoc = col_alg.selectbox("Algoritmo:", ["FP-Growth", "Apriori"])
        min_sup = col_sup.slider("Suporte Mínimo:", 0.001, 0.05, 0.01, format="%.3f")
        min_conf = col_conf.slider("Confiança Mínima:", 0.1, 1.0, 0.3)

        df_assoc_input = df_bin_f.drop(columns=["Concurso"]).astype(bool)

        try:
            if alg_assoc == "Apriori":
                frequent_itemsets = apriori(df_assoc_input, min_support=min_sup, use_colnames=True)
            else:
                frequent_itemsets = fpgrowth(df_assoc_input, min_support=min_sup, use_colnames=True)

            if not frequent_itemsets.empty:
                rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_conf)
                if not rules.empty:
                    rules["antecedents"] = rules["antecedents"].apply(lambda x: list(x))
                    rules["consequents"] = rules["consequents"].apply(lambda x: list(x))
                    st.write(f"**Regras Encontradas ({len(rules)}):**")
                    st.dataframe(
                        rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].sort_values('lift',
                                                                                                           ascending=False),
                        use_container_width=True)
                else:
                    st.warning("Nenhuma regra encontrada com a confiança mínima definida.")
            else:
                st.warning("Nenhum conjunto frequente encontrado com o suporte mínimo definido.")
        except Exception as e:
            st.error(f"Erro no processamento de associação: {e}")

    with tabs[6]:  # Predição ML
        st.subheader("🤖 Predição com Machine Learning (Random Forest)")
        qtd_pred = st.slider("Quantidade de dezenas para prever:", 10, 20, 15)

        if st.button("Executar Predição"):
            with st.spinner("Treinando modelo..."):
                # Preparar dados para ML (Prever se o número sai no próximo concurso baseado no histórico)
                X = []
                y = []
                window = 10
                for i in range(window, len(df_binario) - 1):
                    X.append(df_binario.iloc[i - window:i].drop(columns=["Concurso"]).values.flatten())
                    y.append(df_binario.iloc[i + 1].drop(columns=["Concurso"]).values)

                X = np.array(X)
                y = np.array(y)

                # Treinar um classificador para cada dezena (simplificado)
                model = RandomForestClassifier(n_estimators=50, random_state=42)
                # Usar os dados mais recentes para predição
                last_window = df_binario.tail(window).drop(columns=["Concurso"]).values.flatten().reshape(1, -1)

                # Para fins de performance no Streamlit, vamos usar uma abordagem de probabilidade baseada em RF
                # Treinamos um modelo global para entender a probabilidade de cada dezena
                probs = []
                for i in range(60):
                    model.fit(X, y[:, i])
                    probs.append(model.predict_proba(last_window)[0][1])

                df_pred = pd.DataFrame({'Número': range(1, 61), 'Probabilidade': probs}).sort_values('Probabilidade',
                                                                                                     ascending=False)
                top_pred = df_pred.head(qtd_pred)

                st.success(f"Top {qtd_pred} dezenas com maior probabilidade segundo o modelo:")
                st.write(top_pred['Número'].tolist())

                fig_pred = px.bar(top_pred, x=top_pred['Número'].astype(str), y='Probabilidade', color='Probabilidade',
                                  color_continuous_scale='Reds')
                st.plotly_chart(fig_pred, use_container_width=True)

    with tabs[7]:  # Clustering
        st.subheader("🧩 Agrupamento (Clustering) de Padrões")
        tipo_cluster = st.radio("Agrupar por:", ["Concursos (Padrões de Sorteio)", "Dezenas (Afinidade)"])
        n_clusters = st.slider("Número de Clusters:", 2, 10, 4)

        if tipo_cluster == "Concursos (Padrões de Sorteio)":
            data_cluster = df_bin_f.drop(columns=["Concurso"])
            kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(data_cluster)
            df_bin_f['Cluster'] = kmeans.labels_

            st.write("**Distribuição dos Concursos nos Clusters:**")
            fig_c = px.scatter(df_bin_f, x="Concurso", y="Cluster", color=df_bin_f["Cluster"].astype(str),
                               title="Concursos por Cluster")
            st.plotly_chart(fig_c, use_container_width=True)
        else:
            data_cluster = df_bin_f.drop(columns=["Concurso"]).T
            kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(data_cluster)
            df_cluster_dez = pd.DataFrame({'Número': range(1, 61), 'Cluster': kmeans.labels_})

            st.write("**Dezenas Agrupadas por Afinidade:**")
            for i in range(n_clusters):
                st.write(f"**Cluster {i}:** {df_cluster_dez[df_cluster_dez['Cluster'] == i]['Número'].tolist()}")

    with tabs[8]:  # Analisador de Jogos
        st.subheader("🔍 Analisador de Jogos Personalizados")
        dezenas_user = st.multiselect("Selecione de 4 a 15 dezenas:", list(range(1, 61)))

        if len(dezenas_user) >= 4:
            if len(dezenas_user) <= 15:
                # Verificar ocorrências
                set_user = set(dezenas_user)
                ocorrencias = []
                for idx, row in df_bruto.iterrows():
                    sorteados = set(row[cols_b].values)
                    acertos = len(set_user.intersection(sorteados))
                    if acertos >= 4:
                        ocorrencias.append({
                            "Concurso": row["Concurso"],
                            "Data": row["Data Sorteio"] if "Data Sorteio" in row else "N/A",
                            "Acertos": acertos,
                            "Ganhadores 6": row["Ganhadores_Sena"] if "Ganhadores_Sena" in row else 0
                        })

                df_ocorr = pd.DataFrame(ocorrencias)

                c1, c2, c3 = st.columns(3)
                score_jogo = sum(
                    [df_ranking[df_ranking['Número'] == n]['Score'].values[0] for n in dezenas_user]) / len(
                    dezenas_user)
                c1.metric("Score Médio SIPP", f"{score_jogo:.2f}")
                c2.metric("Vezes que saíram juntos (4+)", len(df_ocorr))

                # Cálculo de probabilidade teórica (simplificado)
                prob = (len(df_ocorr) / len(df_bruto)) * 100 if len(df_bruto) > 0 else 0
                c3.metric("Frequência Histórica", f"{prob:.4f}%")

                if not df_ocorr.empty:
                    st.write("**Histórico de Premiações deste Grupo:**")
                    st.dataframe(df_ocorr.sort_values("Concurso", ascending=False), use_container_width=True)
                else:
                    st.info("Este grupo de dezenas nunca premiou com 4 ou mais acertos.")
            else:
                st.error("Selecione no máximo 15 dezenas.")
        else:
            st.info("Selecione pelo menos 4 dezenas para análise.")

    with tabs[9]:  # Gerador
        st.subheader("🎲 Gerador Inteligente de Dezenas")
        c1, c2, c3 = st.columns(3)
        qtd_gerar = c1.number_input("Quantidade de números:", 6, 15, 6)
        n_pares = c2.slider("Quantidade de Pares:", 0, qtd_gerar, qtd_gerar // 2)
        soma_range = c3.slider("Faixa de Soma:", 21, 345, (150, 250))

        if st.button("Gerar Jogo"):
            # Lógica de geração baseada em scores e restrições
            pool_quentes = df_ranking.head(30)['Número'].tolist()

            tentativas = 0
            while tentativas < 1000:
                jogo = np.random.choice(range(1, 61), qtd_gerar, replace=False)
                pares = sum(1 for n in jogo if n % 2 == 0)
                soma = sum(jogo)

                if pares == n_pares and soma_range[0] <= soma <= soma_range[1]:
                    st.success(f"Jogo Gerado: {sorted(jogo)}")
                    st.write(f"Soma: {soma} | Pares: {pares} | Ímpares: {qtd_gerar - pares}")
                    break
                tentativas += 1
            if tentativas == 1000:
                st.warning("Não foi possível gerar um jogo com as restrições exatas. Tente ampliar a faixa de soma.")

    with tabs[10]:  # Governança
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
            st.write("- **Uso Ético:** Estudo estatístico. Não garante ganhos.")
            st.write("- **Transparência:** Algoritmos: SIPP, Apriori, FP-Growth, Random Forest, KMeans.")

    with tabs[11]:  # CRISP-DM
        st.subheader("📖 Metodologia CRISP-DM na Mineração")
        st.markdown("""
        <div class='crisp-box'>
        <b>1. Business Understanding (Entendimento do Negócio):</b> O objetivo é identificar padrões estatísticos e de associação nos sorteios da Mega Sena para auxiliar na tomada de decisão baseada em dados.
        </div>
        <div class='crisp-box'>
        <b>2. Data Understanding (Entendimento dos Dados):</b> Coleta automática via API da Caixa, análise de integridade e exploração visual de tendências (paridade, somas).
        </div>
        <div class='crisp-box'>
        <b>3. Data Preparation (Preparação dos Dados):</b> Transformação dos sorteios em uma matriz binária (One-Hot Encoding) e aplicação de pesos temporais (Linear/Exponencial).
        </div>
        <div class='crisp-box'>
        <b>4. Modeling (Modelagem):</b> Aplicação de algoritmos de Associação (Apriori/FP-Growth), Agrupamento (K-Means) e Predição (Random Forest).
        </div>
        <div class='crisp-box'>
        <b>5. Evaluation (Avaliação):</b> Uso de métricas como Suporte, Confiança, Lift e Probabilidades de Predição para validar os padrões encontrados.
        </div>
        <div class='crisp-box'>
        <b>6. Deployment (Implantação):</b> Disponibilização do conhecimento através desta interface interativa em Streamlit.
        </div>
        """, unsafe_allow_html=True)

st.markdown(f'<div class="footer-text notranslate">© {NOME_SISTEMA} {VERSAO} | 2026 | By: {PROGRAMADOR}</div>',
            unsafe_allow_html=True)

# 🤖 Predição ML
with tabs2[3]:  # Predição ML
    st.subheader("🤖 Predição com Machine Learning (Projeto consome muito processamnto Desativado)")
    qtd_pred = st.slider("Quantidade de dezenas para prever:", 10, 20, 15)
    '''
    if st.button("Executar Predição"):
        with st.spinner("Treinando modelo..."):
            # Preparar dados para ML (Prever se o número sai no próximo concurso baseado no histórico)
            X = []
            y = []
            window = 10
            for i in range(window, len(df_binario) - 1):
                X.append(df_binario.iloc[i - window:i].drop(columns=["Concurso"]).values.flatten())
                y.append(df_binario.iloc[i + 1].drop(columns=["Concurso"]).values)

            X = np.array(X)
            y = np.array(y)

            # Treinar um classificador para cada dezena (simplificado)
            model = RandomForestClassifier(n_estimators=50, random_state=42)
            # Usar os dados mais recentes para predição
            last_window = df_binario.tail(window).drop(columns=["Concurso"]).values.flatten().reshape(1, -1)

            # Para fins de performance no Streamlit, vamos usar uma abordagem de probabilidade baseada em RF
            # Treinamos um modelo global para entender a probabilidade de cada dezena
            probs = []
            for i in range(60):
                model.fit(X, y[:, i])
                probs.append(model.predict_proba(last_window)[0][1])

            df_pred = pd.DataFrame({'Número': range(1, 61), 'Probabilidade': probs}).sort_values('Probabilidade',
                                                                                                 ascending=False)
            top_pred = df_pred.head(qtd_pred)

            st.success(f"Top {qtd_pred} dezenas com maior probabilidade segundo o modelo:")
            st.write(top_pred['Número'].tolist())

            fig_pred = px.bar(top_pred, x=top_pred['Número'].astype(str), y='Probabilidade', color='Probabilidade',
                              color_continuous_scale='Reds')
            st.plotly_chart(fig_pred, use_container_width=True)
        '''
import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
import plotly.express as px
import plotly.graph_objects as go
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import datetime

# --- 1. CONFIGURAÇÕES TÉCNICAS E ESTÉTICAS ---
ARQUIVO_CAIXA = "resultadoJogoMegaSena.xlsx"
CAIXA_URL = "https://servicebus2.caixa.gov.br/portaldeloterias/api/resultados/download?modalidade=Mega-Sena"
COR_QUENTE = "#FF4B4B"
COR_FRIO = "#007BFF"
NOME_SISTEMA = "Mega Sena"
VERSAO = "SIPP v2.0.20260105"
PROGRAMADOR = "PRINCE, K.B"
LINK_PESSOAL = "https://klaytonprincems.github.io/site/" # Placeholder para o link original
cols_b = ['Bola1', 'Bola2', 'Bola3', 'Bola4', 'Bola5', 'Bola6']
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
    .crisp-box {{background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid {COR_FRIO}; margin-bottom: 10px;}}
    </style>
    """, unsafe_allow_html=True)


# --- 2. MOTOR DE DADOS ---

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
@st.cache_data
def processar_base_par_soma():
    c1, c2 = st.columns(2)

    paridade = df_bruto_f[cols_b].apply(lambda x: sum(1 for n in x if n % 2 == 0), axis=1)
    df_par = pd.DataFrame({'Concurso': df_bruto_f["Concurso"], 'Pares': paridade, 'Ímpares': 6 - paridade, 'Soma': df_bruto_f[cols_b].sum(axis=1)})
    return df_par


# --- 3. INTERFACE E LÓGICA ---
st.markdown(f"<h1 class='main-title'>🍀 {NOME_SISTEMA}</h1>", unsafe_allow_html=True)

df_bruto, df_binario = processar_base_completa()

if df_bruto is not None:
    # --- SIDEBAR: CONFIGURAÇÃO COMPLETA ---
    st.sidebar.header("⚙️ Mineração")
    min_c, max_c = int(df_bruto["Concurso"].min()), int(df_bruto["Concurso"].max())

    # Atalhos rápidos
    st.sidebar.write("**Intervalo de jogos**")
    c1, c2 = st.sidebar.columns(2)
    if c1.button("100"): st.session_state.ini, st.session_state.fim = max_c - 100, max_c
    if c2.button("Total"): st.session_state.ini, st.session_state.fim = min_c, max_c

    # Slider e Input para Precisão 100%
    val_ini = st.session_state.get('ini', max_c - 500)
    val_fim = st.session_state.get('fim', max_c)
    range_c = st.sidebar.slider("Janela Temporal:", min_c, max_c, (val_ini, val_fim))

    col_i, col_f = st.sidebar.columns(2)
    c_ini = col_i.number_input("Primeiro:", min_c, max_c, range_c[0])
    c_fim = col_f.number_input("Ultimo:", min_c, max_c, range_c[1])

    st.sidebar.write("**Def Fogo e Gelo**")
    qtd_top = st.sidebar.slider("Total:", 4, 30, 15)
    metodo = st.sidebar.selectbox("Peso de Recência:", ["Linear", "Exponencial"])

    if st.sidebar.button("🔄 ATUALIZAR DB"):
        if baixar_dados(): st.cache_data.clear(); st.rerun()

    # --- PROCESSAMENTO DOS DADOS FILTRADOS ---
    df_bruto_f = df_bruto[(df_bruto["Concurso"] >= c_ini) & (df_bruto["Concurso"] <= c_fim)].copy()
    df_bin_f = df_binario[(df_binario["Concurso"].astype(int) >= c_ini) & (df_binario["Concurso"].astype(int) <= c_fim)].copy()

    # Cálculo do Score SIPP
    weights = np.exp(np.linspace(0, 3, len(df_bin_f))) if metodo == "Exponencial" else np.linspace(0.1, 1.0,
                                                                                                   len(df_bin_f))
    scores = df_bin_f.drop(columns=["Concurso"]).astype(float).multiply(weights, axis=0).sum()
    df_ranking = pd.DataFrame({'Número': scores.index.astype(int), 'Score': scores.values}).sort_values('Score', ascending=False)


    # --- TABS ---
    tabs1 = st.tabs(["📈🪙 Paridade", "📈➕ Soma",  "🚀 Fogo e Gelo", "🟦 Quadrantes"])
    # 📈🪙 Paridade
    with tabs1[0]:  # Tendências
        st.subheader("📈 Tendências")
        df_par = processar_base_par_soma()
        st.write("**Distribuição Par vs Ímpar**")
        st.bar_chart(df_par.set_index('Concurso')[['Pares', 'Ímpares']])
        st.dataframe(df_par.iloc[::-1], use_container_width=True, hide_index=True)
    # 📈➕ Soma
    with tabs1[1]:  # Tendências
        st.subheader("📈 Tendências")
        df_par = processar_base_par_soma()
        st.write("**Evolução da Soma das Dezenas**")
        st.line_chart(df_par.set_index('Concurso')['Soma'])
        st.dataframe(df_par.iloc[::-1], use_container_width=True, hide_index=True)
    # 🚀fogo e Gelo
    with tabs1[2]:  # Fogo e Gelo
        st.subheader("🚀 Fogo e Gelo")
        top = df_ranking.head(qtd_top)
        bottom = df_ranking.tail(qtd_top)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=top['Número'].astype(str), y=top['Score'], name='Quentes', marker_color=COR_QUENTE))
        fig.add_trace(go.Bar(x=bottom['Número'].astype(str), y=bottom['Score'], name='Frios', marker_color=COR_FRIO))
        fig.update_layout(title="Extremos de Tendência", barmode='group')
        st.plotly_chart(fig, use_container_width=True)
        st.write("**Peso de possibilidade total**")
        st.dataframe(df_ranking, use_container_width=True, hide_index=True)
    with tabs1[3]:
        st.subheader("🟦 Distribuição por Quadrantes")
        st.markdown("""
        O volante é dividido em 4 áreas. Jogos equilibrados costumam ter dezenas 
        distribuídas em pelo menos 3 quadrantes.
        """)

        df_q = processar_quadrantes(df_bruto_f)

        # Gráfico de Barras Empilhadas
        fig_q = px.bar(df_q.reset_index(), x="Concurso", y=["Q1", "Q2", "Q3", "Q4"],
                       title="Equilíbrio de Quadrantes por Concurso",
                       color_discrete_sequence=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])

        fig_q.update_layout(yaxis_title="Qtd de Dezenas", barmode='stack')
        st.plotly_chart(fig_q, use_container_width=True)

        # Médias de ocupação
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Média Q1", f"{df_q['Q1'].mean():.2f}")
        c2.metric("Média Q2", f"{df_q['Q2'].mean():.2f}")
        c3.metric("Média Q3", f"{df_q['Q3'].mean():.2f}")
        c4.metric("Média Q4", f"{df_q['Q4'].mean():.2f}")

        st.write("**Dados de Ocupação por Concurso**")
        st.dataframe(df_q.iloc[::-1], use_container_width=True)

    st.divider()
    tabs2 = st.tabs(["🤖 Predição ML","🎯 Percentis", "🔗 Associação"])
    # 🤖 Predição ML
    with tabs2[0]:  # Predição ML
        st.subheader("🤖 Predição com Machine Learning (Projeto consome muito processamnto Desativado)")
        qtd_pred = st.slider("Quantidade de dezenas para prever:", 10, 20, 15)
        '''
        if st.button("Executar Predição"):
            with st.spinner("Treinando modelo..."):
                # Preparar dados para ML (Prever se o número sai no próximo concurso baseado no histórico)
                X = []
                y = []
                window = 10
                for i in range(window, len(df_binario) - 1):
                    X.append(df_binario.iloc[i - window:i].drop(columns=["Concurso"]).values.flatten())
                    y.append(df_binario.iloc[i + 1].drop(columns=["Concurso"]).values)

                X = np.array(X)
                y = np.array(y)

                # Treinar um classificador para cada dezena (simplificado)
                model = RandomForestClassifier(n_estimators=50, random_state=42)
                # Usar os dados mais recentes para predição
                last_window = df_binario.tail(window).drop(columns=["Concurso"]).values.flatten().reshape(1, -1)

                # Para fins de performance no Streamlit, vamos usar uma abordagem de probabilidade baseada em RF
                # Treinamos um modelo global para entender a probabilidade de cada dezena
                probs = []
                for i in range(60):
                    model.fit(X, y[:, i])
                    probs.append(model.predict_proba(last_window)[0][1])

                df_pred = pd.DataFrame({'Número': range(1, 61), 'Probabilidade': probs}).sort_values('Probabilidade',
                                                                                                     ascending=False)
                top_pred = df_pred.head(qtd_pred)

                st.success(f"Top {qtd_pred} dezenas com maior probabilidade segundo o modelo:")
                st.write(top_pred['Número'].tolist())

                fig_pred = px.bar(top_pred, x=top_pred['Número'].astype(str), y='Probabilidade', color='Probabilidade',
                                  color_continuous_scale='Reds')
                st.plotly_chart(fig_pred, use_container_width=True)
            '''
    # 🎯 Percentis
    with tabs2[1]:  # Percentis
        st.subheader("🎯 Análise de Percentis")
        p_alta = df_ranking['Score'].quantile(0.80)
        p_baixa = df_ranking['Score'].quantile(0.20)
        c1, c2 = st.columns(2)
        c1.metric("Corte Percentil 80 (Quentes)", f"{p_alta:.2f}")
        c2.metric("Corte Percentil 20 (Frios)", f"{p_baixa:.2f}")
        fig_p = px.ecdf(df_ranking, x="Score", title="Curva de Probabilidade Acumulada")
        st.plotly_chart(fig_p, use_container_width=True)
    # 🔗 Associação
    with tabs2[2]:  # Associação (Apriori / FP-Growth)
        st.subheader("🔗 Mineração de Regras de Associação")
        col_alg, col_sup, col_conf = st.columns(3)
        alg_assoc = col_alg.selectbox("Algoritmo:", ["FP-Growth", "Apriori"])
        min_sup = col_sup.slider("Suporte Mínimo:", 0.001, 0.05, 0.01, format="%.3f")
        min_conf = col_conf.slider("Confiança Mínima:", 0.1, 1.0, 0.10)

        df_assoc_input = df_bin_f.drop(columns=["Concurso"]).astype(bool)

        try:
            if alg_assoc == "Apriori":
                frequent_itemsets = apriori(df_assoc_input, min_support=min_sup, use_colnames=True)
            else:
                frequent_itemsets = fpgrowth(df_assoc_input, min_support=min_sup, use_colnames=True)

            if not frequent_itemsets.empty:
                rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_conf)
                if not rules.empty:
                    rules["antecedents"] = rules["antecedents"].apply(lambda x: list(x))
                    rules["consequents"] = rules["consequents"].apply(lambda x: list(x))
                    st.write(f"**Regras Encontradas ({len(rules)}):**")
                    st.dataframe(
                        rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].sort_values('lift',
                                                                                                           ascending=False),
                        use_container_width=True)
                else:
                    st.warning("Nenhuma regra encontrada com a confiança mínima definida.")
            else:
                st.warning("Nenhum conjunto frequente encontrado com o suporte mínimo definido.")
        except Exception as e:
            st.error(f"Erro no processamento de associação: {e}")

    st.divider()
    tabs3 = st.tabs(["🎲 Gerador", "🧩 Clustering", "🔍 Analisador"])
    # 🎲 Gerador
    with tabs3[0]:  # Gerador
        st.subheader("🎲 Gerador Inteligente de Dezenas")
        c1, c2, c3 = st.columns(3)
        qtd_gerar = c1.number_input("Quantidade de números:", 6, 15, 6)
        n_pares = c2.slider("Quantidade de Pares:", 0, qtd_gerar, qtd_gerar // 2)
        soma_range = c3.slider("Faixa de Soma:", 21, 345, (150, 250))

        if st.button("Gerar Jogo"):
            # Lógica de geração baseada em scores e restrições
            pool_quentes = df_ranking.head(30)['Número'].tolist()

            tentativas = 0
            while tentativas < 1000:
                jogo = np.random.choice(range(1, 61), qtd_gerar, replace=False)
                pares = sum(1 for n in jogo if n % 2 == 0)
                soma = sum(jogo)

                if pares == n_pares and soma_range[0] <= soma <= soma_range[1]:
                    st.success(f"Jogo Gerado: {sorted(jogo)}")
                    st.write(f"Soma: {soma} | Pares: {pares} | Ímpares: {qtd_gerar - pares}")
                    break
                tentativas += 1
            if tentativas == 1000:
                st.warning("Não foi possível gerar um jogo com as restrições exatas. Tente ampliar a faixa de soma.")
    #🧩 Clustering
    with tabs3[1]:  # Clustering
        st.subheader("🧩 Agrupamento (Clustering) de Padrões")
        tipo_cluster = st.radio("Agrupar por:", ["Concursos (Padrões de Sorteio)", "Dezenas (Afinidade)"])
        n_clusters = st.slider("Número de Clusters:", 2, 10, 4)

        if tipo_cluster == "Concursos (Padrões de Sorteio)":
            data_cluster = df_bin_f.drop(columns=["Concurso"])
            kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(data_cluster)
            df_bin_f['Cluster'] = kmeans.labels_

            st.write("**Distribuição dos Concursos nos Clusters:**")
            fig_c = px.scatter(df_bin_f, x="Concurso", y="Cluster", color=df_bin_f["Cluster"].astype(str),
                               title="Concursos por Cluster")
            st.plotly_chart(fig_c, use_container_width=True)
        else:
            data_cluster = df_bin_f.drop(columns=["Concurso"]).T
            kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(data_cluster)
            df_cluster_dez = pd.DataFrame({'Número': range(1, 61), 'Cluster': kmeans.labels_})

            st.write("**Dezenas Agrupadas por Afinidade:**")
            for i in range(n_clusters):
                st.write(f"**Cluster {i}:** {df_cluster_dez[df_cluster_dez['Cluster'] == i]['Número'].tolist()}")
    # 🔍 Analisador
    with tabs3[2]:  # Analisador de Jogos
        st.subheader("🔍 Analisador de Jogos Personalizados")
        dezenas_user = st.multiselect("Selecione de 4 a 15 DEZENAS :", list(range(1, 61)))

        if len(dezenas_user) >= 4:
            if len(dezenas_user) <= 15:
                # Verificar ocorrências
                set_user = set(dezenas_user)
                ocorrencias = []
                for idx, row in df_bruto.iterrows():
                    sorteados = set(row[cols_b].values)
                    acertos = len(set_user.intersection(sorteados))
                    if acertos >= 4:
                        ocorrencias.append({
                            "Concurso": row["Concurso"],
                            "Data": row["Data Sorteio"] if "Data Sorteio" in row else "N/A",
                            "Acertos": acertos,
                            "Ganhadores 6": row["Ganhadores_Sena"] if "Ganhadores_Sena" in row else 0
                        })

                df_ocorr = pd.DataFrame(ocorrencias)

                c1, c2, c3 = st.columns(3)
                score_jogo = sum(
                    [df_ranking[df_ranking['Número'] == n]['Score'].values[0] for n in dezenas_user]) / len(
                    dezenas_user)
                c1.metric("Score Médio SIPP", f"{score_jogo:.2f}")
                c2.metric("Vezes que saíram juntos (4+)", len(df_ocorr))

                # Cálculo de probabilidade teórica (simplificado)
                prob = (len(df_ocorr) / len(df_bruto)) * 100 if len(df_bruto) > 0 else 0
                c3.metric("Frequência Histórica", f"{prob:.4f}%")

                if not df_ocorr.empty:
                    st.write("**Histórico de Premiações deste Grupo:**")
                    st.dataframe(df_ocorr.sort_values("Concurso", ascending=False), use_container_width=True)
                else:
                    st.info("Este grupo de dezenas nunca premiou com 4 ou mais acertos.")
            else:
                st.error("Selecione no máximo 15 dezenas.")
        else:
            st.info("Selecione pelo menos 4 dezenas para análise.")

    st.divider()
    tabelaBasica = st.tabs(["📊 Dados Brutos", "🔢 Dados Binário", "🛡️ Governança", "📖 Metodologias Utilizadas"])
    with tabelaBasica[0]:  # Dados
        st.subheader("📋 Histórico Oficial de Concursos")
        st.dataframe(df_bruto_f.iloc[::-1], use_container_width=True, hide_index=True)
    with tabelaBasica[1]:  # Binário
        st.subheader("🔢 Matriz de Identificação (0 e 1)")
        st.dataframe(df_bin_f.iloc[::-1], use_container_width=True, hide_index=True)
    with tabelaBasica[2]:  # Governança
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
            st.write("- **Uso Ético:** Estudo estatístico. Não garante ganhos.")
            st.write("- **Transparência:** Algoritmos: SIPP, Apriori, FP-Growth, Random Forest, KMeans.")
    with tabelaBasica[3]:
        st.subheader("📖 Metodologia SIPP e CRISP-DM na Mineração")
        st.markdown("""
        <div class='crisp-box'>
        <b>1. Business Understanding (Entendimento do Negócio):</b> 
        <p>O objetivo deste projeto é identificar padrões estatísticos e de associação nos sorteios da Mega Sena, este 
        trabalho não têm pretenção de realizar previsões de jogos ou auxiliar na tomada de decisão.</p>

        <p>Objetivo principal e baseado na analise de dadosdados, e estudos probabilisticos, para tal foi usado uma mescla
         das metodologias SIPP e CRISP-DM</p>

        <p>A metodologia SIPP(acrônimo para Seleção, Integração, Processamento e Predição) é um framework estruturado
        utilizado na mineração de dados(Data Mining) para transformar dados brutos em conhecimento útil. Embora o modelo 
        CRISP - DM seja o padrão mais conhecido da indústria, o SIPP é frequentemente aplicado em contextos acadêmicos e 
        específicos de análise técnica por ser o mais direto nas etapas operacionais de manipulação de dados</p></div>

        <div class='crisp-box'>
        <b>2. Data Understanding (Entendimento dos Dados):</b> 
        O sistema faz a coleta automática de dados direto da Caixa Economica Federal, realiza análise de integridade e 
        faz a exploração visual, apresentando tendências (paridade, somas).</div>

        <div class='crisp-box'>
        <b>3. Data Preparation (Preparação dos Dados):</b> 
        É realizada uma transformação dos dados dos sorteios em uma matriz binária (One-Hot Encoding) e aplicação de 
        pesos temporais (Linear/Exponencial), grande parte dos processos e algoritimos são aplicados na base de dados binaria.</div>

        <div class='crisp-box'>
        <b>4. Modeling (Modelagem):</b> 
        E realizado uma aplicação de algoritmos de Associação (Apriori/FP-Growth), Agrupamento (K-Means) e Predição (Random Forest).</div>

        <div class='crisp-box'>
        <b>5. Evaluation (Avaliação):</b> 
        É realizado uso de métricas como Suporte, Confiança, Lift e Probabilidades para Predição afim de validar os padrões encontrados.</div>

        <div class='crisp-box'>
        <b>6. Deployment (Implantação):</b> Disponibilização do conhecimento através de uma interface interativa utilizando o em Streamlit e Python.</div>
        """, unsafe_allow_html=True)
    st.divider()

st.markdown(f'<div class="footer-text notranslate">© {NOME_SISTEMA} {VERSAO} | 2026 | By: {PROGRAMADOR}</div>',
            unsafe_allow_html=True)

with tabs4[4]:
    st.subheader("🤖 Predição Evolutiva via Rede Neural (MLP)")
    c_min_ia, c_max_ia = int(df_bin_f["Concurso"].min()), int(df_bin_f["Concurso"].max())
    col_ia1, col_ia2, col_ia3 = st.columns(3)
    ini_ia = col_ia1.number_input("Treinar a partir do:", c_min_ia, c_max_ia - 1, max(c_min_ia, c_max_ia - 500),
                                  key="ia_ini")
    fim_ia = col_ia2.number_input("Até o concurso:", ini_ia + 1, c_max_ia, c_max_ia, key="ia_fim")
    qtd_sugerida = col_ia3.slider("Qtd de dezenas no Pool:", 6, 30, 20, key="ia_pool")
    proximo_alvo = fim_ia + 1
    st.info(f"🎯 **Objetivo:** Prever o comportamento do concurso **{proximo_alvo}**.")
    if st.button("🚀 Iniciar Treinamento e Predição"):
        with st.spinner("A rede neural está processando..."):
            probs = realizar_predicao_mlp_custom(df_bin_f, ini_ia, fim_ia)
            if probs is not None:
                df_prev = pd.DataFrame({'Dezena': range(1, 61), 'Probabilidade': probs}).sort_values('Probabilidade',
                                                                                                     ascending=False)
                pool_ia = df_prev.head(qtd_sugerida)
                dezenas_sugeridas = sorted(pool_ia['Dezena'].tolist())
                st.code(", ".join(f"{d:02d}" for d in dezenas_sugeridas), language='text')
                concurso_real = df_bruto[df_bruto["Concurso"] == proximo_alvo]
                if not concurso_real.empty:
                    reais = [int(n) for n in concurso_real[cols_b].values[0]]
                    acertos = set(dezenas_sugeridas).intersection(set(reais))
                    st.metric("Acertos no Pool vs Resultado Real", len(acertos))
            else:
                st.error("Dados insuficientes para este intervalo de treino.")
with tabs4[4]:  # A nova aba para o Random Forest
    st.subheader("🌳 Predição via Floresta Aleatória (Random Forest)")
    st.markdown(
        "O Random Forest treina múltiplas árvores de decisão para encontrar as dezenas com maior probabilidade de aparecer, sendo um modelo robusto e eficiente.")

    # Widgets de controle para o usuário (com chaves únicas)
    c_min_rf, c_max_rf = int(df_bin_f["Concurso"].min()), int(df_bin_f["Concurso"].max())
    col_rf1, col_rf2, col_rf3 = st.columns(3)
    ini_rf = col_rf1.number_input("Treinar a partir do:", c_min_rf, c_max_rf - 1, max(c_min_rf, c_max_rf - 500),
                                  key="rf_ini")
    fim_rf = col_rf2.number_input("Até o concurso:", ini_rf + 1, c_max_rf, c_max_rf, key="rf_fim")
    qtd_sugerida_rf = col_rf3.slider("Qtd de dezenas no Pool:", 6, 30, 20, key="rf_pool")

    proximo_alvo_rf = fim_rf + 1
    st.info(f"🎯 **Objetivo:** Prever o comportamento do concurso **{proximo_alvo_rf}** usando Random Forest.")

    if st.button("🌲 Iniciar Treinamento e Predição RF", key="rf_button"):
        with st.spinner("A floresta aleatória está crescendo e analisando os dados..."):
            # Chama a nova função de predição
            probs_rf = realizar_predicao_rf_custom(df_bin_f, ini_rf, fim_rf)

            if probs_rf is not None:
                # Cria o DataFrame de resultados
                df_prev_rf = pd.DataFrame({'Dezena': range(1, 61), 'Probabilidade': probs_rf}).sort_values(
                    'Probabilidade', ascending=False)

                # Seleciona o pool de dezenas sugeridas
                pool_rf = df_prev_rf.head(qtd_sugerida_rf)
                dezenas_sugeridas_rf = sorted(pool_rf['Dezena'].tolist())

                st.write(f"### 🌳 Pool de {qtd_sugerida_rf} Dezenas Sugeridas pelo Random Forest")
                st.code(", ".join(f"{d:02d}" for d in dezenas_sugeridas_rf), language='text')

                # Backtesting: Verifica com o resultado real, se disponível
                concurso_real_rf = df_bruto[df_bruto["Concurso"] == proximo_alvo_rf]
                if not concurso_real_rf.empty:
                    st.divider()
                    st.write(f"### ⚖️ Verificação com o Resultado Real do Concurso {proximo_alvo_rf}")
                    reais_rf = [int(n) for n in concurso_real_rf[cols_b].values[0]]
                    acertos_rf = set(dezenas_sugeridas_rf).intersection(set(reais_rf))

                    c_ver1, c_ver2 = st.columns(2)
                    texto_reais = ", ".join(map(str, sorted(reais_rf)))
                    c_ver1.markdown(f"**Números Sorteados:**\n`{texto_reais}`")
                    c_ver1.metric("Acertos do Random Forest no Pool", len(acertos_rf))

                    if len(acertos_rf) >= 4:
                        st.balloons()
                        c_ver2.success(
                            f"🔥 Excelente! O pool de {qtd_sugerida_rf} dezenas capturou {len(acertos_rf)} acertos.")
                    else:
                        c_ver2.warning(f"O pool capturou {len(acertos_rf)} acertos.")

                # Gráfico de importância das dezenas
                fig_rf = px.bar(pool_rf, x='Dezena', y='Probabilidade',
                                title=f"Força Estatística das Top {qtd_sugerida_rf} Dezenas (Random Forest)",
                                labels={'Dezena': 'Número da Bola', 'Probabilidade': 'Peso do Modelo'})
                fig_rf.update_xaxes(type='category')
                st.plotly_chart(fig_rf, use_container_width=True)
            else:
                st.error(
                    "Dados insuficientes para este intervalo de treino. O Random Forest precisa de mais exemplos.")