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
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neural_network import MLPClassifier
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
@st.cache_data
def processar_quadrantes(df_f):
    def identificar_quadrante(n):
        # Lógica de divisão do volante 6x10
        row = (n - 1) // 10  # Linha (0 a 5)
        col = (n - 1) % 10  # Coluna (0 a 9)
        if row < 3:
            return "Q1" if col < 5 else "Q2"
        else:
            return "Q3" if col < 5 else "Q4"

    quadrantes_contagem = []
    # Usamos o df filtrado (df_bruto_f) que contém as colunas Bola1...Bola6
    for _, row in df_f[['Bola1', 'Bola2', 'Bola3', 'Bola4', 'Bola5', 'Bola6']].iterrows():
        qs = [identificar_quadrante(n) for n in row]
        quadrantes_contagem.append({
            "Q1": qs.count("Q1"), "Q2": qs.count("Q2"),
            "Q3": qs.count("Q3"), "Q4": qs.count("Q4")
        })

    return pd.DataFrame(quadrantes_contagem, index=df_f["Concurso"])
@st.cache_data
def executar_agnes(df_bin_f, n_clusters=4):
    # 1. Removemos o 'Concurso' para não enviesar o modelo
    data = df_bin_f.drop(columns=["Concurso"], errors='ignore').copy()

    # 2. RESOLUÇÃO DO ERRO: Converter nomes das colunas para String
    data.columns = data.columns.astype(str)

    # 3. Execução do modelo
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='euclidean',
        linkage='ward'
    )

    labels = model.fit_predict(data)
    return labels
@st.cache_data
def plot_dendrograma(df_bin_f):
    # O Dendrograma é a "árvore" que mostra como o AGNES juntou os grupos
    data = df_bin_f.drop(columns=["Concurso"]).head(50) # Limitamos para visualização
    fig = go.Figure(sch.dendrogram(sch.linkage(data, method='ward'), no_plot=True))
    # Aqui você usaria o Plotly para desenhar as linhas do dendrograma
    return fig
@st.cache_data
def plotar_dendrograma(df_bin_f, limite=50):
    # Pegamos apenas os últimos X concursos para visualização clara
    df_small = df_bin_f.tail(limite).copy()
    concursos = df_small['Concurso'].astype(str).values
    data = df_small.drop(columns=["Concurso"], errors='ignore')
    data.columns = data.columns.astype(str)

    # Calculando a matriz de ligação (Linkage Matrix)
    # O método 'ward' minimiza a variância dentro dos grupos
    linkage_matrix = sch.linkage(data, method='ward')

    # Criando a figura do Matplotlib
    fig, ax = plt.subplots(figsize=(10, 6))
    sch.dendrogram(
        linkage_matrix,
        labels=concursos,
        leaf_rotation=90,
        leaf_font_size=10,
        ax=ax
    )
    ax.set_title(f"Dendrograma: Genealogia dos Últimos {limite} Concursos")
    ax.set_xlabel("Número do Concurso")
    ax.set_ylabel("Distância Euclidiana (Dissimilaridade)")

    return fig
@st.cache_data
def plotar_dendrograma_range(df_bin_f, c_ini, c_fim):
    # Filtragem baseada no range escolhido pelo usuário
    df_range = df_bin_f[(df_bin_f["Concurso"] >= c_ini) & (df_bin_f["Concurso"] <= c_fim)].copy()

    if df_range.empty:
        return None

    concursos = df_range['Concurso'].astype(str).values
    data = df_range.drop(columns=["Concurso"], errors='ignore')
    data.columns = data.columns.astype(str)

    # Calculando a matriz de ligação
    linkage_matrix = sch.linkage(data, method='ward')

    # Criando a figura
    fig, ax = plt.subplots(figsize=(12, 6))
    sch.dendrogram(
        linkage_matrix,
        labels=concursos,
        leaf_rotation=90,
        leaf_font_size=9,
        ax=ax
    )
    ax.set_title(f"Dendrograma: De {c_ini} até {c_fim}")
    ax.set_ylabel("Distância (Dissimilaridade)")
    plt.tight_layout()

    return fig
@st.cache_data
def executar_diana(df_bin_f, n_clusters=4):
    from sklearn.cluster import BisectingKMeans

    # Preparação dos dados
    data = df_bin_f.drop(columns=["Concurso"], errors='ignore').copy()
    data.columns = data.columns.astype(str)

    # CORREÇÃO: O parâmetro correto é bisecting_strategy
    model = BisectingKMeans(
        n_clusters=n_clusters,
        random_state=42,
        bisecting_strategy='biggest_inertia'
    )

    labels = model.fit_predict(data)
    return labels
@st.cache_data
def executar_dbscan1(df_bin_f, eps=0.5, min_samples=3):
    # Preparação dos dados
    data = df_bin_f.drop(columns=["Concurso"], errors='ignore').copy()
    data.columns = data.columns.astype(str)

    # O DBSCAN é sensível à escala, mas como os dados são 0 e 1,
    # ele funciona bem com métricas de distância binária (como Jaccard ou Hamming)
    # Aqui usaremos a métrica 'euclidean' que é o padrão
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(data)

    return labels
@st.cache_data
def executar_dbscan(df_bin_f, eps=0.5, min_samples=3):
    data = df_bin_f.drop(columns=["Concurso"], errors='ignore').copy()
    data.columns = data.columns.astype(str)

    # Usar 'hamming' ou 'jaccard' pode ser mais eficiente para dezenas 0/1
    # Se continuar dando ruído, aumente o EPS significativamente (ex: 2.5)
    model = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    labels = model.fit_predict(data)

    return labels
@st.cache_data
def realizar_predicao_mlp_custom(df_bin_f, c_ini, c_fim):
    df_modelo = df_bin_f[(df_bin_f["Concurso"] >= c_ini) & (df_bin_f["Concurso"] <= c_fim)].copy()

    if len(df_modelo) < 20:
        return None

    data = df_modelo.drop(columns=["Concurso"], errors='ignore').values
    X = data[:-1]
    y = data[1:]

    # --- AJUSTES AVANÇADOS ---
    model = MLPClassifier(
        hidden_layer_sizes=(120, 80, 40),  # Arquitetura em pirâmide
        activation='relu',  # Função de ativação para redes profundas
        solver='adam',  # Otimizador estocástico
        alpha=0.001,  # Regularização L2 (Evita Overfitting)
        learning_rate='adaptive',  # Diminui a velocidade de aprendizado se estagnar
        max_iter=1000,  # Aumenta o limite de épocas
        early_stopping=True,  # Para o treino se a rede parar de aprender
        validation_fraction=0.1,  # Usa 10% dos dados para validar o treino
        random_state=42
    )

    model.fit(X, y)

    ultimo_concurso_conhecido = data[-1].reshape(1, -1)
    probabilidades = model.predict_proba(ultimo_concurso_conhecido)[0]

    return probabilidades
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
    val_ini = st.session_state.get('ini', max_c - 75)
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
    tabs1 = st.tabs(["","📈🪙 Paridade", "📈➕ Soma",  "🚀 Fogo e Gelo", "🟦 Quadrantes"])
    # Inicio
    with tabs1[0]:  # Tendências
        st.subheader("Escolha uma dos itens na tabela")
    # 📈🪙 Paridade
    with tabs1[1]:  # Tendências
        st.subheader("📈 Tendências")
        df_par = processar_base_par_soma()
        st.write("**Distribuição Par vs Ímpar**")
        st.bar_chart(df_par.set_index('Concurso')[['Pares', 'Ímpares']])
        st.dataframe(df_par.iloc[::-1], use_container_width=True, hide_index=True)
    # 📈➕ Soma
    with tabs1[2]:  # Tendências
        st.subheader("📈 Tendências")
        df_par = processar_base_par_soma()
        st.write("**Evolução da Soma das Dezenas**")
        st.line_chart(df_par.set_index('Concurso')['Soma'])
        st.dataframe(df_par.iloc[::-1], use_container_width=True, hide_index=True)
    # 🚀fogo e Gelo
    with tabs1[3]:  # Fogo e Gelo
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
    # 🟦 Quadrantes
    with tabs1[4]:
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
    tabs2 = st.tabs(["","🎯 Percentis", "🔗 Associação", "🔍 Analisador"])
    # Inicio
    with tabs2[0]:  # Tendências
        st.subheader("Escolha uma dos itens na tabela")
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
    # 🔍 Analisador
    with tabs2[3]:  # Analisador de Jogos
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
    tabs3 = st.tabs(["", "🧩 Clustering",  "🧩 Agnes", "🧩 Agnes x Diana", "🧩 Dendograma", "🧩 DBSCAN"])
    # Inicio
    with tabs3[0]:  # Tendências
        st.subheader("Escolha uma dos itens na tabela")
    #🧩 Clustering
    with tabs3[1]:  # Aba de Clustering
        st.subheader("🧩 Agrupamento (Clustering) de Padrões")
        tipo_cluster = st.radio("Agrupar por:", ["Concursos (Padrões de Sorteio)", "Dezenas (Afinidade)"])
        n_clusters = st.slider("Número de Clusters:", 2, 10, 4)

        if tipo_cluster == "Concursos (Padrões de Sorteio)":
            # Pegamos os dados e removemos a coluna de identificação
            data_cluster = df_bin_f.drop(columns=["Concurso"], errors='ignore').copy()

            # --- CORREÇÃO DO ERRO AQUI ---
            data_cluster.columns = data_cluster.columns.astype(str)
            # -----------------------------

            kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(data_cluster)
            df_bin_f['Cluster'] = kmeans.labels_

            st.write("**Distribuição dos Concursos nos Clusters:**")
            fig_c = px.scatter(df_bin_f, x="Concurso", y="Cluster", color=df_bin_f["Cluster"].astype(str),
                               title="Concursos por Cluster")
            st.plotly_chart(fig_c, use_container_width=True)

        else:
            # Agrupamento por Dezenas (Matriz Transposta)
            data_cluster = df_bin_f.drop(columns=["Concurso"], errors='ignore').T

            # --- CORREÇÃO DO ERRO AQUI ---
            data_cluster.columns = data_cluster.columns.astype(str)
            # -----------------------------

            kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(data_cluster)
            df_cluster_dez = pd.DataFrame({'Número': range(1, 61), 'Cluster': kmeans.labels_})

            st.write("**Dezenas Agrupadas por Afinidade:**")
            for i in range(n_clusters):
                st.write(f"**Cluster {i}:** {df_cluster_dez[df_cluster_dez['Cluster'] == i]['Número'].tolist()}")
    # 🧩 Agnes
    with tabs3[2]:  # Aba Clustering
        st.subheader("🧩 Agrupamento Hierárquico (AGNES)")

        # Parâmetros para o usuário
        metrica = st.selectbox("Métrica de Distância:", ["euclidean", "manhattan", "cosine"])
        n_cl = st.slider("Quantidade de Grupos Hierárquicos:", 2, 8, 4)

        if st.button("Executar Análise Hierárquica"):
            labels = executar_agnes(df_bin_f, n_cl)
            df_bin_f['Agnes_Cluster'] = labels

            st.success(f"Análise concluída usando métrica {metrica}")

            # Visualização: Quais dezenas compõem cada grupo do AGNES
            fig_agnes = px.scatter(df_bin_f, x="Concurso", y="Agnes_Cluster",
                                   color=df_bin_f["Agnes_Cluster"].astype(str),
                                   title="Concursos Agrupados por Similaridade Hierárquica")
            st.plotly_chart(fig_agnes, use_container_width=True)
    with tabs3[3]:  # Supondo que seja a aba de Clustering
        st.subheader("🧬 Clustering Hierárquico: AGNES vs DIANA")

        col_metodo, col_n = st.columns(2)
        metodo_h = col_metodo.selectbox("Escolha a Direção:", ["AGNES (Agrupar)", "DIANA (Dividir)"])
        n_h = col_n.slider("Número de Grupos:", 2, 10, 4, key="slider_h")

        if st.button("Executar Mineração Hierárquica"):
            if metodo_h == "AGNES (Agrupar)":
                labels = executar_agnes(df_bin_f, n_h)
                desc = "Bottom-Up: Identificou como os concursos se unem."
            else:
                labels = executar_diana(df_bin_f, n_h)
                desc = "Top-Down: Identificou como as tendências se separam."

            df_bin_f['Cluster_H'] = labels
            st.success(desc)

            # Gráfico comparativo
            fig_h = px.scatter(df_bin_f, x="Concurso", y="Cluster_H",
                               color=labels.astype(str),
                               title=f"Resultado via {metodo_h}")
            st.plotly_chart(fig_h, use_container_width=True)
    with tabs3[4]:  # Aba do Dendrograma / Agnes
        st.subheader("🌳 Análise Genealógica Customizada")
        st.markdown("Selecione o intervalo de concursos para visualizar a árvore de similaridade.")

        # Seletores de Range específicos para o Dendrograma
        col_r1, col_r2 = st.columns(2)
        min_hist = int(df_bin_f["Concurso"].min())
        max_hist = int(df_bin_f["Concurso"].max())

        # Sugerimos os últimos 50 por padrão, mas deixamos mudar
        c_start = col_r1.number_input("Concurso Inicial:", min_hist, max_hist, max_hist - 50)
        c_end = col_r2.number_input("Concurso Final:", min_hist, max_hist, max_hist)

        if c_start >= c_end:
            st.error("O concurso inicial deve ser menor que o final.")
        else:
            # Verificamos se o range não é grande demais (Dendrogramas com >150 itens ficam ilegíveis)
            if (c_end - c_start) > 150:
                st.warning("⚠️ Range muito grande! A visualização pode ficar poluída. O ideal é até 150 concursos.")

            fig_d = plotar_dendrograma_range(df_bin_f, c_start, c_end)

            if fig_d:
                st.pyplot(fig_d)
            else:
                st.info("Nenhum dado encontrado para este intervalo.")
    # 🧩 DBSCAN
    with tabs3[5]:  # Aba de Clustering / DBSCAN
        st.subheader("📡 Detector de Anomalias e Padrões (DBSCAN)")

        # Explicação técnica para o usuário
        st.markdown("""
        O DBSCAN separa o que é **Padrão** (concentração de sorteios parecidos) do que é **Ruído** (sorteios atípicos).
        """)

        # Controles de Sensibilidade
        c1, c2 = st.columns(2)
        val_eps = c1.slider("Sensibilidade (Epsilon):", 0.5, 3.0, 1.2,
                            help="Menor valor = Mais rigoroso (gera mais ruído).")
        val_min = c2.slider("Mínimo de Vizinhos:", 2, 10, 3,
                            help="Quantos concursos parecidos precisam existir para formar um padrão.")

        # Execução do Algoritmo
        labels_db = executar_dbscan(df_bin_f, eps=val_eps, min_samples=val_min)
        df_bin_f['Cluster_DBSCAN'] = labels_db

        # Seletor de Visualização
        modo_exibicao = st.radio(
            "Filtrar por integridade estatística:",
            ["Exibir Todos", "Apenas Padrões (Sinal)", "Apenas Anomalias (Ruído)"],
            horizontal=True
        )

        # Lógica de Filtragem
        if modo_exibicao == "Apenas Padrões (Sinal)":
            df_filtrado_db = df_bin_f[df_bin_f['Cluster_DBSCAN'] != -1]
            cor_alerta = "green"
        elif modo_exibicao == "Apenas Anomalias (Ruído)":
            df_filtrado_db = df_bin_f[df_bin_f['Cluster_DBSCAN'] == -1]
            cor_alerta = "red"
        else:
            df_filtrado_db = df_bin_f
            cor_alerta = "blue"

        # Cruzamento com dados reais para exibição
        concursos_id = df_filtrado_db['Concurso'].tolist()
        df_final_db = df_bruto[df_bruto['Concurso'].isin(concursos_id)].copy()
        df_final_db = df_final_db.merge(df_bin_f[['Concurso', 'Cluster_DBSCAN']], on='Concurso')

        # Métricas de Resumo
        m1, m2, m3 = st.columns(3)
        total = len(df_bin_f)
        ruidos = list(labels_db).count(-1)
        m1.metric("Total Analisado", total)
        m2.metric("Concursos no Padrão", total - ruidos)
        m3.metric("Concursos no Ruído", ruidos, delta=f"{(ruidos / total) * 100:.1f}%", delta_color="inverse")

        # Exibição da Tabela
        st.write(f"### 📋 Listagem: {modo_exibicao}")
        st.dataframe(
            df_final_db[['Concurso', 'Cluster_DBSCAN'] + cols_b].sort_values('Concurso', ascending=False),
            use_container_width=True,
            hide_index=True
        )

        # Visualização Gráfica do Ruído
        fig_db_view = px.scatter(
            df_bin_f, x="Concurso", y="Cluster_DBSCAN",
            color=df_bin_f["Cluster_DBSCAN"].apply(lambda x: "Ruído" if x == -1 else "Padrão"),
            color_discrete_map={"Ruído": "red", "Padrão": "#00CC96"},
            title="Mapa de Densidade: Padrões vs Anomalias"
        )
        st.plotly_chart(fig_db_view, use_container_width=True)

    st.divider()
    tabs4 = st.tabs(["", "🤖 Gerador", "🤖 Predição MLP", "🤖 Predição ML"])
    # Inicio
    with tabs4[0]:  # Tendências
        st.subheader("Escolha uma dos itens na tabela")
    # 🤖 Predição MLP
    with tabs4[1]:  # Gerador
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
    # 🤖 Predição MLP
    with tabs4[2]:  # Ajuste o índice conforme a posição da sua aba de IA
        st.subheader("🤖 Predição Evolutiva via Rede Neural (MLP) CORRIGIR EU")
        st.markdown("A rede neural analisa as sequências históricas para calcular a probabilidade de cada dezena.")

        # 1. Definição dos limites baseada no seu DataFrame real
        c_min_ia = int(df_bin_f["Concurso"].min())
        c_max_ia = int(df_bin_f["Concurso"].max())

        col_ia1, col_ia2, col_ia3 = st.columns(3)

        # Input dinâmico para início (evita erro de min_value abaixo de 2880)
        ini_ia = col_ia1.number_input(
            "Treinar a partir do:",
            min_value=c_min_ia,
            max_value=c_max_ia - 1,
            value=max(c_min_ia, c_max_ia - 500)  # Sugere os últimos 500 por padrão
        )

        # Input dinâmico para fim (garante que fim_ia > ini_ia)
        fim_ia = col_ia2.number_input(
            "Até o concurso:",
            min_value=ini_ia + 1,
            max_value=c_max_ia,
            value=c_max_ia
        )

        qtd_sugerida = col_ia3.slider("Qtd de dezenas no Pool:", 6, 30, 20)

        proximo_alvo = fim_ia + 1
        st.info(f"🎯 **Objetivo:** Treinar o modelo para prever o comportamento do concurso **{proximo_alvo}**.")

        if st.button("🚀 Iniciar Treinamento e Predição"):
            with st.spinner(f"A rede neural está processando o intervalo {ini_ia} a {fim_ia}..."):
                # Chama a função de predição (certifique-se que ela está definida no seu código)
                probs = realizar_predicao_mlp_custom(df_bin_f, ini_ia, fim_ia)

                if probs is not None:
                    # Criar DataFrame de resultados e limpar tipos NumPy
                    df_prev = pd.DataFrame({
                        'Dezena': [int(i + 1) for i in range(60)],
                        'Probabilidade': probs
                    }).sort_values(by='Probabilidade', ascending=False)

                    # Pegar as Top X dezenas sugeridas
                    pool_ia = df_prev.head(qtd_sugerida)
                    dezenas_sugeridas = sorted(pool_ia['Dezena'].tolist())

                    # --- EXIBIÇÃO DO POOL ---
                    st.write(f"### 🎯 Pool de {qtd_sugerida} Dezenas Sugeridas")
                    texto_pool = ", ".join([f"{d:02d}" for d in dezenas_sugeridas])
                    st.code(texto_pool, language='text')

                    # --- VERIFICAÇÃO COM RESULTADO REAL (BACKTESTING) ---
                    concurso_real = df_bruto[df_bruto["Concurso"] == proximo_alvo]

                    if not concurso_real.empty:
                        st.divider()
                        st.write(f"### ⚖️ Verificação com o Resultado Real do {proximo_alvo}")

                        # Converte de np.int64 para int comum e limpa a exibição
                        reais = [int(n) for n in concurso_real[cols_b].values[0]]
                        acertos = set(dezenas_sugeridas).intersection(set(reais))

                        c_ver1, c_ver2 = st.columns(2)

                        # Formatação limpa conforme solicitado (sem np.int64)
                        texto_reais = ", ".join(map(str, sorted(reais)))
                        c_ver1.markdown(f"**Números Sorteados:**\n`{texto_reais}`")
                        c_ver1.metric("Acertos no Pool", len(acertos))

                        if len(acertos) >= 4:
                            st.balloons()
                            c_ver2.success(
                                f"🔥 Resultado Excelente! O pool de {qtd_sugerida} dezenas capturou {len(acertos)} acertos.")
                        else:
                            c_ver2.warning(f"O pool capturou {len(acertos)} acertos.")

                    # Gráfico das Top Probabilidades
                    fig_ia = px.bar(pool_ia, x='Dezena', y='Probabilidade',
                                    title=f"Força Estatística das Top {qtd_sugerida} Dezenas",
                                    labels={'Dezena': 'Número da Bola', 'Probabilidade': 'Peso da IA'})
                    fig_ia.update_xaxes(type='category')
                    st.plotly_chart(fig_ia, use_container_width=True)
                else:
                    st.error("Dados insuficientes para este intervalo de treino.")
    # 🤖 Predição ML
    with tabs4[3]:  # Predição ML
        st.subheader("🤖 Predição com Machine Learning (Projeto consome muito processamnto Desativado)")

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
