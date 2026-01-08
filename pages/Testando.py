import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
import plotly.express as px
import plotly.graph_objects as go
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, BisectingKMeans
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neural_network import MLPRegressor
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import bcrypt


# --- 1. CONFIGURAÇÕES TÉCNICAS E ESTÉTICAS ---
ARQUIVO_CAIXA = "resultadoJogoMegaSena.xlsx"
CAIXA_URL = "https://servicebus2.caixa.gov.br/portaldeloterias/api/resultados/download?modalidade=Mega-Sena"
COR_QUENTE = "#FF4B4B"
COR_FRIO = "#007BFF"
NOME_SISTEMA = "Mega Sena"
VERSAO = "SIPP v2.0.20260105"
PROGRAMADOR = "PRINCE, K.B"
LINK_PESSOAL = "https://klaytonprincems.github.io/site/"
cols_b = ['Bola1', 'Bola2', 'Bola3', 'Bola4', 'Bola5', 'Bola6']
st.set_page_config(    page_title=NOME_SISTEMA,    page_icon="🍀",    layout="wide",    initial_sidebar_state='expanded',    menu_items={"About": LINK_PESSOAL} )
st.markdown(f"""    <style>    .main-title {{text-align: center; color: {COR_QUENTE}; font-weight: bold; margin-bottom: 20px;}}    .stButton>button {{width: 100%; font-weight: bold; border-radius: 10px; height: 45px;}}    .footer-text {{text-align: center; padding: 20px; color: #888; font-size: 14px;}}    .crisp-box {{background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid {COR_FRIO}; margin-bottom: 10px;}}    </style>    """, unsafe_allow_html=True)





# --- 2. MOTOR DE DADOS ---

def baixar_dados():
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(CAIXA_URL, headers=headers, timeout=30)
        if res.status_code == 200:
            with open(ARQUIVO_CAIXA, "wb") as f: f.write(res.content)
            return True
        return False
    except Exception as e:
        st.error(f"Falha no download dos dados: {e}")
        return False
@st.cache_data
def processar_base_completa():
    if not os.path.exists(ARQUIVO_CAIXA):
        if not baixar_dados():
            return None, None
    try:
        df = pd.read_excel(ARQUIVO_CAIXA, engine='openpyxl')
        df.columns = [str(c).strip() for c in df.columns]
        df = df.dropna(subset=["Concurso"])
        df["Concurso"] = df["Concurso"].astype(int)

        cols_bolas = ['Bola1', 'Bola2', 'Bola3', 'Bola4', 'Bola5', 'Bola6']
        df_melt = df.melt(id_vars=["Concurso"], value_vars=cols_bolas, value_name='N').dropna()
        df_bin = pd.crosstab(df_melt["Concurso"], df_melt['N'])
        for i in range(1, 61):
            if i not in df_bin.columns: df_bin[i] = 0
        df_bin = df_bin.reindex(columns=sorted(df_bin.columns)).reset_index()
        return df, df_bin
    except Exception as e:
        st.error(f"Erro ao processar o arquivo Excel: {e}")
        return None, None
@st.cache_data
def processar_base_par_soma(df_bruto_f):
    paridade = df_bruto_f[cols_b].apply(lambda x: sum(1 for n in x if n % 2 == 0), axis=1)
    df_par = pd.DataFrame({
        'Concurso': df_bruto_f["Concurso"],
        'Pares': paridade,
        'Ímpares': 6 - paridade,
        'Soma': df_bruto_f[cols_b].sum(axis=1)
    })
    return df_par
@st.cache_data
def processar_quadrantes(df_f):
    def identificar_quadrante(n):
        row = (n - 1) // 10
        col = (n - 1) % 10
        if row < 3:
            return "Q1" if col < 5 else "Q2"
        else:
            return "Q3" if col < 5 else "Q4"
    quadrantes_contagem = []
    for _, row in df_f[cols_b].iterrows():
        qs = [identificar_quadrante(n) for n in row]
        quadrantes_contagem.append({
            "Q1": qs.count("Q1"), "Q2": qs.count("Q2"),
            "Q3": qs.count("Q3"), "Q4": qs.count("Q4")
        })
    return pd.DataFrame(quadrantes_contagem, index=df_f["Concurso"])
@st.cache_data
def executar_agnes(df_bin_f, n_clusters=4, metric='euclidean', linkage='ward'):
    data = df_bin_f.drop(columns=["Concurso"], errors='ignore').copy()
    data.columns = data.columns.astype(str)
    model = AgglomerativeClustering(        n_clusters=n_clusters,        metric=metric,        linkage=linkage) # Agora o linkage é um parâmetro
    labels = model.fit_predict(data)
    return labels
@st.cache_data
def plotar_dendrograma_range(df_bin_f, c_ini, c_fim):
    df_range = df_bin_f[(df_bin_f["Concurso"] >= c_ini) & (df_bin_f["Concurso"] <= c_fim)].copy()
    if df_range.empty:
        return None
    concursos = df_range['Concurso'].astype(str).values
    data = df_range.drop(columns=["Concurso"], errors='ignore')
    data.columns = data.columns.astype(str)
    linkage_matrix = sch.linkage(data, method='ward')
    fig, ax = plt.subplots(figsize=(12, 6))
    sch.dendrogram(linkage_matrix, labels=concursos, leaf_rotation=90, leaf_font_size=9, ax=ax)
    ax.set_title(f"Dendrograma: De {c_ini} até {c_fim}")
    ax.set_ylabel("Distância (Dissimilaridade)")
    plt.tight_layout()
    return fig
@st.cache_data
def executar_diana(df_bin_f, n_clusters=4):
    data = df_bin_f.drop(columns=["Concurso"], errors='ignore').copy()
    data.columns = data.columns.astype(str)
    model = BisectingKMeans(n_clusters=n_clusters, random_state=42, bisecting_strategy='biggest_inertia')
    labels = model.fit_predict(data)
    return labels
@st.cache_data
def executar_dbscan(df_bin_f, eps=0.5, min_samples=3):
    data = df_bin_f.drop(columns=["Concurso"], errors='ignore').copy()
    data.columns = data.columns.astype(str)
    model = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    labels = model.fit_predict(data)
    return labels
@st.cache_data
def realizar_predicao_mlp_custom(df_bin_f, c_ini, c_fim):
    """
    Realiza uma predição multi-label usando MLPRegressor, a abordagem correta
    para prever múltiplas probabilidades contínuas.
    """
    df_modelo = df_bin_f[(df_bin_f["Concurso"] >= c_ini) & (df_bin_f["Concurso"] <= c_fim)].copy()
    if len(df_modelo) < 20:
        return None
    # Garante que estamos usando apenas as 60 dezenas
    dezenas_cols = list(range(1, 61))
    data = df_modelo[dezenas_cols].values
    X = data[:-1]
    y = data[1:]
    # SOLUÇÃO: Usar MLPRegressor em vez de MLPClassifier
    model = MLPRegressor(        hidden_layer_sizes=(120, 80, 40),        activation='relu',        solver='adam',        alpha=0.001,        learning_rate='adaptive',        max_iter=1000,        early_stopping=True,        validation_fraction=0.1,        random_state=42    )
    # O fit agora funciona, pois MLPRegressor suporta multi-output nativamente
    model.fit(X, y)
    # A predição agora retorna diretamente os valores previstos (nossas probabilidades)
    ultimo_concurso_conhecido = data[-1].reshape(1, -1)
    probabilidades = model.predict(ultimo_concurso_conhecido)[0]
    # Garante que as probabilidades fiquem no intervalo [0, 1]
    probabilidades = np.clip(probabilidades, 0, 1)
    return probabilidades
@st.cache_data
def realizar_predicao_rf_custom(df_bin_f, c_ini, c_fim, n_estimators=100, max_depth=10):
    """
    Realiza uma predição multi-label usando RandomForestClassifier de forma
    otimizada e robusta, garantindo a estrutura correta dos dados.
    """
    df_modelo = df_bin_f[(df_bin_f["Concurso"] >= c_ini) & (df_bin_f["Concurso"] <= c_fim)].copy()
    if len(df_modelo) < 30:
        return None

    # --- INÍCIO DA CORREÇÃO ---
    # 1. Garante que estamos usando apenas as 60 dezenas como colunas.
    #    As colunas no df_bin_f são números (1, 2, ..., 60).
    dezenas_cols = list(range(1, 61))
    data = df_modelo[dezenas_cols].values
    # Agora 'data' tem GARANTIDAMENTE 60 colunas (índices 0-59).
    # --- FIM DA CORREÇÃO ---
    X = data[:-1]
    y = data[1:]
    cols_com_variacao = [i for i in range(y.shape[1]) if len(np.unique(y[:, i])) > 1]
    if not cols_com_variacao:
        return np.zeros(60)
    y_filtrado = y[:, cols_com_variacao]
    base_classifier = RandomForestClassifier(        n_estimators=n_estimators,        max_depth=max_depth,        random_state=42,        n_jobs=None    )
    multi_target_forest = MultiOutputClassifier(base_classifier, n_jobs=-1)
    multi_target_forest.fit(X, y_filtrado)
    ultimo_concurso_conhecido = data[-1].reshape(1, -1)
    probabilidades_parciais = multi_target_forest.predict_proba(ultimo_concurso_conhecido)
    probs_agregadas = np.zeros(60)
    for i, prob in enumerate(probabilidades_parciais):
        indice_dezena_original = cols_com_variacao[i]
        classes_aprendidas = multi_target_forest.estimators_[i].classes_
        if 1 in classes_aprendidas:
            idx_classe_1 = np.where(classes_aprendidas == 1)[0][0]
            prob_classe_1 = prob[0][idx_classe_1]
            probs_agregadas[indice_dezena_original] = prob_classe_1
    return probs_agregadas


def gerar_hash(senha_plana):
    # Transforma a senha em bytes e gera o salt
    senha_bytes = senha_plana.encode('utf-8')
    salt = bcrypt.gensalt()
    # Gera o hash
    hash_resultado = bcrypt.hashpw(senha_bytes, salt)
    return hash_resultado.decode('utf-8')
print(f'Gerando senha {gerar_hash("jose")}')


# 1. Carregar arquivo YAML
caminho_base = r"F:\Documents\klayton\Git_hub2026\kbprince"
caminho_yaml = os.path.join(caminho_base, "usuarios.yaml")

with open(caminho_yaml) as file:
    config = yaml.load(file, Loader=SafeLoader)

# 2. Criar o objeto de autenticação (SEM o pre-authorized)
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

# 3. Renderizar formulário de login
# Nas versões novas, usamos o método login sem passar 'main' se quiser o padrão
# login retorna apenas o status de autenticação diretamente
authenticator.login()

# 4. Lógica de Verificação
if st.session_state["authentication_status"]:
    # BOTÃO DE SAIR
    authenticator.logout('Sair')

    st.info(f'Bem-vindo, {st.session_state["name"]}')


    # --- 3. INTERFACE E LÓGICA ---
    st.markdown(f"<h1 class='main-title'>🍀 {NOME_SISTEMA}</h1>", unsafe_allow_html=True)
    df_bruto, df_binario = processar_base_completa()

    # Inicio do sidebar, menu lateral
    if df_bruto is not None and df_binario is not None:
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
        # Fim do sidebar, menu lateral




        # --- O RESTANTE DO CÓDIGO CONTINUA DAQUI ---
        # O filtro agora usará c_ini e c_fim que estão sempre sincronizados
        df_bruto_f = df_bruto[(df_bruto["Concurso"] >= c_ini) & (df_bruto["Concurso"] <= c_fim)].copy()
        df_bin_f = df_binario[(df_binario["Concurso"].astype(int) >= c_ini) & (df_binario["Concurso"].astype(int) <= c_fim)].copy()





        # ... (o resto do seu código permanece o mesmo) ...
        df_bruto_f = df_bruto[(df_bruto["Concurso"] >= c_ini) & (df_bruto["Concurso"] <= c_fim)].copy()
        df_bin_f = df_binario[(df_binario["Concurso"].astype(int) >= c_ini) & (df_binario["Concurso"].astype(int) <= c_fim)].copy()

        weights = np.exp(np.linspace(0, 3, len(df_bin_f))) if metodo == "Exponencial" else np.linspace(0.1, 1.0, len(df_bin_f))
        scores = df_bin_f.drop(columns=["Concurso"]).astype(float).multiply(weights, axis=0).sum()
        df_ranking = pd.DataFrame({'Número': scores.index.astype(int), 'Score': scores.values}).sort_values('Score', ascending=False)

        df_par_soma = processar_base_par_soma(df_bruto_f)
        # Paridade, Soma, Fogo e Gelo, Quadrantes
        try:
            tabs1 = st.tabs(["", "📈🪙 Paridade", "📈➕ Soma", "🚀 Fogo e Gelo", "🟦 Quadrantes"])
            with tabs1[0]:
                st.info("Navegue pelas abas para explorar as análises de tendências básicas.")
            with tabs1[1]:
                st.subheader("📈 Tendências de Paridade")
                st.bar_chart(df_par_soma.set_index('Concurso')[['Pares', 'Ímpares']])
                st.dataframe(df_par_soma[['Concurso', 'Pares', 'Ímpares']].iloc[::-1], width="stretch", hide_index=True)
            with tabs1[2]:
                st.subheader("📈 Tendências de Soma com Análise Estatística")

                # 1. Calcular as métricas estatísticas necessárias
                soma_media = df_par_soma['Soma'].mean()
                soma_std = df_par_soma['Soma'].std()
                limite_superior = soma_media + soma_std
                limite_inferior = soma_media - soma_std

                # Exibir as métricas em colunas para uma visualização limpa
                c1, c2, c3 = st.columns(3)
                c1.metric("Média da Soma", f"{soma_media:.2f}")
                c2.metric("Limite Superior (Média + 1σ)", f"{limite_superior:.2f}")
                c3.metric("Limite Inferior (Média - 1σ)", f"{limite_inferior:.2f}")

                # 2. Usar Plotly para criar um gráfico com múltiplas camadas
                fig = go.Figure()
                # Adiciona a linha principal da Soma
                fig.add_trace(go.Scatter(                x=df_par_soma['Concurso'],                y=df_par_soma['Soma'],                mode='lines',                name='Soma das Dezenas',                line=dict(color='#1f77b4') )) # Cor azul padrão
                # Adiciona a linha da Média
                fig.add_hline(                y=soma_media,                line_dash="dash",                line_color="#ff7f0e",                  annotation_text="Média",                annotation_position="bottom right"            ) # Laranja #ff7f0e
                # Adiciona a linha do Limite Superior
                fig.add_hline(                y=limite_superior,                line_dash="dot",                line_color="#d62728",                  annotation_text="Média +1σ",                annotation_position="bottom right"            ) # Vermelho #d62728
                # Adiciona a linha do Limite Inferior
                fig.add_hline(                y=limite_inferior,                line_dash="dot",                line_color="#2ca02c",                  annotation_text="Média -1σ",                annotation_position="bottom right"            ) # Verde #2ca02c
                # Ajustes finais de layout do gráfico
                fig.update_layout(                title="Evolução da Soma das Dezenas com Bandas de Média",                xaxis_title="Concurso",                yaxis_title="Soma",                showlegend=True            )
                st.plotly_chart(fig, width="stretch")
                # A tabela de dados continua a mesma
                st.dataframe(df_par_soma[['Concurso', 'Soma']].iloc[::-1], width="stretch", hide_index=True)
            with tabs1[3]:
                st.subheader("🚀 Fogo e Gelo")
                # 1. Seleciona os dados (como antes)
                top = df_ranking.head(qtd_top)
                bottom = df_ranking.tail(qtd_top)
                # 2. Adiciona uma coluna de categoria para diferenciar os grupos
                top['Categoria'] = 'Quente'
                bottom['Categoria'] = 'Frio'
                # 3. Concatena os dois DataFrames em um só
                df_plot = pd.concat([top, bottom])
                # 4. SOLUÇÃO: Ordena o DataFrame pelo número da dezena
                df_plot = df_plot.sort_values('Número')
                # 5. Cria o gráfico de barras
                fig = px.bar(                df_plot,                x='Número',                y='Score',                color='Categoria',                title="Extremos de Tendência: Dezenas Quentes vs. Frias",                labels={'Número': 'Dezena', 'Score': 'Pontuação de Tendência'},                color_discrete_map={
                        'Quente': COR_QUENTE,
                        'Frio': COR_FRIO
                    }            )

                # 6. SOLUÇÃO: Força o eixo X a ter a categoria completa de 1 a 60
                # Isso garante que as barras apareçam em suas posições corretas no volante.
                fig.update_xaxes(
                    type='category',
                    categoryorder='array',  # Garante a ordem que definimos
                    categoryarray=list(range(1, 61))  # Define a ordem e o range completo
                )

                st.plotly_chart(fig, width="stretch")

                # A exibição da tabela de ranking completa continua a mesma
                st.write("**Classificação Completa por Pontuação (Ranking SIPP)**")
                st.dataframe(df_ranking, width="stretch", hide_index=True)
            with tabs1[4]:
                st.subheader("🟦 Distribuição por Quadrantes")
                df_q = processar_quadrantes(df_bruto_f)
                fig_q = px.bar(df_q.reset_index(), x="Concurso", y=["Q1", "Q2", "Q3", "Q4"], title="Equilíbrio de Quadrantes por Concurso")
                st.plotly_chart(fig_q, width="stretch")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Média Q1", f"{df_q['Q1'].mean():.2f}")
                c2.metric("Média Q2", f"{df_q['Q2'].mean():.2f}")
                c3.metric("Média Q3", f"{df_q['Q3'].mean():.2f}")
                c4.metric("Média Q4", f"{df_q['Q4'].mean():.2f}")
                st.dataframe(df_q.iloc[::-1], width="stretch")
        except Exception as e:
            st.error(f"Erro em Dados brutos, Binarios, Governança e Metodologia.  \n{e}")

        # Percentis, Associação, Analise
        try:
            st.divider()
            tabs2 = st.tabs(["", "🎯 Percentis", "🔗 Associação", "🔍 Analisador"])
            with tabs2[0]:
                st.info("Explore as abas para análises estatísticas e de associação.")
            with tabs2[1]:
                st.subheader("🎯 Análise de Percentis")
                p_alta = df_ranking['Score'].quantile(0.80)
                p_baixa = df_ranking['Score'].quantile(0.20)
                c1, c2 = st.columns(2)
                c1.metric("Corte Percentil 80 (Quentes)", f"{p_alta:.2f}")
                c2.metric("Corte Percentil 20 (Frios)", f"{p_baixa:.2f}")
                fig_p = px.ecdf(df_ranking, x="Score", title="Curva de Probabilidade Acumulada")
                st.plotly_chart(fig_p, width="stretch")
            with tabs2[2]:
                st.subheader("🔗 Mineração de Regras de Associação")
                col_alg, col_sup, col_conf = st.columns(3)
                alg_assoc = col_alg.selectbox("Algoritmo:", ["FP-Growth", "Apriori"], key="select_assoc")
                min_sup = col_sup.slider("Suporte Mínimo:", 0.001, 0.05, 0.01, format="%.3f", key="slider_sup")
                min_conf = col_conf.slider("Confiança Mínima:", 0.1, 1.0, 0.10, key="slider_conf")
                df_assoc_input = df_bin_f.drop(columns=["Concurso"]).astype(bool)
                try:
                    frequent_itemsets = fpgrowth(df_assoc_input, min_support=min_sup, use_colnames=True) if alg_assoc == "FP-Growth" else apriori(df_assoc_input, min_support=min_sup, use_colnames=True)
                    if not frequent_itemsets.empty:
                        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_conf)
                        if not rules.empty:
                            rules["antecedents"] = rules["antecedents"].apply(list)
                            rules["consequents"] = rules["consequents"].apply(list)
                            st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].sort_values('lift', ascending=False), width="stretch")
                        else:
                            st.warning("Nenhuma regra encontrada com a confiança mínima.")
                    else:
                        st.warning("Nenhum conjunto frequente encontrado com o suporte mínimo.")
                except Exception as e:
                    st.error(f"Erro na associação: {e}")
            with tabs2[3]:
                st.subheader("🔍 Analisador de Jogos Personalizados")
                dezenas_user = st.multiselect("Selecione de 4 a 15 DEZENAS:", list(range(1, 61)), key='multiselect_analisador_1')
                if 4 <= len(dezenas_user) <= 15:
                    set_user = set(dezenas_user)
                    ocorrencias = []
                    for _, row in df_bruto.iterrows():
                        sorteados = set(row[cols_b].values)
                        acertos = len(set_user.intersection(sorteados))
                        if acertos >= 4:
                            ocorrencias.append({
                                "Concurso": row["Concurso"], "Data": row.get("Data Sorteio", "N/A"),
                                "Acertos": acertos, "Ganhadores 6": row.get("Ganhadores_Sena", 0)
                            })
                    df_ocorr = pd.DataFrame(ocorrencias)
                    c1, c2, c3 = st.columns(3)
                    score_jogo = sum(df_ranking[df_ranking['Número'] == n]['Score'].values[0] for n in dezenas_user) / len(dezenas_user)
                    c1.metric("Score Médio SIPP", f"{score_jogo:.2f}")
                    c2.metric("Vezes Premiado (4+)", len(df_ocorr))
                    prob = (len(df_ocorr) / len(df_bruto)) * 100 if len(df_bruto) > 0 else 0
                    c3.metric("Frequência Histórica", f"{prob:.4f}%")
                    if not df_ocorr.empty:
                        st.dataframe(df_ocorr.sort_values("Concurso", ascending=False), width="stretch")
                    else:
                        st.info("Este grupo nunca premiou com 4+ acertos.")
                else:
                    st.info("Selecione entre 4 e 15 dezenas.")
        except Exception as e:
            st.info("Erro PErcentis, Associação e Analisador.")

        # Clustering, Agnes, Agnes x Diana, Dendograma, DBSCAN
        try:
            st.divider()
            tabs3 = st.tabs(["", "🧩 Clustering", "🧩 Agnes", "🧩 Agnes x Diana", "🧩 Dendograma", "🧩 DBSCAN"])
            with tabs3[0]:
                st.info("Explore diferentes algoritmos de clusterização para encontrar padrões.")
            with tabs3[1]:
                st.subheader("🧩 Agrupamento (Clustering) de Padrões")
                tipo_cluster = st.radio("Agrupar por:", ["Concursos (Padrões de Sorteio)", "Dezenas (Afinidade)"], key="radio_cluster")
                n_clusters = st.slider("Número de Clusters:", 2, 10, 4, key="slider_kmeans")
                data_cluster = df_bin_f.drop(columns=["Concurso"], errors='ignore').copy()
                data_cluster.columns = data_cluster.columns.astype(str)
                if tipo_cluster == "Concursos (Padrões de Sorteio)":
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(data_cluster)
                    df_bin_f['Cluster'] = kmeans.labels_
                    fig_c = px.scatter(df_bin_f, x="Concurso", y="Cluster", color=df_bin_f["Cluster"].astype(str), title="Concursos por Cluster")
                    st.plotly_chart(fig_c, width="stretch")
                else:
                    data_cluster_t = data_cluster.T
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(data_cluster_t)
                    df_cluster_dez = pd.DataFrame({'Número': range(1, 61), 'Cluster': kmeans.labels_})
                    for i in range(n_clusters):
                        st.write(f"**Cluster {i}:** {df_cluster_dez[df_cluster_dez['Cluster'] == i]['Número'].tolist()}")
            with tabs3[2]:
                st.subheader("🧩 Agrupamento Hierárquico (AGNES)")

                # A interface para o usuário permanece simples.
                # SOLUÇÃO: A chave foi renomeada para ser única e evitar conflito.
                metrica = st.selectbox(
                    "Métrica de Distância:",
                    ["euclidean", "manhattan", "cosine"],
                    key="agnes_metric_selector",  # <-- CHAVE RENOMEADA E ÚNICA
                    help="O método de agrupamento se ajustará automaticamente à métrica escolhida."
                )

                # A chave do slider também precisa ser única para esta aba.
                n_cl = st.slider("Qtd de Grupos:", 2, 8, 4, key="agnes_cluster_slider")

                if st.button("Executar AGNES", key="agnes_execute_button"):  # Adicionar key ao botão também é uma boa prática
                    # --- LÓGICA DE ADAPTAÇÃO AUTOMÁTICA ---
                    if metrica == 'euclidean':
                        linkage_metodo = 'ward'
                    else:
                        linkage_metodo = 'average'
                    # -----------------------------------------

                    st.info(f"Executando com a combinação: Métrica='{metrica}' e Linkage='{linkage_metodo}'.")

                    # Chama a função com os parâmetros corretos e seguros
                    # (Assumindo que a função executar_agnes já foi atualizada para aceitar 'linkage')
                    labels = executar_agnes(df_bin_f, n_cl, metrica, linkage_metodo)
                    df_bin_f['Agnes_Cluster'] = labels

                    # Exibe os resultados
                    fig_agnes = px.scatter(
                        df_bin_f,
                        x="Concurso",
                        y="Agnes_Cluster",
                        color=df_bin_f["Agnes_Cluster"].astype(str),
                        title=f"Concursos por Similaridade Hierárquica ({metrica.capitalize()})"
                    )
                    st.plotly_chart(fig_agnes, width="stretch")
            with tabs3[3]:
                st.subheader("🧬 AGNES vs DIANA")
                metodo_h = st.selectbox("Direção:", ["AGNES (Agrupar)", "DIANA (Dividir)"], key="select_h_method")
                n_h = st.slider("Número de Grupos:", 2, 10, 4, key="slider_h")
                if st.button("Executar Mineração Hierárquica"):
                    labels = executar_agnes(df_bin_f, n_h) if metodo_h == "AGNES (Agrupar)" else executar_diana(df_bin_f, n_h)
                    df_bin_f['Cluster_H'] = labels
                    fig_h = px.scatter(df_bin_f, x="Concurso", y="Cluster_H", color=labels.astype(str), title=f"Resultado via {metodo_h}")
                    st.plotly_chart(fig_h, width="stretch")
            with tabs3[4]:
                st.subheader("🌳 Análise Genealógica Customizada (Dendrograma)")
                min_hist, max_hist = int(df_bin_f["Concurso"].min()), int(df_bin_f["Concurso"].max())
                c_start = st.number_input("Concurso Inicial:", min_hist, max_hist, max(min_hist, max_hist - 50), key="dendro_start")
                c_end = st.number_input("Concurso Final:", min_hist, max_hist, max_hist, key="dendro_end")
                if c_start < c_end:
                    if (c_end - c_start) > 150: st.warning("Range grande pode poluir a visualização.")
                    fig_d = plotar_dendrograma_range(df_bin_f, c_start, c_end)
                    if fig_d: st.pyplot(fig_d)
                else:
                    st.error("O concurso inicial deve ser menor que o final.")
            with tabs3[5]:
                st.subheader("📡 Detector de Anomalias (DBSCAN)")
                c1, c2 = st.columns(2)
                val_eps = c1.slider("Sensibilidade (Epsilon):", 0.5, 3.0, 1.2, key="slider_eps")
                val_min = c2.slider("Mínimo de Vizinhos:", 2, 10, 3, key="slider_min_samples")
                labels_db = executar_dbscan(df_bin_f, eps=val_eps, min_samples=val_min)
                df_bin_f['Cluster_DBSCAN'] = labels_db
                m1, m2, m3 = st.columns(3)
                total, ruidos = len(df_bin_f), list(labels_db).count(-1)
                m1.metric("Total Analisado", total)
                m2.metric("Concursos no Padrão", total - ruidos)
                m3.metric("Concursos no Ruído", ruidos, delta=f"{(ruidos / total) * 100:.1f}%" if total > 0 else "0.0%", delta_color="inverse")
                fig_db_view = px.scatter(df_bin_f, x="Concurso", y="Cluster_DBSCAN", color=df_bin_f["Cluster_DBSCAN"].apply(lambda x: "Ruído" if x == -1 else "Padrão"), color_discrete_map={"Ruído": "red", "Padrão": "#00CC96"}, title="Mapa de Densidade: Padrões vs Anomalias")
                st.plotly_chart(fig_db_view, width="stretch")
        except Exception as e:
            st.error(f"Erro em Clustering, Agnes, Agnes x Diana, Dendograma, DBSCAN.  \n{e}")

        # Gerador, Analisador, Predição
        try:
            st.divider()
            tabs4 = st.tabs(["", "🤖 Gerador", "🔍 Analisador", "🤖 Predição MLP"])
            with tabs4[0]:
                st.info("Use as ferramentas de IA para gerar jogos ou analisar probabilidades.")
            with tabs4[1]:
                st.subheader("🎲 Gerador Inteligente de Dezenas")
                st.markdown(
                    "Use os filtros para gerar um jogo com base em restrições estatísticas. O gerador tentará encontrar um jogo que satisfaça todas as condições.")

                c1, c2 = st.columns(2)

                # Widget para quantidade de números permanece o mesmo
                qtd_gerar = c1.number_input("Quantidade de números:", 6, 15, 6, key="gerador_qtd")

                # Widget para quantidade de pares também permanece o mesmo
                n_pares = c2.slider("Quantidade de Pares:", 0, qtd_gerar, qtd_gerar // 2, key="gerador_pares")

                # --- INÍCIO DA CORREÇÃO DINÂMICA ---

                # 1. Calcula os limites teóricos da soma para a quantidade de dezenas escolhida
                soma_min_possivel = sum(range(1, qtd_gerar + 1))
                soma_max_possivel = sum(range(61 - qtd_gerar, 61))

                # 2. Define um valor padrão dinâmico e razoável para o slider
                #    Vamos usar uma faixa em torno do ponto médio teórico.
                ponto_medio = (soma_min_possivel + soma_max_possivel) / 2
                spread = (soma_max_possivel - soma_min_possivel) * 0.15  # Uma faixa de 30% em torno do meio
                default_min = max(soma_min_possivel, int(ponto_medio - spread))
                default_max = min(soma_max_possivel, int(ponto_medio + spread))

                # 3. Cria o slider de soma com os limites e valores padrão DINÂMICOS
                st.write(
                    f"Para **{qtd_gerar} números**, a soma pode variar de **{soma_min_possivel}** a **{soma_max_possivel}**.")
                soma_range = st.slider(
                    "Faixa de Soma:",
                    min_value=soma_min_possivel,
                    max_value=soma_max_possivel,
                    value=(default_min, default_max),  # Usa o padrão dinâmico
                    key="gerador_soma_dinamico"  # Nova chave para evitar conflitos
                )
                # --- FIM DA CORREÇÃO DINÂMICA ---

                usar_score = st.toggle(
                    "Ponderar pela pontuação SIPP (dezenas 'quentes')",
                    value=True,
                    help="Se ativado, o gerador dará preferência às dezenas com maior pontuação de tendência (quentes).",
                    key="gerador_usar_score"
                )

                if st.button("🍀 Gerar Jogo Inteligente", key="gerador_button"):
                    # (O restante do código para gerar e exibir o jogo permanece o mesmo)
                    with st.spinner("Procurando a combinação perfeita..."):
                        tentativas = 0
                        jogo_encontrado = None

                        if usar_score:
                            probabilidades = df_ranking.sort_values('Número')['Score'].values
                            probabilidades /= probabilidades.sum()
                            pool_dezenas = range(1, 61)
                        else:
                            probabilidades = None
                            pool_dezenas = range(1, 61)

                        while tentativas < 5000:
                            jogo = np.random.choice(pool_dezenas, qtd_gerar, replace=False, p=probabilidades)
                            soma_jogo = sum(jogo)
                            pares_jogo = sum(1 for n in jogo if n % 2 == 0)

                            if pares_jogo == n_pares and soma_range[0] <= soma_jogo <= soma_range[1]:
                                jogo_encontrado = sorted(jogo)
                                break
                            tentativas += 1

                    if jogo_encontrado:
                        st.success("🍀 Combinação encontrada com sucesso!")
                        with st.container(border=True):
                            st.write("### Jogo Gerado:")
                            cols = st.columns(len(jogo_encontrado))
                            for i, num in enumerate(jogo_encontrado):
                                cols[i].markdown(
                                    f"<div style='background-color: #262730; border-radius: 50%; width: 50px; height: 50px; display: flex; justify-content: center; align-items: center; color: white; font-size: 20px; font-weight: bold;'>{num:02d}</div>",
                                    unsafe_allow_html=True
                                )
                            st.divider()
                            st.write("#### Estatísticas do Jogo:")
                            soma_final = sum(jogo_encontrado)
                            pares_final = sum(1 for n in jogo_encontrado if n % 2 == 0)
                            impares_final = len(jogo_encontrado) - pares_final
                            score_jogo = df_ranking[df_ranking['Número'].isin(jogo_encontrado)]['Score'].mean()
                            m1, m2, m3, m4 = st.columns(4)
                            m1.metric("Soma Total", soma_final)
                            m2.metric("Nº de Pares", pares_final)
                            m3.metric("Nº de Ímpares", impares_final)
                            m4.metric("Score SIPP Médio", f"{score_jogo:.2f}")
                    else:
                        st.warning(
                            "Não foi possível gerar um jogo com as restrições exatas. Tente ampliar a faixa de soma ou desativar a ponderação SIPP.")
            with tabs4[2]:
                st.subheader("🔍 Analisador de Jogos Personalizados")
                dezenas_user_2 = st.multiselect("Selecione de 4 a 15 DEZENAS:", list(range(1, 61)), key='multiselect_analisador_2')
                if 4 <= len(dezenas_user_2) <= 15:
                    set_user = set(dezenas_user_2)
                    ocorrencias = []
                    for _, row in df_bruto.iterrows():
                        sorteados = set(row[cols_b].values)
                        acertos = len(set_user.intersection(sorteados))
                        if acertos >= 4:
                            ocorrencias.append({
                                "Concurso": row["Concurso"], "Data": row.get("Data Sorteio", "N/A"),
                                "Acertos": acertos, "Ganhadores 6": row.get("Ganhadores_Sena", 0)
                            })
                    df_ocorr = pd.DataFrame(ocorrencias)
                    c1, c2, c3 = st.columns(3)
                    score_jogo = sum(df_ranking[df_ranking['Número'] == n]['Score'].values[0] for n in dezenas_user_2) / len(dezenas_user_2)
                    c1.metric("Score Médio SIPP", f"{score_jogo:.2f}")
                    c2.metric("Vezes Premiado (4+)", len(df_ocorr))
                    prob = (len(df_ocorr) / len(df_bruto)) * 100 if len(df_bruto) > 0 else 0
                    c3.metric("Frequência Histórica", f"{prob:.4f}%")
                    if not df_ocorr.empty:
                        st.dataframe(df_ocorr.sort_values("Concurso", ascending=False), width="stretch")
                    else:
                        st.info("Este grupo nunca premiou com 4+ acertos.")
                else:
                    st.info("Selecione entre 4 e 15 dezenas.")
            with tabs4[3]:  # Esta é agora a sua aba unificada de "Predição com IA"
                st.subheader("🧠 Predição com Inteligência Artificial")
                st.markdown("O usuário escolhe qual IA usar")

                # 1. SELETOR DE MODELO: O usuário escolhe qual IA usar
                modelo_escolhido = st.selectbox(
                    "Escolha o Modelo de Predição:",
                    ("Rede Neural (MLP)", "Floresta Aleatória (Random Forest)"),
                    key="ia_model_selector"
                )

                # Mensagem de ajuda para o usuário
                if modelo_escolhido == "Rede Neural (MLP)":
                    st.markdown(
                        "O MLP é uma rede neural profunda que busca padrões complexos e sequenciais nos dados. Pode ser mais lento para treinar.")
                else:
                    st.markdown(
                        "O Random Forest treina múltiplas árvores de decisão, sendo um modelo robusto, rápido e geralmente ótimo para dados tabulares.")

                # 2. WIDGETS DE CONTROLE UNIFICADOS: Usam chaves únicas
                c_min_ia, c_max_ia = int(df_bin_f["Concurso"].min()), int(df_bin_f["Concurso"].max())
                col_ia1, col_ia2, col_ia3 = st.columns(3)
                ini_ia = col_ia1.number_input("Treinar a partir do:", c_min_ia, c_max_ia - 1, max(c_min_ia, c_max_ia - 500),
                                              key="ia_ini_unificado")
                fim_ia = col_ia2.number_input("Até o concurso:", ini_ia + 1, c_max_ia, c_max_ia, key="ia_fim_unificado")
                qtd_sugerida = col_ia3.slider("Qtd de DEZENAS:", 6, 30, 20, key="ia_pool_unificado")

                proximo_alvo = fim_ia + 1
                st.info(f"🎯 **Objetivo:** Prever o comportamento do concurso **{proximo_alvo}** usando **{modelo_escolhido}**.")

                # 3. BOTÃO E LÓGICA CONDICIONAL
                if st.button(f"🚀 Iniciar Treinamento com {modelo_escolhido}", key="ia_button_unificado"):

                    # Define a mensagem do spinner e a função a ser chamada com base na escolha
                    if modelo_escolhido == "Rede Neural (MLP)":
                        spinner_message = "A rede neural está processando..."
                        funcao_predicao = realizar_predicao_mlp_custom
                    else:
                        spinner_message = "A floresta aleatória está crescendo e analisando os dados..."
                        funcao_predicao = realizar_predicao_rf_custom

                    with st.spinner(spinner_message):
                        # Chama a função de predição selecionada
                        probs = funcao_predicao(df_bin_f, ini_ia, fim_ia)

                        if probs is not None and len(probs) == 60:
                            # 4. EXIBIÇÃO DE RESULTADOS (CÓDIGO GENÉRICO)
                            # Esta parte é a mesma para ambos os modelos
                            df_prev = pd.DataFrame({'Dezena': range(1, 61), 'Probabilidade': probs}).sort_values(
                                'Probabilidade', ascending=False)
                            pool_ia = df_prev.head(qtd_sugerida)
                            dezenas_sugeridas = sorted(pool_ia['Dezena'].tolist())

                            st.write(f"### 🎯 Pool de {qtd_sugerida} Dezenas Sugeridas pelo {modelo_escolhido}")
                            st.code(", ".join(f"{d:02d}" for d in dezenas_sugeridas), language='text')

                            # Backtesting
                            concurso_real = df_bruto[df_bruto["Concurso"] == proximo_alvo]
                            if not concurso_real.empty:
                                st.divider()
                                st.write(f"### ⚖️ Verificação com o Resultado Real do Concurso {proximo_alvo}")
                                reais = [int(n) for n in concurso_real[cols_b].values[0]]
                                acertos = set(dezenas_sugeridas).intersection(set(reais))

                                c_ver1, c_ver2 = st.columns(2)
                                texto_reais = ", ".join(map(str, sorted(reais)))
                                c_ver1.markdown(f"**Números Sorteados:**\n`{texto_reais}`")
                                c_ver1.metric(f"Acertos do {modelo_escolhido} no Pool", len(acertos))

                                if len(acertos) >= 4:
                                    st.balloons()
                                    c_ver2.success(f"🔥 Excelente! O pool capturou {len(acertos)} acertos.")
                                else:
                                    c_ver2.warning(f"O pool capturou {len(acertos)} acertos.")

                            # Gráfico
                            fig_ia = px.bar(pool_ia, x='Dezena', y='Probabilidade',
                                            title=f"Força Estatística das Top {qtd_sugerida} Dezenas ({modelo_escolhido})",
                                            labels={'Dezena': 'Número da Bola', 'Probabilidade': 'Peso do Modelo'})
                            fig_ia.update_xaxes(type='category')
                            st.plotly_chart(fig_ia, width="stretch") # use_container_width=True trocado por po width=Stretch
                        else:
                            st.error("Dados insuficientes para este intervalo de treino. Tente um intervalo maior.")
        except Exception as e:
            st.error(f"Erro em Gerador, Analisador e Predição.  \n{e}")

        # Dados Brutos, Binarios, Governança, Metodologias
        try:
            st.divider()
            tabelaBasica = st.tabs(["📊 Dados Brutos", "🔢 Dados Binário", "🛡️ Governança", "📖 Metodologias"])
            with tabelaBasica[0]:
                # 1. Primeiro, criamos uma cópia da seleção para evitar o aviso de 'SettingWithCopyWarning'
                # e já invertemos a ordem com o .iloc[::-1]
                df_view = df_bruto_f[['Bola1', 'Bola2', 'Bola3', 'Bola4', 'Bola5', 'Bola6', 'Concurso', 'Data do Sorteio']].iloc[::-1].copy()

                # 2. Convertemos todos os nomes de colunas para string (resolve o UserWarning de mixed type)
                df_view.columns = df_view.columns.astype(str)

                # 3. Exibimos no Streamlit com a sintaxe de 2026
                st.dataframe(
                    df_view,
                    width="stretch",
                    hide_index=True
                )
            with tabelaBasica[1]:
                st.dataframe(df_bin_f.iloc[::-1], width="stretch", hide_index=True)
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
        except Exception as e:
            st.error(f"Erro em Dados brutos, Binarios, Governança e Metodologia.  \n{e}")

elif st.session_state["authentication_status"] is False:
    st.error('Usuário ou senha incorretos')
elif st.session_state["authentication_status"] is None:
    st.warning('Por favor, insira seu usuário e senha.')

st.markdown(f'<div class="footer-text notranslate">© {NOME_SISTEMA} {VERSAO} | 2026 | By: {PROGRAMADOR}</div>', unsafe_allow_html=True)
