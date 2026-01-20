import streamlit as st
import basedosdados as bd
import pandas as pd
import plotly.express as px
import basedosdados
# Configuração da Página
st.set_page_config(page_title="Evolução do Emprego Brasil", layout="wide")

st.title("📊 Evolução do Emprego (RAIS/CAGED)")
st.markdown("Analise a geração de empregos e salários usando dados da **Base dos Dados**.")


# Função para carregar dados (com cache para não gastar créditos à toa)
@st.cache_data
def carregar_dados_cidade(estado, cidade):
    # Exemplo de query para a RAIS (ajustar conforme as tabelas da BD)
    query = f"""
    SELECT ano, sigla_uf, id_municipio, qtde_vinculos_ativos, valor_remuneracao_media_nominal
    FROM `basedosdados.br_me_rais.microdados_vinculos`
    WHERE sigla_uf = '{estado}' 
    AND id_municipio = (SELECT id_municipio FROM `basedosdados.br_bd_diretorios_brasil.municipio` WHERE nome = '{cidade}' LIMIT 1)
    LIMIT 1000
    """
    return bd.read_sql(query, billing_project_id="TEU-PROJECT-ID-AQUI")


# Sidebar para Filtros
with st.sidebar:
    st.header("Filtros")
    estado_alvo = st.selectbox("Selecione o Estado (UF):", ["SP", "RJ", "MG", "BA", "RS", "SC"])  # Adicionar todos
    cidade_alvo = st.text_input("Digite o nome da cidade:", "São Paulo")
    botao_buscar = st.button("Analisar Dados")

if botao_buscar:
    with st.spinner('Consultando a Base dos Dados...'):
        try:
            df = carregar_dados_cidade(estado_alvo, cidade_alvo)

            if df.empty:
                st.warning("Nenhum dado encontrado para essa combinação.")
            else:
                # Layout de Colunas para Métricas
                col1, col2 = st.columns(2)
                col1.metric("Vínculos Ativos (Média)", int(df['qtde_vinculos_ativos'].mean()))
                col2.metric("Remuneração Média", f"R$ {df['valor_remuneracao_media_nominal'].mean():.2f}")

                # Gráfico de Evolução
                fig = px.line(df.sort_values('ano'), x='ano', y='qtde_vinculos_ativos',
                              title=f"Evolução de Vínculos Ativos em {cidade_alvo}")
                st.plotly_chart(fig, use_container_width=True)

                st.dataframe(df)
        except Exception as e:
            st.error(f"Erro ao conectar: {e}")