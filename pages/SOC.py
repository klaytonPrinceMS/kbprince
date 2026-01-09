import streamlit as st
import requests
from parsel import Selector
import base64
import re
import functools
from urllib.parse import quote
from datetime import datetime
import streamlit.components.v1 as components

# --- 1. CONFIGURAÇÕES TÉCNICAS E ESTÉTICAS (ESTILO SOC) ---
NOME_SISTEMA = "SOC Incidentes"
VERSAO = "v1.0.20260109"
ICONE_APP="🛡️"
PROGRAMADOR = "PRINCE, K.B"
COR_CRITICO = "#FF0000"
COR_ALTO = "#FF4B4B"
COR_MEDIO = "#FFA500"
COR_BAIXO = "#00FF00"
COR_FUNDO_SOC = "#0E1117"
URL_POWER_BI = "https://app.powerbi.com/view?r=eyJrIjoiYzViMzUyYWUtZGE4Ny00NjczLWFlZTYtYTU3Y2VlOTgzMDQ4IiwidCI6IjU2YzFlMmZiLTg3YzEtNGRlMC1hNmFjLWQwNTY2YzA4Y2U2NiJ9"

st.set_page_config(
    page_title=NOME_SISTEMA,
    page_icon=ICONE_APP,
    layout="wide",
    initial_sidebar_state='expanded'
)

# CSS Avançado para Estilo SOC
st.markdown(f"""
    <style>
    .main {{background-color: {COR_FUNDO_SOC}; color: white;}}
    .main-title {{
        text-align: center; 
        color: {COR_ALTO}; 
        font-family: 'Courier New', Courier, monospace;
        font-weight: bold; 
        text-transform: uppercase;
        letter-spacing: 3px;
        border-bottom: 2px solid {COR_ALTO};
        padding-bottom: 10px;
        margin-bottom: 30px;
    }}
    .stTabs [data-baseweb="tab-list"] {{gap: 24px;}}
    .stTabs [data-baseweb="tab"] {{
        height: 50px;
        white-space: pre-wrap;
        background-color: #1E1E1E;
        border-radius: 5px 5px 0px 0px;
        color: #888;
        font-weight: bold;
    }}
    .stTabs [aria-selected="true"] {{background-color: #262730; color: {COR_ALTO} !important;}}

    .incident-card {{
        background-color: #1E1E1E; 
        padding: 20px; 
        border-radius: 5px; 
        border-left: 10px solid;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }}
    .critico {{border-left-color: {COR_CRITICO};}}
    .alto {{border-left-color: {COR_ALTO};}}
    .medio {{border-left-color: {COR_MEDIO};}}
    .baixo {{border-left-color: {COR_BAIXO};}}

    .news-title {{color: #FFFFFF; font-size: 18px; font-weight: bold; text-decoration: none;}}
    .news-title:hover {{color: {COR_ALTO};}}
    .news-meta {{color: #AAAAAA; font-size: 12px; margin-top: 5px;}}
    .badge {{
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 10px;
        font-weight: bold;
        text-transform: uppercase;
        margin-right: 5px;
    }}
    .footer-text {{text-align: center; padding: 20px; color: #555; font-size: 12px; font-family: monospace;}}
    </style>
    """, unsafe_allow_html=True)


# --- 2. FUNÇÕES DE INTELIGÊNCIA ---

@functools.lru_cache(maxsize=256)
def decodificar_url_google(url_google: str) -> str:
    try:
        if not url_google.startswith("https://news.google.com/rss/articles/"):
            return url_google
        encoded_part = url_google.split("/")[-1].split("?")[0]
        padding = len(encoded_part) % 4
        if padding: encoded_part += "=" * (4 - padding)
        decoded_bytes = base64.urlsafe_b64decode(encoded_part)
        decoded_str = decoded_bytes.decode('latin1', errors='ignore')
        match = re.search(r'https?://[^\s\x00-\x1f\x7f-\xff]+', decoded_str)
        return match.group(0) if match else url_google
    except Exception:
        return url_google


def classificar_incidente(titulo):
    """Classifica o incidente por tipo e criticidade baseada em palavras-chave."""
    titulo_lower = titulo.lower()

    # Classificação de Tipo
    tipo = "OUTROS"
    if any(x in titulo_lower for x in ["ransomware", "sequestro", "bloqueio"]):
        tipo = "RANSOMWARE"
    elif any(x in titulo_lower for x in ["vazamento", "exposição", "dados expostos", "lgpd"]):
        tipo = "VAZAMENTO"
    elif any(x in titulo_lower for x in ["pichação", "defacement", "alteração"]):
        tipo = "DEFACEMENT"
    elif any(x in titulo_lower for x in ["indisponibilidade", "fora do ar", "instabilidade", "ddos"]):
        tipo = "INDISPONIBILIDADE"
    elif any(x in titulo_lower for x in ["invasão", "hacker", "acesso indevido"]):
        tipo = "INVASÃO"

    # Classificação de Criticidade
    criticidade = "baixo"
    if any(x in titulo_lower for x in ["ministério", "stf", "tse", "pf", "federal", "serpro", "dataprev"]):
        criticidade = "critico"
    elif any(x in titulo_lower for x in ["prefeitura", "governo", "estado", "tribunal", "justiça"]):
        criticidade = "alto"
    elif any(x in titulo_lower for x in ["ataque", "hacker", "vulnerabilidade"]):
        criticidade = "medio"

    return tipo, criticidade


def buscar_incidentes(url_rss):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
    try:
        response = requests.get(url_rss, headers=headers)
        response.raise_for_status()
        selector = Selector(text=response.text, type="xml")
        items = selector.xpath("//item")

        incidentes = []
        for item in items[:20]:
            titulo = item.xpath("title/text()").get()
            link_google = item.xpath("link/text()").get()
            data_pub = item.xpath("pubDate/text()").get()
            fonte = item.xpath("source/text()").get()

            tipo, criticidade = classificar_incidente(titulo)
            link_direto = decodificar_url_google(link_google)

            incidentes.append({
                "titulo": titulo,
                "link": link_direto,
                "data": data_pub,
                "fonte": fonte,
                "tipo": tipo,
                "criticidade": criticidade
            })
        return incidentes
    except Exception as e:
        st.error(f"ERRO NA COLETA DE DADOS: {e}")
        return []


# --- 3. INTERFACE PRINCIPAL ---
st.markdown(f"<h1 class='main-title'>🛡️ {NOME_SISTEMA}</h1>", unsafe_allow_html=True)

# --- 4. SIDEBAR (CONTROLE DE OPERAÇÕES) ---
st.sidebar.image("https://img.icons8.com/fluency/96/shield.png", width=80)
st.sidebar.title("OPERATIONS CENTER")
st.sidebar.markdown(f"**Status:** <span style='color:{COR_BAIXO}'>ONLINE</span>", unsafe_allow_html=True)
st.sidebar.markdown(f"**Data:** {datetime.now().strftime('%d/%m/%Y %H:%M')}")

st.sidebar.divider()

# Filtros de Monitoramento
st.sidebar.subheader("Filtros de Inteligência")
tema_custom = st.sidebar.text_input("🔍 Busca por Alvo Específico", placeholder="Ex: Prefeitura de São Paulo")

opcoes_monitoramento = {
    "🚨 Incidentes Críticos (Gov Federal)": "https://news.google.com/rss/search?q=ataque+hacker+OR+ransomware+OR+vazamento+federal+OR+ministério+OR+serpro+OR+dataprev+when:7d&hl=pt-BR&gl=BR&ceid=BR:pt-419",
    "🏛️ Monitoramento de Prefeituras": "https://news.google.com/rss/search?q=ataque+hacker+prefeitura+OR+invasão+sistemas+prefeitura+Brasil+when:7d&hl=pt-BR&gl=BR&ceid=BR:pt-419",
    "⚖️ Judiciário e Tribunais": "https://news.google.com/rss/search?q=ataque+hacker+tribunal+OR+tj+OR+trf+OR+stj+when:30d&hl=pt-BR&gl=BR&ceid=BR:pt-419",
    "🌐 Ameaças Globais (Cyber)": "https://news.google.com/rss/search?q=cybersecurity+threat+OR+critical+vulnerability+OR+zero-day+when:7d&hl=en-US&gl=US&ceid=US:en"
}

if tema_custom:
    categoria_ativa = f"ALVO: {tema_custom}"
    url_ativa = f"https://news.google.com/rss/search?q={quote(tema_custom)}+ataque+OR+hacker+OR+segurança+when:30d&hl=pt-BR&gl=BR&ceid=BR:pt-419"
else:
    categoria_ativa = st.sidebar.selectbox("Selecione o Vetor de Monitoramento", list(opcoes_monitoramento.keys()))
    url_ativa = opcoes_monitoramento[categoria_ativa]

if st.sidebar.button("⚡ REFRESH SYSTEM"):
    st.rerun()

# --- 5. DASHBOARD E ABAS ---
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Nível de Alerta", "ELEVADO", delta="24h", delta_color="inverse")
with col2:
    st.metric("Incidentes Detectados", "20+", delta="5 novos")
with col3:
    st.metric("Tempo de Resposta", "1.2s", delta="0.1s")

tabs = st.tabs(["📡 LIVE FEED","DADOS PF" ,"📊 ANÁLISE DE RISCO", "📖 PROTOCOLOS"])

with tabs[0]:
    st.subheader(f"Monitoramento Ativo: {categoria_ativa}")
    with st.spinner("Sincronizando com redes de inteligência..."):
        dados = buscar_incidentes(url_ativa)

        if dados:
            for inc in dados:
                # Badge de Criticidade
                cor_badge = {"critico": COR_CRITICO, "alto": COR_ALTO, "medio": COR_MEDIO, "baixo": COR_BAIXO}[
                    inc['criticidade']]

                st.markdown(f"""
                <div class="incident-card {inc['criticidade']}">
                    <span class="badge" style="background-color: {cor_badge}; color: white;">{inc['criticidade']}</span>
                    <span class="badge" style="background-color: #444; color: white;">{inc['tipo']}</span>
                    <br><br>
                    <a href="{inc['link']}" target="_blank" class="news-title">{inc['titulo']}</a>
                    <div class="news-meta">
                        📡 Fonte: {inc['fonte']} | 🕒 Detectado em: {inc['data']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("Nenhum incidente detectado nos parâmetros atuais.")
with tabs[1]:
    st.subheader("Análise de Dados Governamentais (Power BI)")
    # Integrando o Power BI via Iframe
    components.iframe(URL_POWER_BI, height=600, scrolling=True)


with tabs[2]:
    st.subheader("Distribuição de Ameaças")
    if 'dados' in locals() and dados:
        import pandas as pd

        df = pd.DataFrame(dados)
        st.bar_chart(df['tipo'].value_counts())
        st.write("### Resumo de Criticidade")
        st.table(df['criticidade'].value_counts())
    else:
        st.info("Aguardando dados para análise estatística.")
with tabs[3]:
    st.subheader("Protocolos de Resposta a Incidentes para Usuários Domésticos e Empresas Sem um Centro de Respostas")

    st.markdown("""
    Este protocolo segue as melhores práticas do **CERT.br** e do **NIST**, adaptado para uma resposta rápida sem necessidade de infraestrutura complexa:

    1. **Identificação**: 
        * Verifique comportamentos anômalos (lentidão extrema, arquivos renomeados, logins em horários estranhos).
        * Valide se a ameaça é real ou um *hoax* (boato).
        * **Ação:** Tire fotos da tela ou prints de mensagens de erro/ameaça antes de qualquer ação.

    2. **Contenção**:
        * **Curto Prazo:** Não desligue os equipamentos, repito NÂO desligue.
        * **Curto Prazo:** Desconecte o cabo de rede ou desligue o Wi-Fi imediatamente.
        * **Contas:** Acesse suas contas (e-mail, bancos) de um dispositivo **limpo** e troque as senhas. Ative a Autenticação de Dois Fatores (MFA).
        * **Preservação:** Não formate o disco imediatamente; as evidências residem nos logs e arquivos temporários.

    3. **Erradicação**:
        * Execute uma varredura completa com antivírus atualizado e ferramentas de remoção de malware (ex: Malwarebytes).
        * Identifique e remova contas de usuários criadas recentemente ou desconhecidas.
        * Revise permissões de aplicativos e extensões de navegador.

    4. **Recuperação**:
        * Restaure arquivos apenas de backups que você tenha certeza que foram feitos **antes** do ataque.
        * Atualize todos os softwares, sistemas operacionais e firmware de roteadores.
        * Monitore o tráfego de rede e o uso de CPU por alguns dias para garantir que a ameaça cessou.

    5. **Lições Aprendidas e Registro**:
        * **Registro Legal:** Em caso de perda financeira ou invasão de privacidade, registre um **Boletim de Ocorrência Online**.
        * **Notificação:** Informe seus contatos caso seu e-mail tenha sido usado para espalhar vírus.
        * **Prevenção:** Documente o que causou o ataque (ex: um anexo de e-mail ou link falso) para evitar a reincidência.
    """)

    st.warning("⚠️ **Dica de Segurança:** Nunca utilize o mesmo dispositivo atacado para trocar senhas bancárias antes de garantir que ele foi totalmente limpo ou formatado.")

    st.info("Consulte o Guia do CTIR Gov para mais detalhes sobre conformidade governamental.")



# --- 6. FOOTER ---
st.markdown(f'<div class="footer-text">SYSTEM SECURE | {VERSAO} | ENCRYPTED CONNECTION | BY: {PROGRAMADOR}</div>',
            unsafe_allow_html=True)
