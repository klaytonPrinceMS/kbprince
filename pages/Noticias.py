import streamlit as st
import requests
from parsel import Selector
import base64
import re
import functools
from urllib.parse import quote

# --- 1. CONFIGURAÇÕES TÉCNICAS E ESTÉTICAS ---
NOME_SISTEMA = "Noticias"
VERSAO = "v0.0.20260109"
PROGRAMADOR = "PRINCE, K.B"
COR_QUENTE = "#FF4B4B"
COR_FRIO = "#007BFF"
LINK_PESSOAL = "https://klaytonprincems.github.io/site/"

st.set_page_config(
    page_title=NOME_SISTEMA,
    page_icon="📰",
    layout="wide",
    initial_sidebar_state='expanded',
    menu_items={"About": LINK_PESSOAL}
)

st.markdown(f"""
    <style>
    .main-title {{text-align: center; color: {COR_QUENTE}; font-weight: bold; margin-bottom: 20px;}}
    .stButton>button {{width: 100%; font-weight: bold; border-radius: 10px; height: 45px;}}
    .footer-text {{text-align: center; padding: 20px; color: #888; font-size: 14px;}}
    .crisp-box {{
        background-color: #f0f2f6; 
        padding: 20px; 
        border-radius: 10px; 
        border-left: 5px solid {COR_FRIO}; 
        margin-bottom: 15px;
    }}
    .news-title {{color: {COR_FRIO}; font-size: 18px; font-weight: bold; text-decoration: none;}}
    .news-source {{color: #555; font-size: 12px; font-style: italic;}}
    .news-time {{color: #888; font-size: 12px;}}
    </style>
    """, unsafe_allow_html=True)


# --- 2. FUNÇÕES DO SISTEMA ---

@functools.lru_cache(maxsize=128)
def decodificar_url_google(url_google: str) -> str:
    """Decodifica a URL do Google News para obter o link original da notícia."""
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

def buscar_noticias(url_rss):
    """Realiza o web scraping de notícias do Google News."""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
    try:
        response = requests.get(url_rss, headers=headers)
        response.raise_for_status()
        selector = Selector(text=response.text, type="xml")
        items = selector.xpath("//item")
        noticias = []
        for item in items[:15]:
            titulo = item.xpath("title/text()").get()
            link_google = item.xpath("link/text()").get()
            data_pub = item.xpath("pubDate/text()").get()
            fonte = item.xpath("source/text()").get()
            link_direto = decodificar_url_google(link_google)
            noticias.append({"titulo": titulo, "link": link_direto, "data": data_pub, "fonte": fonte})
        return noticias
    except Exception as e:
        st.error(f"Erro ao buscar notícias: {e}")
        return []

# --- 3. INTERFACE E LÓGICA ---
st.markdown(f"<h1 class='main-title'>📰 {NOME_SISTEMA}</h1>", unsafe_allow_html=True)

# --- 4. Menu Lateral ---
st.sidebar.header("Configuração")
tema_busca = st.sidebar.text_input("🔍 Pesquisar Tema", placeholder="Ex: Inteligência Artificial")

opcoes_rss = {
    "🛡️ Cybersegurança (Brasil)": "https://news.google.com/rss/search?q=cybersegurança+OR+cibersegurança+OR+ataque+hacker+Brasil+when:7d&hl=pt-BR&gl=BR&ceid=BR:pt-419",
    "🌐 Cybersecurity (Mundo)": "https://news.google.com/rss/search?q=cybersecurity+OR+data+breach+OR+ransomware+when:7d&hl=en-US&gl=US&ceid=US:en",
    "🏢 Ataques a Prefeituras (BR)": "https://news.google.com/rss/search?q=ataque+hacker+prefeitura+OR+invasão+sistemas+governo+Brasil+when:30d&hl=pt-BR&gl=BR&ceid=BR:pt-419",
    "🔥 Principais Notícias": "https://news.google.com/rss?hl=pt-BR&gl=BR&ceid=BR:pt-419",
    "💻 Tecnologia": "https://news.google.com/rss/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRGRqTVhZU0FtVnVHZ0pIUWlnQVAB?hl=pt-BR&gl=BR&ceid=BR:pt-419",
    "📈 Negócios": "https://news.google.com/rss/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRGx6TVdZU0FtVnVHZ0pKVGlnQVAB?hl=pt-BR&gl=BR&ceid=BR:pt-419"
}

if tema_busca:
    categoria = f"Busca: {tema_busca}"
    url_selecionada = f"https://news.google.com/rss/search?q={quote(tema_busca)}&hl=pt-BR&gl=BR&ceid=BR:pt-419"
else:
    categoria = st.sidebar.selectbox("Escolha a Categoria", list(opcoes_rss.keys()))
    url_selecionada = opcoes_rss[categoria]

if st.sidebar.button("🔄 ATUALIZAR"):
    st.rerun()

# --- Pagina Principal ---
tabs1 = st.tabs([f"Notícias: {categoria}"])
with tabs1[0]:
    with st.spinner(f"Buscando notícias de {categoria}..."):
        lista_noticias = buscar_noticias(url_selecionada)
        if lista_noticias:
            for noticia in lista_noticias:
                st.markdown(f"""
                <div class="crisp-box">
                    <a href="{noticia['link']}" target="_blank" class="news-title">{noticia['titulo']}</a><br>
                    <span class="news-source">Fonte: {noticia['fonte']}</span> |
                    <span class="news-time">{noticia['data']}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("Nenhuma notícia encontrada.")

try:
    tabsInformacao = st.tabs(["", "Informações", "Detalhes"])
    with tabsInformacao[0]: st.info("Fique por dentro das últimas ameaças e tendências de segurança.")
    with tabsInformacao[1]: st.info(f"Sistema: {NOME_SISTEMA}\\nVersão: {VERSAO}\\nProgramador: {PROGRAMADOR}")
    with tabsInformacao[2]: st.write("Monitoramento especializado em Cybersegurança e ataques a órgãos públicos.")
except Exception as e:
    st.error(f"Erro:  \\n{e}")

st.markdown(f'<div class="footer-text notranslate">© {NOME_SISTEMA} {VERSAO} | 2026 | By: {PROGRAMADOR}</div>', unsafe_allow_html=True)
