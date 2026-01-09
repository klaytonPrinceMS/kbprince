import streamlit as st
import requests
from parsel import Selector
import base64
import re
import functools
from urllib.parse import quote
from datetime import datetime
import email.utils  # Para parsing eficiente de datas RFC 822

# --- 1. CONFIGURAÇÕES TÉCNICAS E ESTÉTICAS ---
NOME_SISTEMA = "Noticias"
VERSAO = "v1.0.20260109"
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

# Estilização CSS Superior
st.markdown(f"""
    <style>
    .main-title {{text-align: center; color: {COR_QUENTE}; font-weight: bold; margin-bottom: 20px;}}
    .stButton>button {{width: 100%; font-weight: bold; border-radius: 10px; height: 45px; transition: 0.3s;}}
    .stButton>button:hover {{border: 2px solid {COR_FRIO}; color: {COR_FRIO};}}
    .footer-text {{text-align: center; padding: 20px; color: #888; font-size: 14px;}}
    .crisp-box {{
        background-color: #f8f9fa; 
        padding: 18px; 
        border-radius: 12px; 
        border-left: 6px solid {COR_FRIO}; 
        margin-bottom: 15px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
    }}
    .news-title {{color: #1A1A1A; font-size: 19px; font-weight: bold; text-decoration: none;}}
    .news-title:hover {{color: {COR_FRIO};}}
    .news-source {{color: {COR_QUENTE}; font-size: 13px; font-weight: bold;}}
    .news-time {{color: #666; font-size: 12px;}}
    </style>
    """, unsafe_allow_html=True)


# --- 2. FUNÇÕES DO SISTEMA ---

def formatar_data(data_str):
    """Converte datas RFC 822 (Google News) para o padrão brasileiro."""
    try:
        data_tuple = email.utils.parsedate_tz(data_str)
        if data_tuple:
            dt_obj = datetime.fromtimestamp(email.utils.mktime_tz(data_tuple))
            return dt_obj.strftime("%d/%m/%Y %H:%M")
        return data_str
    except Exception:
        return data_str
@functools.lru_cache(maxsize=256)
def decodificar_url_google(url_google: str) -> str:
    """Decodifica a URL do Google News para obter o link direto."""
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
    """Realiza o web scraping com tratamento de timeout e erros."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
    }
    try:
        # Adicionado timeout para evitar travamentos
        response = requests.get(url_rss, headers=headers, timeout=12)
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
            data_formatada = formatar_data(data_pub)

            noticias.append({
                "titulo": titulo,
                "link": link_direto,
                "data": data_formatada,
                "fonte": fonte
            })
        return noticias
    except requests.exceptions.Timeout:
        st.error("⌛ A conexão com o servidor de notícias expirou. Tente atualizar.")
        return []
    except Exception as e:
        st.error(f"❌ Erro ao buscar notícias: {e}")
        return []


# --- 3. INTERFACE E LÓGICA ---
st.markdown(f"<h1 class='main-title'>📰 {NOME_SISTEMA}</h1>", unsafe_allow_html=True)

# --- 4. Menu Lateral ---
st.sidebar.header("🛠️ Configuração")
tema_busca = st.sidebar.text_input("🔍 Pesquisa Global", placeholder="Ex: Ransomware Brasil")

opcoes_rss = {
    "🛡️ Cybersegurança (Brasil)": "https://news.google.com/rss/search?q=cybersegurança+OR+cibersegurança+OR+ataque+hacker+Brasil+when:7d&hl=pt-BR&gl=BR&ceid=BR:pt-419",
    "🏢 Ataques Governamentais": "https://news.google.com/rss/search?q=ataque+hacker+prefeitura+OR+invasão+sistemas+governo+Brasil+when:30d&hl=pt-BR&gl=BR&ceid=BR:pt-419",
    "🔥 Principais Notícias": "https://news.google.com/rss?hl=pt-BR&gl=BR&ceid=BR:pt-419",
    "💻 Tecnologia": "https://news.google.com/rss/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRGRqTVhZU0FtVnVHZ0pIUWlnQVAB?hl=pt-BR&gl=BR&ceid=BR:pt-419",
    "📈 Negócios": "https://news.google.com/rss/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRGx6TVdZU0FtVnVHZ0pKVGlnQVAB?hl=pt-BR&gl=BR&ceid=BR:pt-419"
}

if tema_busca:
    categoria_nome = f"Resultado para: {tema_busca}"
    url_selecionada = f"https://news.google.com/rss/search?q={quote(tema_busca)}+when:7d&hl=pt-BR&gl=BR&ceid=BR:pt-419"
else:
    categoria_nome = st.sidebar.selectbox("Escolha uma Categoria", list(opcoes_rss.keys()))
    url_selecionada = opcoes_rss[categoria_nome]

if st.sidebar.button("🔄 ATUALIZAR FEED"):
    st.cache_data.clear()
    st.rerun()

# --- Pagina Principal ---
col_feed, col_info = st.columns([3, 1])

with col_feed:
    st.subheader(f"📍 Monitorando fontes globais de Cybersegurança")
    with st.spinner("Sincronizando feed..."):
        lista_noticias = buscar_noticias(url_selecionada)
        if lista_noticias:
            for noticia in lista_noticias:
                st.markdown(f"""
                <div class="crisp-box">
                    <a href="{noticia['link']}" target="_blank" class="news-title">{noticia['titulo']}</a><br>
                    <span class="news-source">{noticia['fonte']}</span> • 
                    <span class="news-time">📅 {noticia['data']}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("Nenhum registro encontrado para este tema no momento.")

with col_info:
    st.info("**Sobre o Monitor**")
    st.divider()
    st.caption(f"**Dev:** {PROGRAMADOR}")
    st.caption(f"**Build:** {VERSAO}")
    st.markdown(f"[Acessar Portfolio]({LINK_PESSOAL})")

st.markdown(f'<div class="footer-text">© {NOME_SISTEMA} | 2026 | Tecnologias Python com Streamlit</div>', unsafe_allow_html=True)