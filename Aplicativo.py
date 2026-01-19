import streamlit as st

PROGRAMADOR = "PRINCE, K.B"
ICONE_PAGINA = "📰"
NOME_SISTEMA = "T! SOS Sistemas"
VERSAO = "TSistemas v1.0.20250104"
LINK_PESSOAL = "https://klaytonprincems.github.io/site/"
COR_QUENTE = "#FF4B4B"
COR_FRIO = "#007BFF"

# 1. Configuração ÚNICA da Página
st.set_page_config(
    page_title=NOME_SISTEMA,
    page_icon=ICONE_PAGINA,
    layout="wide",
    initial_sidebar_state='expanded',
    menu_items={"About": LINK_PESSOAL}
)

# 2. Estilo CSS Unificado e Organizado com BOTÕES UNIFORMES
st.markdown(f"""
    <style>
    /* Esconder elementos padrão */
    #MainMenu {{visibility: hidden;}}
    header {{visibility: hidden;}}
    footer {{visibility: hidden;}}

    /* Títulos */
    .main-title {{
        text-align: center; 
        color: {COR_QUENTE}; 
        font-weight: bold; 
        margin-bottom: 20px; 
        font-size: 30px;
    }}

    .subtitle {{
        text-align: center; 
        color: #666; 
        margin-bottom: 20px;
    }}

    /* ========== BOTÕES COM MÁXIMO DESTAQUE ========== */

    /* Todos os botões com tamanho FIXO e cores vibrantes */
    div.stButton > button {{
        width: 100% !important;
        height: 100px !important;
        min-height: 100px !important;
        max-height: 100px !important;

        /* CORES VIBRANTES E GRADIENTE */
        background: linear-gradient(135deg, {COR_QUENTE} 0%, #FF6B6B 100%) !important;
        color: white !important;

        font-size: 20px !important;
        font-weight: 900 !important;

        border-radius: 15px !important;
        border: 3px solid rgba(255, 255, 255, 0.3) !important;

        /* SOMBRA FORTE para dar profundidade */
        box-shadow: 0 8px 20px rgba(255, 75, 75, 0.4),
                    0 0 20px rgba(255, 75, 75, 0.2) !important;

        /* Centralizar texto */
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;

        white-space: normal !important;
        word-wrap: break-word !important;
        text-align: center !important;

        padding: 15px !important;

    /* Garantir que as colunas tenham altura igual */
    div[data-testid="column"] {{
        display: flex !important;
        flex-direction: column !important;
    }}

    /* ========== RODAPÉ ========== */
    .footer-text {{
        text-align: center; 
        padding: 20px; 
        color: #888; 
        font-size: 14px;
    }}

    .custom-footer {{
        text-align: center;
        padding: 20px;
        margin-top: 50px;
        border-top: 1px solid #333;
        color: #888;
        font-size: 14px;
        line-height: 1.6;
    }}

    .custom-footer a {{
        color: {COR_QUENTE};
        text-decoration: none;
        font-weight: bold;
    }}
    </style>
    """, unsafe_allow_html=True)

# 3. Cabeçalho
st.markdown(f"<h1 class='main-title'>{ICONE_PAGINA} {NOME_SISTEMA}</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Selecione uma ferramenta abaixo para iniciar</p>", unsafe_allow_html=True)
st.divider()

# 4. Layout de Menu em Colunas
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("🍀 Mega Sena", key="mega1"):
        st.switch_page("pages/Jogo_Mega_Sena_liberado.py")


    if st.button("⛏️ Mega Sena", key="mega2"):
        st.switch_page("pages/Jogo_Mega_Sena.py")

with col2:
    if st.button("📰 Notícias", key="news1"):
        st.switch_page("pages/Noticias.py")

    if st.button("📰 -- SOC --", key="news2"):
        st.switch_page("pages/Noticias.py")

with col3:
    if st.button("📈 Bovespa", key="bovespa"):
        st.switch_page("pages/Bovespa.py")

with col4:
    if st.button("✊ Jokenpô", key="jokenpo"):
        st.switch_page("pages/Jogo_Jokenpo.py")

    if st.button("✊ Hanói", key="hanoi"):
        st.switch_page("pages/Hanoi.py")


    if st.button("💣 Testando", key="test"):
        st.switch_page("pages/Testando.py")

st.divider()

# 5. Rodapé Final Unificado
st.markdown(
    f'<div class="footer-text notranslate">© {NOME_SISTEMA} {VERSAO} | 2026 | By: {PROGRAMADOR} | T! SOS Sistemas</div>',
    unsafe_allow_html=True
)