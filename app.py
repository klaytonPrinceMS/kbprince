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

# 4. Layout de Menu em Colunas
col1, col2 = st.columns(2)
with col1:
    st.subheader("✊ Jokenpô")
    st.write("Desafie o computador no clássico Pedra, Papel e Tesoura.")
    if st.button("  JOGAR JOKENPÔ  "):
        st.switch_page("pages/Jogo_Jokenpo.py")
with col1:
    st.subheader("🍀 Mega Sena")
    st.write("Mineração e Analise de Dados.")
    if st.button("Analise Mega Sena"):
        st.switch_page("pages/Jogo_Mega_Sena.py")
with col2:
    st.subheader("Testando")
    st.write("Testes.")
    if st.button("    Testando    "):
        st.switch_page("pages/Testando.py")
with col2:
    st.subheader("🍀 Extras")
    st.write("Extras.")
    if st.button("     Extras     "):
        st.switch_page("pages/zzz_extras.py")
st.divider()



# 5. Rodapé Final Unificado
st.markdown(f'<div class="footer-text notranslate">© {NOME_SISTEMA} {VERSAO} | 2026 | By: {PROGRAMADOR} | T! SOS Sistemas</div>', unsafe_allow_html=True)


