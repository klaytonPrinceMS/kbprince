import streamlit as st
import random
import os

# 1. Configuração ÚNICA da Página
st.set_page_config(
    page_title="Pedra, Papel ou Tesoura",
    page_icon="✊",
    layout="centered",
    initial_sidebar_state='expanded',
    menu_items={"About": "https://klaytonprincems.github.io/site/"}
)

# 2. Estilo CSS Unificado e Organizado
st.markdown("""
    <style>
    /* Interface Geral e Limpeza */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}

    /* Títulos e Textos */
    .main-title {
        text-align: center;
        color: #ff4b4b;
        margin-top: -30px;
    }

    .emoji-grande { 
        font-size: 80px; 
        text-align: center; 
        margin: 0; 
        padding: 0; 
    }

    .label-jogador { 
        font-weight: bold; 
        text-align: center; 
        color: #888; 
    }

    /* Estilização Global dos Botões */
    .stButton>button {
        width: 100%;
        font-weight: bold;
        border-radius: 10px;
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        border-color: #ff4b4b;
        color: #ff4b4b;
        transform: scale(1.02);
    }

    /* Botões Grandes do Jogo */
    div.stButton > button {
        height: 100px;
        font-size: 20px !important;
    }

    /* Rodapé Customizado */
    .custom-footer {
        text-align: center;
        padding: 20px;
        margin-top: 50px;
        border-top: 1px solid #333;
        color: #288;
        font-size: 14px;
        line-height: 1.6;
    }

    .custom-footer a {
        color: #ff4b4b;
        text-decoration: none;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)


# --- FUNÇÕES DE PERSISTÊNCIA ---
def carregar_placar():
    arquivo = "resultadoJokenpo.txt"
    if not os.path.exists(arquivo):
        return 0, 0
    with open(arquivo, "r") as f:
        try:
            dados = f.read().split(",")
            return int(dados[0]), int(dados[1])
        except:
            return 0, 0


def salvar_placar(v, d):
    arquivo = "resultadoJokenpo.txt"
    with open(arquivo, "w") as f:
        f.write(f"{v},{d}")


# Inicialização do estado da sessão
if 'vitorias' not in st.session_state:
    v, d = carregar_placar()
    st.session_state.vitorias = v
    st.session_state.derrotas = d

if 'mostrar_resultado' not in st.session_state:
    st.session_state.mostrar_resultado = False

# --- LÓGICA DO JOGO ---
icones = {"Pedra": "✊", "Papel": "🖐️", "Tesoura": "✌️"}


def registrar_jogada(escolha_player):
    escolha_robo = random.choice(["Pedra", "Papel", "Tesoura"])
    st.session_state.escolha_usuario = escolha_player
    st.session_state.escolha_robo = escolha_robo
    st.session_state.mostrar_resultado = True

    if escolha_player != escolha_robo:
        if (escolha_player == "Pedra" and escolha_robo == "Tesoura") or \
                (escolha_player == "Papel" and escolha_robo == "Pedra") or \
                (escolha_player == "Tesoura" and escolha_robo == "Papel"):
            st.session_state.vitorias += 1
        else:
            st.session_state.derrotas += 1
        salvar_placar(st.session_state.vitorias, st.session_state.derrotas)


# --- INTERFACE ---
st.markdown("<h1 class='main-title'>🎮 Jokenpô</h1>", unsafe_allow_html=True)

col_v, col_d = st.columns(2)
col_v.metric("Minhas Vitórias", st.session_state.vitorias)
col_d.metric("Vitórias do Robô", st.session_state.derrotas)
st.divider()

if st.session_state.mostrar_resultado:
    user = st.session_state.escolha_usuario
    robo = st.session_state.escolha_robo

    col1, col_vs, col2 = st.columns([2, 1, 2])
    with col1:
        st.markdown(f"<p class='emoji-grande'>{icones[user]}</p>", unsafe_allow_html=True)
        st.markdown("<p class='label-jogador'>VOCÊ</p>", unsafe_allow_html=True)
    with col_vs:
        st.markdown("<h2 style='text-align: center; padding-top: 30px;'>VS</h2>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<p class='emoji-grande'>{icones[robo]}</p>", unsafe_allow_html=True)
        st.markdown("<p class='label-jogador'>ROBÔ</p>", unsafe_allow_html=True)

    if user == robo:
        st.warning(f"🤝 **Empate!** Ambos escolheram {user}.")
    elif (user == "Pedra" and robo == "Tesoura") or \
            (user == "Papel" and robo == "Pedra") or \
            (user == "Tesoura" and robo == "Papel"):
        st.success(f"🎉 **Você Ganhou!**")
    else:
        st.error(f"🤖 **Você Perdeu!**")

    if st.button("✨ JOGAR NOVAMENTE", use_container_width=True, type="primary"):
        st.session_state.mostrar_resultado = False
        st.rerun()
else:
    st.write("Escolha sua arma:")
    c1, c2, c3 = st.columns(3)
    c1.button(f"{icones['Pedra']}\nPedra", use_container_width=True, on_click=registrar_jogada, args=("Pedra",),
              key="btn_pedra")
    c2.button(f"{icones['Papel']}\nPapel", use_container_width=True, on_click=registrar_jogada, args=("Papel",),
              key="btn_papel")
    c3.button(f"{icones['Tesoura']}\nTesoura", use_container_width=True, on_click=registrar_jogada, args=("Tesoura",),
              key="btn_tesoura")

# --- ÁREA DE CONTROLE ---
st.write("---")
cn1, cn2 = st.columns(2)
with cn1:
    if st.button("🏠 Menu Principal", use_container_width=True):
        st.switch_page("Aplicativo.py")
with cn2:
    if st.button("🔄 Zerar Placar", use_container_width=True):
        st.session_state.vitorias = 0
        st.session_state.derrotas = 0
        salvar_placar(0, 0)
        st.rerun()

# --- RODAPÉ FINAL ---
st.markdown('''
    <div class="custom-footer">
        <span>© Copyright T! SOS Sistemas 2026</span><br>
        <span>Design by <a href="#">PRINCE, K.B</a></span>
    </div>
''', unsafe_allow_html=True)