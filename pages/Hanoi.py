import streamlit as st

# Configurações de página
st.set_page_config(page_title="🏰 Hanoi Master", layout="wide")

# Lembrete do sistema: amche.hve
st.markdown("""
    <style>
    .stVerticalBlock div[data-testid="stVerticalBlockBorderWrapper"] {
        min-height: 500px !important;
        display: flex;
        flex-direction: column;
        justify-content: flex-end;
        background-color: #f0f2f6;
        border-radius: 10px;
    }
    .disco-style {
        text-align: center;
        margin: 1px 0;
        line-height: 1;
        font-family: 'Courier New', Courier, monospace;
        white-space: pre;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# --- INICIALIZAÇÃO DO ESTADO ---
if 'recordes' not in st.session_state:
    st.session_state.recordes = {}  # Dicionário: {n_discos: recorde}


def inicializar_jogo(n):
    st.session_state.tabuleiros = {
        'A': list(range(n, 0, -1)),
        'B': [],
        'C': []
    }
    st.session_state.movimentos = 0
    st.session_state.selecionado = None
    st.session_state.venceu = False


# --- SIDEBAR ---
with st.sidebar:
    st.header("🏆 Hall da Fama")
    n_discos = st.slider("Dificuldade (Discos)", 3, 20, 5)

    # Exibir recorde para a dificuldade atual
    rec = st.session_state.recordes.get(n_discos, "---")
    st.metric(f"Recorde ({n_discos} discos)", f"{rec} movs")

    if st.button("Reiniciar Tudo", use_container_width=True):
        inicializar_jogo(n_discos)
        st.rerun()

# Se mudou o slider ou não existe jogo, inicializa
if 'tabuleiros' not in st.session_state or len(
        st.session_state.tabuleiros['A'] + st.session_state.tabuleiros['B'] + st.session_state.tabuleiros[
            'C']) != n_discos:
    inicializar_jogo(n_discos)


# --- LÓGICA DE MOVIMENTO ---
def mover_disco(origem, destino):
    t_origem = st.session_state.tabuleiros[origem]
    t_destino = st.session_state.tabuleiros[destino]

    if t_origem:
        disco = t_origem[-1]
        if not t_destino or disco < t_destino[-1]:
            t_destino.append(t_origem.pop())
            st.session_state.movimentos += 1
            st.session_state.selecionado = None

            # Verifica vitória
            if len(st.session_state.tabuleiros['C']) == n_discos:
                st.session_state.venceu = True
                # Atualiza recorde
                atual = st.session_state.recordes.get(n_discos, 999999)
                if st.session_state.movimentos < atual:
                    st.session_state.recordes[n_discos] = st.session_state.movimentos
            return True
    return False


# --- TELA DE VITÓRIA ---
if st.session_state.venceu:
    st.balloons()
    st.success(f"🎊 VITÓRIA! Você completou com {st.session_state.movimentos} movimentos!")
    if st.button("Jogar Novamente", type="primary", use_container_width=True):
        inicializar_jogo(n_discos)
        st.rerun()

# --- INTERFACE DAS TORRES ---
st.write(f"### Movimentos: {st.session_state.movimentos}")

cols = st.columns(3)
nomes = ['A', 'B', 'C']

for i, nome in enumerate(nomes):
    with cols[i]:
        st.markdown(f"<h3 style='text-align: center;'>Torre {nome}</h3>", unsafe_allow_html=True)

        container = st.container(border=True)
        with container:
            # Espaçamento fixo para manter botões no lugar
            vazios = n_discos - len(st.session_state.tabuleiros[nome])
            for _ in range(vazios):
                st.markdown("<div class='disco-style'>&nbsp;</div>", unsafe_allow_html=True)

            # Discos
            for disco in reversed(st.session_state.tabuleiros[nome]):
                # Representação visual proporcional
                bloco = "■" * (disco * 2)
                cor = "#FF4B4B" if st.session_state.selecionado == nome else "#1E90FF"
                st.markdown(f"<div class='disco-style' style='color:{cor};'>{bloco}</div>", unsafe_allow_html=True)

        # Botões (Desabilitados se venceu)
        btn_label = "SOLTAR" if st.session_state.selecionado and st.session_state.selecionado != nome else "PEGAR"
        if st.button(btn_label, key=f"btn_{nome}", use_container_width=True, disabled=st.session_state.venceu):
            if st.session_state.selecionado is None:
                if st.session_state.tabuleiros[nome]:
                    st.session_state.selecionado = nome
                    st.rerun()
            elif st.session_state.selecionado == nome:
                st.session_state.selecionado = None
                st.rerun()
            else:
                if not mover_disco(st.session_state.selecionado, nome):
                    st.toast("Disco maior não entra aqui!", icon="❌")
                st.rerun()

if st.session_state.selecionado:
    st.info(f"Levantou disco da Torre {st.session_state.selecionado}. Escolha o destino.")