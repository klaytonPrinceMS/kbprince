# ✊ Jokenpô Inteligente (Game Module)

<p align="center">
  <img src="https://img.shields.io/badge/Status-Divertimento-ff69b4?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Módulo-Interativo-yellow?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Versão-1.0.20260113-blue?style=for-the-badge" />
</p>

## 📖 Visão Geral

O módulo `Jokenpo` é um utilitário de entretenimento integrado ao ecossistema, projetado para demonstrar lógica de tomada de decisão e interação em tempo real com o usuário. Ele utiliza um gerador de números pseudo-aleatórios para simular as jogadas da CPU, oferecendo uma interface responsiva e amigável.

## 🛠️ Stack Tecnológica do Módulo

| Tecnologia | Finalidade |
| :--- | :--- |
| ![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white) | Lógica de comparação e controle de estados. |
| ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white) | Renderização da interface e botões de ação. |
| ![Random](https://img.shields.io/badge/Random-Library-000000?style=for-the-badge&logo=python&logoColor=white) | Geração de jogadas imprevisíveis para a IA. |

---

## ⚙️ Regras de Negócio e Lógica

O sistema implementa a lógica clássica de competição, processando as entradas conforme a matriz de vitória:

1.  **Pedra** vence **Tesoura**.
2.  **Tesoura** vence **Papel**.
3.  **Papel** vence **Pedra**.

### Funcionalidades:
* **Contador de Pontuação:** Mantém o placar em tempo real (Jogador vs CPU) durante a sessão.
* **Feedback Visual:** Utiliza emojis e componentes de texto do Streamlit para indicar o vencedor de cada rodada.
* **Botão de Reset:** Reinicializa os estados do jogo (Session State) para uma nova partida.

---

## 🧬 Estrutura do Código

### Gerenciamento de Estado
Para garantir que o placar não seja reiniciado a cada clique, o módulo utiliza o `st.session_state` do Streamlit:
* `st.session_state.vitorias_usuario`: Acumulador de vitórias do player.
* `st.session_state.vitorias_cpu`: Acumulador de vitórias da máquina.

### Fluxo de Execução
1. O usuário seleciona uma opção via botão.
2. A função `random.choice()` define a jogada da CPU.
3. A lógica condicional compara os resultados e atualiza o estado da sessão.
4. O resultado é exibido com mensagens de sucesso (`st.success`), aviso (`st.warning`) ou erro (`st.error`).

---

## 🛡️ Segurança e Integridade

Embora seja um módulo de lazer, a execução do código segue as mesmas diretrizes de segurança do portal principal. A consistência dos scripts e o carregamento seguro das bibliotecas são validados pelo protocolo interno, garantindo que o jogo opere sem vulnerabilidades de injeção de código.

## 📝 Como Jogar

1.  Navegue até a seção de "Jogos" no menu lateral.
2.  Escolha sua jogada entre **Pedra**, **Papel** ou **Tesoura**.
3.  Veja o resultado imediato e acompanhe o placar no topo da tela.


---
<div style="text-align: center;">
  <b>Desenvolvedor:</b> PRINCE, K.B <br>
  © 2026 | T! SOS Sistemas
</div>