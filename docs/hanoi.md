# 🏰 Torre de Hanoi Master

Este documento descreve o funcionamento, a estrutura e as decisões de design do projeto **Torre de Hanoi**, desenvolvido em Python com o framework **Streamlit**.

---
## 📋 Sumário
1. [Visão Geral](#-visão-geral)
2. [Arquitetura de Estado](#-arquitetura-de-estado)
3. [Componentes de Interface](#-componentes-de-interface)
4. [Lógica de Jogo](#-lógica-de-jogo)
5. [Estilização e Estabilidade](#-estilização-e-estabilidade)
--

## 📌 Visão Geral
O **Torre de Hanoi** é uma implementação digital do quebra-cabeça matemático "Torre de Hanói". O sistema permite escolher dificuldades entre **3 e 20 discos** e gerencia recordes persistentes durante a sessão do usuário.

### Regras Implementadas:
- Apenas o disco do topo pode ser movido.
- Um disco maior nunca pode ser sobreposto a um disco menor.
- A vitória é contabilizada apenas quando todos os discos chegam à **Torre C**.

---
## ⚙️ Arquitetura de Estado
O jogo utiliza o `st.session_state` para garantir que os dados não sejam perdidos entre os reruns do Streamlit:

- **`tabuleiros`**: Dicionário `{ 'A': [], 'B': [], 'C': [] }` representando as pilhas.
- **`recordes`**: Armazena o melhor resultado de cada nível (`{ n_discos: min_movimentos }`).
- **`selecionado`**: Variável de controle para o fluxo "Pegar/Soltar".
- **`movimentos`**: Contador incremental de jogadas válidas.
---

## 🖥️ Componentes de Interface

### 1. Barra Lateral (Sidebar)
- **Slider de Dificuldade**: Define o `n` de discos (3 a 20).
- **Métrica de Recorde**: Exibe o `Best Score` dinamicamente conforme a dificuldade selecionada.
- **Botão Reiniciar**: Reseta o estado da sessão para o padrão inicial.

### 2. Área de Jogo
- **Colunas (`st.columns`)**: Três divisões verticais para as torres A, B e C.
- **Containers de Borda**: Espaço visual delimitado onde os discos são renderizados.
- **Botões Dinâmicos**: Alternam entre "PEGAR" e "SOLTAR" dependendo do estado de seleção.

---

## 🧠 Lógica de Jogo

### Movimentação
A função `mover_disco(origem, destino)` valida a jogada antes de alterar as listas. Se o movimento for ilegal, um `st.toast` (notificação flutuante) é disparado para informar o usuário sem quebrar o layout.

### Verificação de Vitória
A cada movimento bem-sucedido para a **Torre C**, o código verifica se o tamanho da lista é igual ao número de discos inicial. Em caso positivo:
1. A flag `venceu` torna-se `True`.
2. O recorde é atualizado se o número de movimentos atual for menor que o salvo anteriormente.
3. Disparam-se os balões comemorativos.

---

## 🎨 Estilização e Estabilidade
Para resolver problemas de "pulos" na tela (layout shift), foram aplicadas as seguintes técnicas de CSS:

- **Altura Mínima Fixa**: O container das torres possui `min-height: 500px`, garantindo que os botões de ação fiquem sempre na mesma linha, independente da quantidade de discos em cada torre.
- **Alinhamento na Base**: Utilização de `justify-content: flex-end` para que a pilha cresça de baixo para cima.
- **Preenchimento de Espaço**: Um loop gera `&nbsp;` (espaços vazios) para as posições não ocupadas por discos, mantendo a integridade visual.

---
