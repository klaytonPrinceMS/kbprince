# 📈 BOVESPA Viewer (Market Intelligence)

<p align="center">
  <img src="https://img.shields.io/badge/Version-1.0.20260109-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Status-Live_Market-success?style=for-the-badge" />
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge" />
</p>

## 📖 Visão Geral

O `BOVESPA Viewer` é um módulo de inteligência financeira projetado para monitorar, analisar e visualizar o desempenho das principais ações da B3 (Bolsa de Valores brasileira). O sistema transforma dados brutos do mercado em insights visuais através de gráficos de candlestick, mapas de calor e rankings de performance em tempo real.

## 🛠️ Stack Tecnológica Padronizada

| Categoria | Tecnologia | Finalidade |
| :--- | :--- | :--- |
| **Linguagem** | ![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white) | Processamento lógico e cálculos de variação. |
| **Interface** | ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white) | Dashboard interativo e componentes UX. |
| **Dados** | ![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white) | Manipulação e estruturação de séries temporais. |
| **Finanças** | ![Yahoo Finance](https://img.shields.io/badge/Yahoo_Finance-6001D2?style=for-the-badge&logo=yahoo&logoColor=white) | Extração de dados históricos e cotações. |
| **Visualização** | ![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white) | Gráficos dinâmicos e Candlesticks. |
| **Design** | ![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=for-the-badge&logo=css3&logoColor=white) | Estilização avançada com Fontes JetBrains Mono. |

---

## ⚙️ Funcionalidades Principais

### 1. Monitoramento Multi-Ativo
O sistema rastreia simultaneamente os principais tickers da B3 (PETR4, VALE3, ITUB4, etc.), calculando:
* **Variação Percentual:** Comparativo entre abertura e fechamento do período selecionado.
* **Volume de Negociação:** Análise de liquidez dos ativos.
* **Classificação de Risco:** Identificação visual de altas (🟢), baixas (🔴) e estabilidade (⚪).

### 2. Análise Técnica Avançada
O módulo oferece ferramentas de visualização profissional:
* **Gráficos de Candlestick:** Detalhamento de Open, High, Low e Close (OHLC).
* **Mapas de Dispersão:** Cruzamento entre Volume vs. Variação (Estratégia de Bolha).
* **Rankings de Performance:** Top 10 Maiores Altas e Baixas com interface em *Cards* responsivos.



### 3. Otimização de Performance
* **Caching de Dados:** Utiliza `@st.cache_data` com TTL (Time-To-Live) configurado para 1 hora (ações) e 5 minutos (cotações), reduzindo o consumo de APIs externas.
* **UX Refinada:** Implementação de `backdrop-filter: blur(10px)` e animações de hover nos cards de ações.

---

## 🧬 Estrutura Lógica

### Algoritmo de Variação
A variação é calculada utilizando a fórmula de retorno simples:
$$V\% = \frac{P_{final} - P_{inicial}}{P_{inicial}} \times 100$$

### Classes de Estilização Customizadas
O sistema utiliza injeção de CSS para definir a gravidade dos movimentos:
* **`.positivo`**: Shadow verde para variações acima de 0%.
* **`.negativo`**: Shadow vermelho para variações abaixo de 0%.

---

## 🛡️ Segurança e Integridade

Integrado ao ecossistema **SIPP & SOC**, o `BOVESPA Viewer` utiliza o protocolo **amche.hve** para validar as requisições de rede feitas via biblioteca `requests`, garantindo que os feeds de dados financeiros não sofram ataques de *Man-in-the-Middle* (MitM).



## 📝 Como Operar

1.  Abra o **Control Center** no menu lateral.
2.  Defina o **Período de Análise** (de 1 dia a 5 anos).
3.  Utilize o **Filtro de Setor** para nichar sua análise (Financeiro, Energia, etc.).
4.  Expanda os **Detalhes** para acessar métricas fundamentais como P/E Ratio e Dividend Yield.

---
<p align="center">
  <b>Developer:</b> PRINCE, K.B <br>
  © 2026 | Bovespa Intelligence System
</p>

