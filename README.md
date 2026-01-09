# 🍀 Mega Sena SIPP Advanced 

Este projeto é uma aplicação web interativa desenvolvida em **Streamlit** para a **Mineração de Dados** dos resultados oficiais da Mega Sena. Ele expande a análise estatística básica com algoritmos avançados de Machine Learning e Associação, seguindo a metodologia **SIPP** e **CRISP-DM**.


## 🚀 Configuração e Execução

Para configurar e executar o projeto em seu ambiente local, siga os passos abaixo:

### 1. Pré-requisitos

Certifique-se de ter o **Python 3.8+** instalado em seu sistema.

### 2. Instalação de Dependências

O projeto utiliza diversas bibliotecas para processamento de dados, visualização e mineração. Instale todas as dependências listadas no arquivo `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 3. Execução da Aplicação

Com as dependências instaladas, execute o aplicativo Streamlit a partir do terminal:
```bash
streamlit run Aplicativo.py
```

## 🐳 Como Usar via Docker

A utilização via Docker garante que todas as bibliotecas de Data Science (Pandas, Scikit-Learn, Mlxtend) funcionem corretamente, independentemente do seu sistema operacional.

### 1. Execução Rápida (Recomendado)
Se você não deseja baixar o código-fonte e quer apenas rodar a aplicação, utilize a imagem oficial hospedada no Docker Hub:

https://hub.docker.com/u/klaytonprince

```bash
docker run -p 8501:8501 klaytonprince/kbprince:latest
```


O aplicativo será aberto automaticamente em seu navegador padrão.

Em alguns casos a biblioteca Streamlit pode solicitar seu email via terminal para liberar o acesso


## ⚙️ Funcionalidades Avançadas (Novas Abas)

A versão introduz novas abas que elevam a análise para o nível de Mineração de Dados, conforme a literatura de **Silva, Peres e Boscarioli**.

| Aba                 | Funcionalidade | Algoritmos Utilizados | Foco Metodológico |
|:--------------------| :--- | :--- | :--- |
| **🔗 Associação**   | Descoberta de regras de co-ocorrência entre dezenas. Permite ajustar **Suporte** e **Confiança** mínimos. | **Apriori** e **FP-Growth** (`mlxtend`) | Regras de Associação (KDD) |
| **🤖 Predição MLP** | Tentativa de prever as dezenas com maior probabilidade de serem sorteadas no próximo concurso. | **Random Forest Classifier** (`scikit-learn`) | Classificação e Predição |
| **🧩 Clustering**   | Identificação de padrões ocultos nos sorteios. Permite agrupar concursos (padrões de sorteio) ou dezenas (afinidade). | **K-Means** (`scikit-learn`) | Agrupamento (Clustering) |
| **🔍 Analisador**   | Análise de jogos personalizados (4 a 15 dezenas), calculando Score SIPP, frequência histórica e ocorrências de premiação. | Estatística Descritiva e Score SIPP | Avaliação de Hipóteses |
| **🎲 Gerador**      | Geração de jogos baseada em predições do sistema e restrições definidas pelo usuário (pares, soma, etc.). | Heurística e Otimização | Implantação e Uso Prático |
| **📖 CRISP-DM**     | Documentação do projeto sob a ótica da metodologia **CRISP-DM** (Cross-Industry Standard Process for Data Mining). | N/A | Governança e Metodologia |

### Detalhes da Aba "🔗 Associação"

Esta aba permite aplicar algoritmos de Regras de Associação para descobrir quais dezenas tendem a sair juntas.

*   **Apriori / FP-Growth:** O usuário pode escolher o algoritmo de mineração de conjuntos de itens frequentes.
*   **Métricas:** O usuário pode definir o **Suporte Mínimo** (frequência mínima de ocorrência) e a **Confiança Mínima** (probabilidade de o consequente ocorrer dado o antecedente).
*   **Resultado:** Exibe as regras de associação encontradas, incluindo as métricas de **Suporte**, **Confiança** e **Lift** (indicador de força da regra).

### Detalhes da Aba "🤖 Predição ML"

Esta funcionalidade utiliza um modelo de Machine Learning para tentar identificar as dezenas mais prováveis de serem sorteadas.

*   **Modelo:** Random Forest Classifier.
*   **Lógica:** O modelo é treinado para prever a ocorrência de cada dezena no próximo sorteio, baseado em uma janela histórica de sorteios anteriores.
*   **Saída:** Apresenta as **Top N** dezenas com maior probabilidade de ocorrência.

### Detalhes da Aba "🧩 Clustering"

O K-Means é utilizado para agrupar dados, revelando padrões que não são óbvios na análise descritiva:

*   **Agrupamento de Concursos:** Identifica grupos de sorteios que compartilham características semelhantes (e.g., sorteios com baixa soma e alta paridade).
*   **Agrupamento de Dezenas:** Identifica grupos de dezenas que tendem a se comportar de forma correlacionada ao longo do tempo.

## 🛡️ Governança e Ética

O projeto mantém o compromisso com a transparência e a ética:

*   **Fonte de Dados:** Loterias Caixa (Dados Oficiais).
*   **Uso Ético:** O sistema é estritamente para fins de **estudo estatístico e mineração de dados**. Não há garantia de ganhos financeiros.
*   **Transparência Algorítmica:** Todos os algoritmos utilizados (SIPP, Apriori, FP-Growth, Random Forest, K-Means) são declarados e fazem parte da análise.

---
* Desenvolvido por PRINCE, K.B


