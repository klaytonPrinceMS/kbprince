# [Clique aqui para ver o aplicativo on-line](https://kbprince1.streamlit.app/)


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


---
<div style="text-align: center;">
  <b>Desenvolvedor:</b> PRINCE, K.B <br>
  © 2026 | T! SOS Sistemas
</div>