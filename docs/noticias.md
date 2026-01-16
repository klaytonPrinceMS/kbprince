# 📡 Monitor de Notícias (Threat Intelligence)

<p align="center">
  <img src="https://img.shields.io/badge/Status-Operacional-brightgreen?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Módulo-Inteligência-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Versão-1.0.20260109-blue?style=for-the-badge" />
</p>

## 📖 Visão Geral

O módulo `Noticias` atua como um agregador de inteligência em tempo real, focado em monitorar fontes globais e nacionais de **Cybersegurança**, **Ataques Hackers** e **Tecnologia**. Ele utiliza técnicas de *Web Scraping* e processamento de feeds RSS para manter o operador informado sobre as ameaças mais recentes do cenário digital.

## 🛠️ Stack Tecnológica do Módulo

| Tecnologia | Finalidade |
| :--- | :--- |
| ![Requests](https://img.shields.io/badge/Requests-005571?style=flat-square&logo=python&logoColor=white) | Requisições HTTP aos servidores de notícias. |
| ![Parsel](https://img.shields.io/badge/Parsel-Scrapy-orange?style=flat-square&logo=scrapy&logoColor=white) | Extração de dados (Parsing) de arquivos XML/RSS. |
| ![Base64](https://img.shields.io/badge/Base64-Encoding-black?style=flat-square) | Decodificação de URLs rastreadas pelo Google News. |
| ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=Streamlit&logoColor=white) | Interface de visualização em tempo real. |

---

## ⚙️ Funcionalidades Principais

### 1. Agregação Multi-Foco
O sistema permite monitorar diferentes vetores através de seletores pré-configurados ou busca global:
* **Cybersegurança (Brasil):** Foco em ataques hackers nacionais.
* **Ataques Governamentais:** Monitoramento específico de invasões a órgãos públicos.
* **Tecnologia e Negócios:** Tendências globais do setor.

### 2. Algoritmo de Decodificação de URL
O módulo implementa uma função avançada de decodificação (`decodificar_url_google`) que:
1.  Isola a parte codificada em **Base64** das URLs.
2.  Trata o *padding* dos bytes para evitar erros de decodificação.
3.  Utiliza **Regex (Expressões Regulares)** para extrair o link original da notícia, garantindo que o usuário acesse o portal de destino sem intermediários.

### 3. Otimização e Performance
* **LRU Cache:** Utiliza `@functools.lru_cache` para armazenar URLs já decodificadas, reduzindo o processamento repetitivo.
* **Tratamento de Datas:** Converte o padrão RFC 822 (servidor) para o padrão brasileiro (`dd/mm/aaaa hh:mm`).
* **Timeout Seguro:** Implementa limites de tempo (12s) nas requisições para evitar travamentos da interface caso o provedor esteja instável.

---

## 🧬 Estrutura do Código

### Funções Essenciais

#### `buscar_noticias(url_rss)`
Realiza o scraping do feed RSS.
* **Entrada:** URL do feed RSS.
* **Saída:** Lista de dicionários contendo `titulo`, `link`, `data` e `fonte`.

#### `formatar_data(data_str)`
Normaliza a data de publicação para o fuso horário e formato local.

---

## 🛡️ Segurança e Integridade

Assim como os demais módulos deste ecossistema, o monitoramento de notícias é validado e sincronizado com o protocolo interno, garantindo que as fontes consumidas passem pelos filtros de integridade do sistema central antes de serem exibidas no Dashboard.

## 📝 Como Usar

1.  Acesse o menu lateral do **Portal SIPP & SOC**.
2.  Selecione "Notícias" ou o ícone 📰.
3.  Escolha uma categoria pré-definida ou digite um termo de busca no campo **"Pesquisa Global"**.
4.  Clique no título de qualquer notícia para ser redirecionado via link direto decodificado.

---
<div style="text-align: center;">
  <b>Desenvolvedor:</b> PRINCE, K.B <br>
  © 2026 | T! SOS Sistemas
</div>