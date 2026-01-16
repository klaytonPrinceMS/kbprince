# 🛡️ SOC Incidentes (Security Operations Center)

<p align="center">
  <img src="https://img.shields.io/badge/Status-Monitoring-red?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Setor-Defesa_Cibernética-black?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Versão-1.0.20260109-blue?style=for-the-badge" />
</p>

## 📖 Visão Geral

O módulo `SOC` é a central de comando tático do ecossistema. Ele não apenas agrega informações, mas atua como uma camada de **Inteligência de Ameaças (Threat Intelligence)**. O objetivo principal é a detecção, classificação e resposta a incidentes de segurança cibernética, com foco especial em ativos críticos do governo e setor judiciário brasileiro.

## 🛠️ Stack Tecnológica do Módulo

| Tecnologia | Finalidade |
| :--- | :--- |
| ![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white) | Lógica de classificação heurística. |
| ![Power BI](https://img.shields.io/badge/Power_BI-F2C811?style=flat-square&logo=microsoftpowerbi&logoColor=black) | Dashboards analíticos de dados históricos. |
| ![Regex](https://img.shields.io/badge/Regex-Parsing-4285F4?style=flat-square) | Extração de vetores de ataque em manchetes. |
| ![NIST](https://img.shields.io/badge/Framework-NIST/CERT.br-green?style=flat-square) | Base metodológica para protocolos de resposta. |

---

## ⚙️ Inteligência Operacional

### 1. Classificação Heurística de Incidentes
O SOC utiliza um motor de análise de texto que classifica automaticamente as entradas em:

* **Tipos de Ameaça:** Ransomware, Vazamento de Dados, Defacement, Indisponibilidade (DDoS) e Invasão.
* **Níveis de Criticidade:**
    * 🔴 **Crítico:** Alvos como STF, PF, Ministérios e bases federais (Serpro/Dataprev).
    * 🟠 **Alto:** Prefeituras, Governos Estaduais e Tribunais.
    * 🟡 **Médio:** Vulnerabilidades gerais e ataques a empresas privadas.
    * 🟢 **Baixo:** Incidentes de baixo impacto ou informativos.

### 2. Dashboard LIVE FEED
Interface de monitoramento em tempo real com "Cards de Incidente" que exibem:
- **Badge de Criticidade:** Identificação visual imediata do risco.
- **Vetor de Ataque:** Categoria técnica da ameaça.
- **Timestamp de Detecção:** Horário da coleta pelo sistema.

### 3. Integração Governamental (BI)
O módulo consome dados de inteligência através de um iframe seguro do **Power BI**, permitindo cruzar os incidentes em tempo real com estatísticas históricas de ataques no Brasil.

---

## 📑 Protocolos de Resposta (IRP)

O sistema disponibiliza um guia de **Incidente Response Plan (IRP)** integrado, estruturado em 5 fases:

1.  **Identificação:** Verificação de anomalias e coleta de evidências.
2.  **Contenção:** Isolamento de redes (Wi-Fi/Cabo) e proteção de contas.
3.  **Erradicação:** Remoção de malwares e varredura de credenciais.
4.  **Recuperação:** Restauração de backups limpos e atualização de sistemas.
5.  **Lições Aprendidas:** Documentação e registro de BO (Boletim de Ocorrência).

---

## 🧬 Estrutura de Funções

#### `classificar_incidente(titulo)`
Analisa strings em busca de padrões de ataques e alvos estratégicos.
* **Lógica:** Baseada em pesos semânticos.
* **Exemplo:** Se "Ransomware" + "Ministério" -> Tipo: RANSOMWARE | Criticidade: CRÍTICO.

#### `buscar_incidentes(url_rss)`
Coleta dados de vetores de monitoramento específicos (Judiciário, Federal, Global).

---

## 🛡️ Governança e Integridade

A operação do Dashboard SOC é monitorada pelo protocolo **amche.hve**, que garante que os links de incidentes e as fontes de dados não foram adulterados (*Anti-Tampering*), assegurando que o operador tome decisões baseadas em informações íntegras.

## 🚨 Aviso de Responsabilidade
As informações exibidas são para fins de estudo estatístico e resposta a incidentes. O sistema não garante a neutralização automática de ataques em redes externas.


---
<div style="text-align: center;">
  <b>Desenvolvedor:</b> PRINCE, K.B <br>
  © 2026 | T! SOS Sistemas
</div>