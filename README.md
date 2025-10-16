# Plenário Preditivo 🗳️🤖

**Um sistema de Machine Learning para prever o resultado de votações na Câmara dos Deputados do Brasil, utilizando dados públicos.**

---

## 📜 Descrição do Projeto

O "Plenário Preditivo" é um projeto de Ciência de Dados que visa aumentar a transparência e a análise sobre o processo legislativo brasileiro. A aplicação coleta dados abertos da Câmara dos Deputados para treinar um modelo de machine learning capaz de prever o voto de cada deputado em proposições futuras.

O objetivo final não é apenas prever o resultado (Aprovado/Rejeitado), mas também entender quais fatores mais influenciam as decisões no plenário, oferecendo insights para jornalistas, pesquisadores e cidadãos interessados.

## ✨ Funcionalidades Principais

-   **Coleta de Dados Automatizada:** Busca e processa dados de deputados, proposições e votações via API da Câmara.
-   **Engenharia de Features:** Criação de variáveis preditivas, como taxa de governismo, fidelidade partidária e análise do texto das proposições com NLP.
-   **Modelo Preditivo:** Utilização de um modelo de Gradient Boosting (LightGBM) para prever o voto (`Sim`, `Não`, `Abstenção`) de cada deputado.
-   **Interpretabilidade (XAI):** Análise com SHAP para explicar os fatores que mais contribuem para cada previsão.
-   **Dashboard Interativo:** Uma interface web construída com Streamlit para visualizar as previsões de votações futuras e explorar os dados.

## 🛠️ Stack de Tecnologias

-   **Linguagem:** Python
-   **Análise de Dados:** Pandas, NumPy
-   **Machine Learning:** Scikit-learn, LightGBM
-   **Interpretabilidade:** SHAP
-   **Aplicação Web:** Streamlit
-   **Deploy:** Docker, Easypanel

## 🚀 Como Executar o Projeto

**1. Clone o repositório:**
```bash
git clone [https://github.com/anrportilho/plenario-preditivo.git](https://github.com/anrportilho/plenario-preditivo.git)
cd plenario-preditivo