# Plenar.io Preditivo

**Uma ferramenta de Análise e Simulação de Votações na Câmara dos Deputados, desenvolvida com Machine Learning.**

---

## Objetivo do Projeto

O "Plenar.io Preditivo" é uma plataforma de Ciência de Dados de ponta a ponta, projetada para aumentar a transparência e a profundidade da análise sobre o processo legislativo brasileiro. O sistema utiliza dados públicos da API da Câmara dos Deputados para treinar um modelo preditivo com **92.8% de acurácia**, capaz de prever o voto de parlamentares.

O objetivo vai além da simples previsão, buscando fornecer insights sobre o comportamento político, a coesão partidária e os fatores que influenciam as decisões no plenário.

## Público-Alvo

Esta ferramenta foi desenvolvida para ser útil a um público variado, incluindo:

* **Jornalistas e Analistas Políticos:** Para enriquecer reportagens e análises com dados quantitativos sobre tendências de voto.
* **Assessores Parlamentares e Equipes de Relações Institucionais:** Para monitorar o alinhamento de bancadas e prever o resultado de pautas de interesse.
* **Cientistas Políticos e Pesquisadores:** Como uma fonte de dados estruturados e um modelo de base para estudos sobre o comportamento legislativo.
* **Cidadãos Engajados:** Para fiscalizar e entender melhor como seus representantes estão votando.

## Funcionalidades Principais

A aplicação é um dashboard interativo multi-abas construído com Streamlit, oferecendo diferentes níveis de análise:

1.  **Dashboard Executivo:** Uma visão macro com gráficos sobre o resultado geral das votações, o comportamento das bancadas (Governo, Oposição, Independente) e o alinhamento de partidos e estados.
2.  **Simulador de Voto Individual:** Permite a análise detalhada de um voto específico de um deputado em uma votação histórica, comparando a previsão do modelo com o voto real.
3.  **Análise de Deputados:** Uma ferramenta para explorar o perfil dos parlamentares, com filtros por partido e posição, exibindo a taxa histórica de alinhamento de cada um.
4.  **Análise de Votações:** Um resumo de todas as votações no dataset, permitindo identificar as pautas mais consensuais e as mais divisivas.
5.  **Previsão de Novas Pautas:** A funcionalidade principal, onde o usuário pode inserir a ementa de um projeto de lei futuro e o sistema prevê o placar, baseado no perfil histórico dos deputados.

## Stack de Tecnologias

* **Coleta e Análise de Dados:** Python, Pandas
* **Machine Learning:** Scikit-learn, LightGBM
* **Explicabilidade (Futuro):** SHAP
* **Aplicação Web e Visualização:** Streamlit, Plotly
* **Deploy:** Docker, Easypanel (Hetzner)

## Como Executar o Pipeline Completo

Para reproduzir o projeto do zero, desde a coleta de dados até a execução do dashboard, siga a ordem de execução dos scripts abaixo. Certifique-se de estar na pasta raiz do projeto com o ambiente virtual (`.venv`) ativado.

**1. Instalar Dependências:**
```bash
pip install -r requirements.txt
```

**2. Executar o Pipeline de Dados e Modelagem (na ordem):**

```bash
# Fase 1: Coleta de Dados Brutos
python -m src.data_collection.api_client
python -m src.data_collection.enrich_deputies_data
python -m src.data_collection.fetch_votings_data
python -m src.data_collection.enrich_votings_data

# Fase 2: Engenharia de Features
python -m src.feature_engineering.build_features
python -m src.feature_engineering.create_modeling_dataset
python -m src.feature_engineering.enrich_behavioral_features

# Fase 3: Treinamento do Modelo
python -m src.modeling.train_model
```

**3. Executar o Dashboard:**
```bash
streamlit run app/🔮_Placar_Preditivo.py
```

---
*Este projeto foi desenvolvido como um portfólio de Ciência de Dados, demonstrando um ciclo completo de desenvolvimento, desde a concepção e coleta de dados até a modelagem, avaliação e entrega de um produto final interativo.*