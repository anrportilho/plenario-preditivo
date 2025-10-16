# app/dashboard.py

import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- Configuração da Página e Carregamento de Dados ---

st.set_page_config(page_title="Plenário Preditivo", page_icon="🗳️", layout="wide")


@st.cache_data
def load_artifacts():
    """Carrega todos os artefatos do modelo e os dados necessários."""
    try:
        model = joblib.load('models/lgbm_model.joblib')
        encoder = joblib.load('models/label_encoder.joblib')
        feature_columns = joblib.load('models/feature_columns.joblib')
        data = pd.read_parquet('data/processed/modeling_dataset_enriched.parquet')
        return model, encoder, feature_columns, data
    except FileNotFoundError:
        st.error("Artefatos do modelo não encontrados. Por favor, execute o pipeline de scripts de 'src/' primeiro.")
        return None, None, None, None


model, encoder, feature_columns, df = load_artifacts()


# --- Funções de Lógica da Aplicação ---

def predict_vote(deputy_id, voting_id):
    """
    Prepara os dados para um deputado e uma votação específicos e retorna a previsão do modelo.
    """
    if df is None:
        return None, None

    instance = df[(df['id_deputado'] == deputy_id) & (df['id_votacao'] == voting_id)]

    if instance.empty:
        st.warning("Não foram encontrados dados para a combinação de deputado e votação selecionada.")
        return None, None

    # Prepara as features exatamente como no treinamento
    X_live = pd.get_dummies(instance[['partido', 'posicao_governo', 'uf', 'escolaridade']], drop_first=True)
    X_live['idade'] = instance['idade'].values
    X_live['pct_sim_historico'] = instance['pct_sim_historico'].values
    X_live['pct_sim_na_votacao'] = instance['pct_sim_na_votacao'].values
    X_live['pct_sim_uf'] = instance['pct_sim_uf'].values
    X_live['pct_sim_posicao_votacao'] = instance['pct_sim_posicao_votacao'].values

    faixa_idade = pd.cut(instance['idade'], bins=[0, 30, 40, 50, 60, 100],
                         labels=['18-30', '31-40', '41-50', '51-60', '60+'])
    X_faixa = pd.get_dummies(faixa_idade, prefix='faixa_idade', drop_first=True)
    X_live = pd.concat([X_live, X_faixa], axis=1)

    X_live = X_live.reindex(columns=feature_columns, fill_value=0)

    prediction_proba = model.predict_proba(X_live)
    real_vote = instance['tipoVoto'].iloc[0]

    return prediction_proba[0], real_vote


# --- Interface do Usuário (UI) ---

st.title("🗳️ Plenário Preditivo")
st.markdown(
    "Uma aplicação para simular e prever o voto de deputados com base em um modelo de Machine Learning com **92.8% de acurácia**.")

if df is not None:
    st.sidebar.header("Simulador de Votação")

    unique_votings = df[['id_votacao', 'proposicao_ementa']].drop_duplicates('id_votacao')


    # --- CORREÇÃO APLICADA AQUI ---
    def format_voting_option(voting_id):
        # Pega a ementa correspondente ao ID
        ementa = unique_votings[unique_votings['id_votacao'] == voting_id]['proposicao_ementa'].iloc[0]
        # Se a ementa for o nosso texto padrão, mostra um formato diferente
        if 'Ementa não disponível' in ementa:
            return f"Votação {voting_id} (Ementa Indisponível)"
        # Caso contrário, mostra o início da ementa real
        return ementa[:100] + "..."


    selected_voting_id = st.sidebar.selectbox(
        "Escolha uma Votação:",
        options=unique_votings['id_votacao'],
        format_func=format_voting_option  # Usamos nossa nova função inteligente
    )

    deputies_in_voting = df[df['id_votacao'] == selected_voting_id]
    selected_deputy_id = st.sidebar.selectbox(
        "Escolha um Deputado:",
        options=deputies_in_voting.sort_values('nome_urna')['id_deputado'],
        format_func=lambda x: deputies_in_voting[deputies_in_voting['id_deputado'] == x]['nome_urna'].iloc[0]
    )

    if st.sidebar.button("Executar Previsão", type="primary"):

        probabilities, real_vote = predict_vote(selected_deputy_id, selected_voting_id)

        if probabilities is not None:
            deputy_info = df[df['id_deputado'] == selected_deputy_id].iloc[0]

            st.header(f"Resultado para o(a) Dep. {deputy_info['nome_urna']}")
            st.write(
                f"**Partido:** {deputy_info['partido']} | **UF:** {deputy_info['uf']} | **Posição:** {deputy_info['posicao_governo']}")

            st.subheader("Ementa da Votação Selecionada:")
            st.info(
                f"{unique_votings[unique_votings['id_votacao'] == selected_voting_id]['proposicao_ementa'].iloc[0]}")

            prob_nao = probabilities[encoder.transform(['Não'])[0]]
            prob_sim = probabilities[encoder.transform(['Sim'])[0]]
            predicted_vote = "Sim" if prob_sim > prob_nao else "Não"

            st.subheader("Previsão do Modelo")
            col1, col2, col3 = st.columns(3)
            col1.metric("Voto Previsto", predicted_vote)
            col2.metric("Probabilidade de Votar **SIM**", f"{prob_sim:.2%}")
            col3.metric("Probabilidade de Votar **NÃO**", f"{prob_nao:.2%}")

            st.progress(prob_sim)

            st.subheader("Comparação")
            st.metric("Voto Real na Ocasião", real_vote)

            if predicted_vote == real_vote:
                st.success("✅ O modelo acertou a previsão!")
            else:
                st.error("❌ O modelo errou a previsão.")
else:
    st.info("Aguardando o carregamento dos artefatos do modelo...")