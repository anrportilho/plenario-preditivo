# app/pages/2_Analise_de_Votacao.py

import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

# --- Configuração da Página e Carregamento de Dados ---
st.set_page_config(page_title="Análise de Votação", page_icon="📊", layout="wide")


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
        st.error("Artefatos não encontrados. Execute o pipeline de scripts de 'src/' primeiro.")
        return None, None, None, None


model, encoder, feature_columns, df = load_artifacts()


# --- Funções de Lógica ---
def predict_votes_for_session(voting_df):
    """
    Recebe um DataFrame de uma votação e retorna as previsões para cada voto.
    """
    if voting_df.empty:
        return None

    # Prepara as features exatamente como no treinamento
    X_live = pd.get_dummies(voting_df[['partido', 'posicao_governo', 'uf', 'escolaridade']], drop_first=True)
    X_live['idade'] = voting_df['idade'].values
    X_live['pct_sim_historico'] = voting_df['pct_sim_historico'].values
    X_live['pct_sim_na_votacao'] = voting_df['pct_sim_na_votacao'].values
    X_live['pct_sim_uf'] = voting_df['pct_sim_uf'].values
    X_live['pct_sim_posicao_votacao'] = voting_df['pct_sim_posicao_votacao'].values
    faixa_idade = pd.cut(voting_df['idade'], bins=[0, 30, 40, 50, 60, 100],
                         labels=['18-30', '41-50', '51-60', '60+', '31-40'])
    X_faixa = pd.get_dummies(faixa_idade, prefix='faixa_idade', drop_first=True)
    X_live = pd.concat([X_live, X_faixa], axis=1)

    # Garante que todas as colunas do treino existam
    X_live = X_live.reindex(columns=feature_columns, fill_value=0)

    # Faz a previsão para todos os votos da sessão
    predictions = model.predict(X_live)

    # Cria um DataFrame com os resultados
    results_df = voting_df[['id_deputado', 'nome_urna', 'partido', 'uf', 'tipoVoto']].copy()
    results_df = results_df.rename(columns={'tipoVoto': 'voto_realizado'})
    results_df['voto_previsto'] = encoder.inverse_transform(predictions)

    return results_df


# --- Interface do Usuário (UI) ---
st.title("📊 Análise de Votação")
st.markdown(
    "Faça uma análise profunda de uma votação que já ocorreu, comparando o resultado real com as previsões do modelo.")

if df is not None:
    st.sidebar.header("Parâmetros da Análise")

    unique_votings = df[['id_votacao', 'proposicao_ementa']].drop_duplicates('id_votacao')


    def format_voting_option(voting_id):
        ementa = unique_votings[unique_votings['id_votacao'] == voting_id]['proposicao_ementa'].iloc[0]
        if 'Ementa não disponível' in ementa: return f"Votação {voting_id} (Processual)"
        return ementa[:100] + "..."


    selected_voting_id = st.sidebar.selectbox("Escolha uma Votação para Analisar:",
                                              options=unique_votings['id_votacao'], format_func=format_voting_option)

    if st.sidebar.button("Analisar Votação", type="primary"):

        # Filtra todos os votos da sessão selecionada
        voting_session_df = df[df['id_votacao'] == selected_voting_id].copy()

        if voting_session_df.empty:
            st.warning("Não há dados disponíveis para a votação selecionada.")
        else:
            with st.spinner('Processando análise...'):
                prediction_results = predict_votes_for_session(voting_session_df)

            ementa_selecionada = \
            unique_votings[unique_votings['id_votacao'] == selected_voting_id]['proposicao_ementa'].iloc[0]
            st.header(f"Análise da Votação: {selected_voting_id}")
            st.info(ementa_selecionada)

            # --- 1. Placar Comparativo ---
            st.subheader("Placar Comparativo")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Resultado Real")
                placar_real = prediction_results['voto_realizado'].value_counts()
                votos_sim_real = placar_real.get('Sim', 0)
                votos_nao_real = placar_real.get('Não', 0)
                st.metric("Votos 'SIM'", f"{votos_sim_real}")
                st.metric("Votos 'NÃO'", f"{votos_nao_real}")

            with col2:
                st.markdown("#### Resultado Previsto")
                placar_previsto = prediction_results['voto_previsto'].value_counts()
                votos_sim_prev = placar_previsto.get('Sim', 0)
                votos_nao_prev = placar_previsto.get('Não', 0)
                st.metric("Votos 'SIM'", f"{votos_sim_prev}", delta=f"{votos_sim_prev - votos_sim_real}")
                st.metric("Votos 'NÃO'", f"{votos_nao_prev}", delta=f"{votos_nao_prev - votos_nao_real}")

            # Acurácia específica da votação
            accuracy = accuracy_score(prediction_results['voto_realizado'], prediction_results['voto_previsto'])
            st.metric("🎯 Acurácia do Modelo (nesta votação)", f"{accuracy:.2%}")

            # --- 2. Análise de Votos Surpreendentes ---
            st.subheader("🔍 Votos Surpreendentes (Previsões Incorretas)")
            surprising_votes = prediction_results[
                prediction_results['voto_realizado'] != prediction_results['voto_previsto']]

            if surprising_votes.empty:
                st.success("🎉 O modelo acertou todas as previsões para esta votação!")
            else:
                st.write(
                    f"O modelo errou a previsão para **{len(surprising_votes)}** dos **{len(prediction_results)}** votos.")
                st.dataframe(surprising_votes[['nome_urna', 'partido', 'uf', 'voto_realizado', 'voto_previsto']],
                             use_container_width=True)
                st.caption("Esta tabela destaca deputados que votaram contra o padrão esperado pelo modelo.")

            # --- 3. Placeholder para SHAP ---
            st.subheader("🔬 Explicabilidade da Previsão (Próximo Passo)")
            st.info(
                "Funcionalidade em desenvolvimento: Selecione um deputado da tabela acima para ver um gráfico SHAP explicando os fatores que mais influenciaram a previsão do modelo para ele.")

else:
    st.info("Aguardando o carregamento dos artefatos do modelo...")