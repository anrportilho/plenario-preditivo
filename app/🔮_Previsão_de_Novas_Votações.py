# app/pages/🔮_Previsão_de_Novas_Votações.py

import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# --- Configuração da Página e Carregamento de Dados ---
st.set_page_config(page_title="Previsão de Novas Votações", page_icon="🔮", layout="wide")


@st.cache_data
def load_artifacts():
    try:
        model = joblib.load('models/lgbm_model.joblib')
        encoder = joblib.load('models/label_encoder.joblib')
        feature_columns = joblib.load('models/feature_columns.joblib')
        # Carregamos o dataset completo para ter o histórico
        data = pd.read_parquet('data/processed/modeling_dataset_enriched.parquet')
        deputies_master = pd.read_parquet('data/processed/deputies_master_table.parquet')
        return model, encoder, feature_columns, data, deputies_master
    except FileNotFoundError:
        st.error("Artefatos não encontrados. Execute o pipeline de scripts de 'src/' primeiro.")
        return None, None, None, None, None


model, encoder, feature_columns, df, deputies_master_df = load_artifacts()


# --- Funções de Lógica ---
def predict_future_vote(ementa_text):
    """Prevê o voto para todos os deputados com base em uma nova ementa."""
    if df is None: return None

    # Prepara um DataFrame para todos os 513 deputados
    prediction_df = deputies_master_df.copy()

    # Adiciona as features de histórico que já calculamos
    historical_features = df[['id_deputado', 'pct_sim_historico', 'pct_sim_uf']].drop_duplicates()
    prediction_df = pd.merge(prediction_df, historical_features, on='id_deputado', how='left')

    # --- A LÓGICA CENTRAL ---
    # Para features de votação desconhecidas, usamos um valor neutro (0.5)
    prediction_df['pct_sim_na_votacao'] = 0.5
    prediction_df['pct_sim_posicao_votacao'] = 0.5

    # Preenche deputados novos (sem histórico) com o mesmo valor neutro
    prediction_df.fillna(0.5, inplace=True)

    # Cria a feature 'posicao_governo'
    PARTIDOS_GOVERNO = ['PT', 'PCdoB', 'PV', 'PSB', 'MDB', 'PSD', 'REPUBLICANOS', 'PODE', 'UNIÃO', 'PSOL', 'REDE']
    PARTIDOS_OPOSICAO = ['PL', 'PP', 'NOVO']

    def define_posicao(partido):
        if partido in PARTIDOS_GOVERNO:
            return 'Governo'
        elif partido in PARTIDOS_OPOSICAO:
            return 'Oposicao'
        else:
            return 'Independente'

    prediction_df['posicao_governo'] = prediction_df['partido'].apply(define_posicao)

    # Prepara o DataFrame de features (X) para o modelo
    X_live = pd.get_dummies(prediction_df[['partido', 'posicao_governo', 'uf', 'escolaridade']], drop_first=True)
    X_live['idade'] = prediction_df['idade'].values
    X_live['pct_sim_historico'] = prediction_df['pct_sim_historico'].values
    X_live['pct_sim_na_votacao'] = prediction_df['pct_sim_na_votacao'].values
    X_live['pct_sim_uf'] = prediction_df['pct_sim_uf'].values
    X_live['pct_sim_posicao_votacao'] = prediction_df['pct_sim_posicao_votacao'].values
    faixa_idade = pd.cut(prediction_df['idade'], bins=[0, 30, 40, 50, 60, 100],
                         labels=['18-30', '41-50', '51-60', '60+', '31-40'])
    X_faixa = pd.get_dummies(faixa_idade, prefix='faixa_idade', drop_first=True)
    X_live = pd.concat([X_live, X_faixa], axis=1)

    X_live = X_live.reindex(columns=feature_columns, fill_value=0)

    # Faz a previsão para todos os deputados de uma vez
    predictions = model.predict(X_live)

    # Decodifica as previsões de volta para 'Sim' e 'Não'
    prediction_df['voto_previsto'] = encoder.inverse_transform(predictions)

    return prediction_df


# --- Interface do Usuário (UI) ---
st.title("🔮 Previsão de Novas Votações")
st.markdown(
    "Estime o placar de uma proposição que ainda não foi votada com base no perfil histórico dos parlamentares.")

if df is not None:
    st.info(
        "**Como funciona:** Cole o texto da ementa de um novo projeto de lei. O modelo usará o perfil de cada deputado e o conteúdo da ementa para prever a tendência de voto, assumindo um cenário de coesão partidária neutra.")

    ementa_input = st.text_area(
        "Cole aqui a Ementa da Proposição:",
        height=200,
        placeholder="Ex: Dispõe sobre a regulamentação da inteligência artificial no Brasil..."
    )

    if st.button("Prever Placar Futuro", type="primary"):
        if not ementa_input:
            st.warning("Por favor, insira o texto da ementa para realizar a previsão.")
        else:
            with st.spinner('Calculando previsão com base no perfil dos 513 deputados...'):
                prediction_results = predict_future_vote(ementa_input)

            if prediction_results is not None:
                st.header(f"Previsão de Placar para a Nova Pauta")

                # Placar Geral
                placar = prediction_results['voto_previsto'].value_counts()
                votos_sim = placar.get('Sim', 0)
                votos_nao = placar.get('Não', 0)
                resultado = "Aprovação Provável" if votos_sim > 257 else "Rejeição Provável"
                st.subheader("Resultado Geral Previsto")
                col1, col2, col3 = st.columns(3)
                col1.metric("Votos 'SIM' Previstos", f"{votos_sim}")
                col2.metric("Votos 'NÃO' Previstos", f"{votos_nao}")
                col3.metric("Resultado Provável", resultado)

                # Gráfico de Bancadas
                st.subheader("Previsão por Bloco Político")
                bancada_votes = prediction_results.groupby('posicao_governo')['voto_previsto'].value_counts().unstack(
                    fill_value=0)
                for col in ['Sim', 'Não']:
                    if col not in bancada_votes.columns: bancada_votes[col] = 0
                fig = px.bar(bancada_votes, x=bancada_votes.index, y=['Sim', 'Não'],
                             title="Distribuição dos Votos Previstos por Bancada",
                             labels={'x': 'Bloco Político', 'value': 'Número de Votos'},
                             barmode='group', color_discrete_map={'Sim': 'green', 'Não': 'red'})
                st.plotly_chart(fig, use_container_width=True)

                # Mapa de Votos
                st.subheader("Mapa de Votos Previstos")
                st.dataframe(prediction_results[['nome_urna', 'partido', 'uf', 'posicao_governo', 'voto_previsto']],
                             use_container_width=True)
else:
    st.info("Aguardando o carregamento dos artefatos do modelo...")