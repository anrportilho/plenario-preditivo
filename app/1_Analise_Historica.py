# app/1_Analise_Historica.py

import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# --- Configuração da Página e Carregamento de Dados ---
st.set_page_config(page_title="Plenar.io Preditivo", page_icon="📊", layout="wide")


@st.cache_data
def load_artifacts():
    """Carrega todos os artefatos do modelo e os dados necessários."""
    try:
        model = joblib.load('models/lgbm_model.joblib')
        encoder = joblib.load('models/label_encoder.joblib')
        feature_columns = joblib.load('models/feature_columns.joblib')
        data = pd.read_parquet('data/processed/modeling_dataset_enriched.parquet')
        deputies_master = pd.read_parquet('data/processed/deputies_master_table.parquet')
        return model, encoder, feature_columns, data, deputies_master
    except FileNotFoundError:
        st.error("Artefatos não encontrados. Execute o pipeline de scripts de 'src/' primeiro.")
        return None, None, None, None, None


model, encoder, feature_columns, df, deputies_master_df = load_artifacts()


# --- Funções de Lógica ---
def predict_plenary_votes(voting_id):
    """Prevê o voto para TODOS os deputados para uma dada votação."""
    if df is None: return None

    voting_data = df[df['id_votacao'] == voting_id].iloc[0]
    prediction_df = deputies_master_df.copy()
    prediction_df['pct_sim_na_votacao'] = voting_data['pct_sim_na_votacao']
    prediction_df['pct_sim_posicao_votacao'] = voting_data['pct_sim_posicao_votacao']
    historical_features = df[['id_deputado', 'pct_sim_historico', 'pct_sim_uf']].drop_duplicates()
    prediction_df = pd.merge(prediction_df, historical_features, on='id_deputado', how='left')
    prediction_df.fillna(0.5, inplace=True)
    PARTIDOS_GOVERNO = ['PT', 'PCdoB', 'PV', 'PSB', 'MDB', 'PSD', 'REPUBLICANOS', 'PODE', 'UNIÃO', 'PSOL', 'REDE']
    PARTIDOS_OPOSICAO = ['PL', 'PP', 'NOVO']

    def define_posicao(partido):
        if partido in PARTIDOS_GOVERNO:
            return 'Governo'
        elif partido in PARTIDOS_OPOSICAO:
            return 'Oposição'
        else:
            return 'Independente'

    prediction_df['posicao_governo'] = prediction_df['partido'].apply(define_posicao)
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
    predictions = model.predict(X_live)
    prediction_df['voto_previsto'] = encoder.inverse_transform(predictions)
    real_votes = df[df['id_votacao'] == voting_id][['id_deputado', 'tipoVoto']]
    real_votes = real_votes.rename(columns={'tipoVoto': 'voto_realizado'})
    prediction_df = pd.merge(prediction_df, real_votes, on='id_deputado', how='left')
    prediction_df['voto_realizado'].fillna('Não Votou', inplace=True)
    return prediction_df


# --- Interface do Usuário (UI) ---
st.title("📊 Análise de Votações Históricas")
st.markdown(
    "Uma ferramenta de análise política para simular o placar de votações na Câmara com um modelo de **92.8% de acurácia**.")

if df is not None:
    st.sidebar.header("Parâmetros da Simulação")
    st.sidebar.markdown("**Passo 1: Defina o Cenário**")
    st.sidebar.caption(
        "A previsão usará o padrão de comportamento dos partidos e blocos desta votação como base para o cálculo.")

    unique_votings = df[['id_votacao', 'proposicao_ementa']].drop_duplicates('id_votacao')


    def format_voting_option(voting_id):
        ementa = unique_votings[unique_votings['id_votacao'] == voting_id]['proposicao_ementa'].iloc[0]
        if 'Ementa não disponível' in ementa: return f"Votação {voting_id} (Processual)"
        return ementa[:100] + "..."


    selected_voting_id = st.sidebar.selectbox("Usar Padrão de Voto da Sessão:", options=unique_votings['id_votacao'],
                                              format_func=format_voting_option)

    st.sidebar.markdown("**Passo 2: Filtre a Visualização (Opcional)**")
    bancada_options = ['Todas', 'Governo', 'Oposição', 'Independente']
    selected_bancada = st.sidebar.radio("Exibir Bloco Político:", options=bancada_options, index=0, horizontal=True)

    st.sidebar.markdown("**Passo 3: Execute a Análise**")
    if st.sidebar.button("Analisar Votação", type="primary"):
        with st.spinner('Calculando previsão para os deputados...'):
            prediction_results = predict_plenary_votes(selected_voting_id)

        if prediction_results is not None:
            ementa_selecionada = \
            unique_votings[unique_votings['id_votacao'] == selected_voting_id]['proposicao_ementa'].iloc[0]
            st.header(f"Análise da Votação")
            st.info(f"**Votação de Referência ({selected_voting_id}):** {ementa_selecionada}")

            if selected_bancada != 'Todas':
                display_df = prediction_results[prediction_results['posicao_governo'] == selected_bancada].copy()
            else:
                display_df = prediction_results.copy()

            if display_df.empty:
                st.warning(
                    f"Nenhum deputado do bloco '{selected_bancada}' participou ou foi encontrado para esta votação.")
            else:
                st.subheader(f"Placar: {selected_bancada}")
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("#### Resultado Real")
                    placar_real = display_df['voto_realizado'].value_counts()
                    votos_sim_real = placar_real.get('Sim', 0)
                    votos_nao_real = placar_real.get('Não', 0)
                    votos_outros_real = len(display_df) - votos_sim_real - votos_nao_real
                    resultado_real = "Aprovada" if votos_sim_real > 257 and selected_bancada == 'Todas' else "Rejeitada" if selected_bancada == 'Todas' else "-"
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Votos 'SIM'", f"{votos_sim_real}")
                    c2.metric("Votos 'NÃO'", f"{votos_nao_real}")
                    c3.metric("Ausentes/Outros", f"{votos_outros_real}")
                    if selected_bancada == 'Todas': st.metric("Status Real da Votação", resultado_real)

                with col2:
                    st.markdown("#### Resultado Previsto")
                    placar_previsto = display_df['voto_previsto'].value_counts()
                    votos_sim_prev = placar_previsto.get('Sim', 0)
                    votos_nao_prev = placar_previsto.get('Não', 0)
                    resultado_previsto = "Aprovada" if votos_sim_prev > 257 and selected_bancada == 'Todas' else "Rejeitada" if selected_bancada == 'Todas' else "-"
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Votos 'SIM'", f"{votos_sim_prev}", delta=f"{votos_sim_prev - votos_sim_real}")
                    c2.metric("Votos 'NÃO'", f"{votos_nao_prev}", delta=f"{votos_nao_prev - votos_nao_real}")
                    c3.metric("Total Previsto", f"{votos_sim_prev + votos_nao_prev}")
                    if selected_bancada == 'Todas': st.metric("Status Previsto da Votação", resultado_previsto)

                st.subheader(f"Distribuição dos Votos Reais ({selected_bancada})")
                bancada_votes = display_df.groupby('posicao_governo')['voto_realizado'].value_counts().unstack(
                    fill_value=0)

                if not bancada_votes.empty:
                    for col in ['Sim', 'Não', 'Não Votou']:
                        if col not in bancada_votes.columns: bancada_votes[col] = 0

                    fig = px.bar(bancada_votes, x=bancada_votes.index, y=['Sim', 'Não', 'Não Votou'],
                                 title=f"Votos Reais por Bloco Político ({selected_bancada})",
                                 labels={'x': 'Bloco Político', 'value': 'Número de Votos'},
                                 barmode='group',
                                 color_discrete_map={'Sim': 'green', 'Não': 'red', 'Não Votou': 'grey'})
                    st.plotly_chart(fig, use_container_width=True)

                st.subheader(f"Mapa de Votos Detalhado ({selected_bancada})")
                st.dataframe(
                    display_df[['nome_urna', 'partido', 'uf', 'posicao_governo', 'voto_previsto', 'voto_realizado']],
                    use_container_width=True)
else:
    st.info("Aguardando o carregamento dos artefatos do modelo...")