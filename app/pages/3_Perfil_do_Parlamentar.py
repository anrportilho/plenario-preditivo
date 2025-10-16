# app/pages/3_Perfil_do_Parlamentar.py

import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# --- Configuração da Página e Carregamento de Dados ---
st.set_page_config(page_title="Perfil do Parlamentar", page_icon="👤", layout="wide")


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
def predict_votes_for_deputy(deputy_df):
    """Recebe um DataFrame com o histórico de um deputado e retorna as previsões."""
    if deputy_df.empty:
        return None

    # Prepara as features para todas as votações do deputado
    X_live = pd.get_dummies(deputy_df[['partido', 'posicao_governo', 'uf', 'escolaridade']], drop_first=True)
    X_live['idade'] = deputy_df['idade'].values
    X_live['pct_sim_historico'] = deputy_df['pct_sim_historico'].values
    X_live['pct_sim_na_votacao'] = deputy_df['pct_sim_na_votacao'].values
    X_live['pct_sim_uf'] = deputy_df['pct_sim_uf'].values
    X_live['pct_sim_posicao_votacao'] = deputy_df['pct_sim_posicao_votacao'].values
    faixa_idade = pd.cut(deputy_df['idade'], bins=[0, 30, 40, 50, 60, 100],
                         labels=['18-30', '41-50', '51-60', '60+', '31-40'])
    X_faixa = pd.get_dummies(faixa_idade, prefix='faixa_idade', drop_first=True)
    X_live = pd.concat([X_live, X_faixa], axis=1)

    X_live = X_live.reindex(columns=feature_columns, fill_value=0)

    # Faz a previsão
    predictions = model.predict(X_live)

    # Adiciona as previsões ao DataFrame do deputado
    deputy_df['voto_previsto'] = encoder.inverse_transform(predictions)
    return deputy_df


# --- Interface do Usuário (UI) ---
st.title("👤 Perfil do Parlamentar")
st.markdown("Analise o perfil de votação, o alinhamento político e a previsibilidade de cada deputado.")

if df is not None:
    st.sidebar.header("Seleção de Parlamentar")

    selected_deputy_id = st.sidebar.selectbox(
        "Escolha um Deputado para Analisar:",
        options=deputies_master_df.sort_values('nome_urna')['id_deputado'],
        format_func=lambda x: deputies_master_df[deputies_master_df['id_deputado'] == x]['nome_urna'].iloc[0]
    )

    if selected_deputy_id:
        deputy_data = df[df['id_deputado'] == selected_deputy_id].copy()

        if deputy_data.empty:
            st.warning("Nenhum histórico de votação encontrado para este parlamentar no dataset.")
        else:
            deputy_info = deputy_data.iloc[0]

            st.header(f"Análise de {deputy_info['nome_urna']}")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Partido", f"{deputy_info['partido']}-{deputy_info['uf']}")
            col2.metric("Posição", deputy_info['posicao_governo'])
            col3.metric("Idade", f"{deputy_info['idade']} anos")
            col4.metric("Taxa Histórica de Votos 'Sim'", f"{deputy_info['pct_sim_historico']:.2%}")

            st.subheader("Alinhamento Político")
            deputy_alignment = deputy_info['pct_sim_historico'] * 100
            party_alignment = df[df['partido'] == deputy_info['partido']]['pct_sim_historico'].mean() * 100
            bloc_alignment = df[df['posicao_governo'] == deputy_info['posicao_governo']][
                                 'pct_sim_historico'].mean() * 100
            alignment_data = pd.DataFrame({
                'Entidade': [f"Dep. {deputy_info['nome_urna']}", f"Partido ({deputy_info['partido']})",
                             f"Bloco ({deputy_info['posicao_governo']})"],
                'Taxa de Votos "Sim" (%)': [deputy_alignment, party_alignment, bloc_alignment]
            })
            fig = px.bar(alignment_data,
                         x='Taxa de Votos "Sim" (%)',
                         y='Entidade',
                         orientation='h',
                         title="Comparativo de Alinhamento (Taxa Histórica de Votos 'Sim')",
                         text=alignment_data['Taxa de Votos "Sim" (%)'].apply(lambda x: f'{x:.1f}%'))
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Histórico de Votações Recentes e Previsibilidade")

            with st.spinner("Processando histórico e previsões..."):
                deputy_predictions = predict_votes_for_deputy(deputy_data)

            if deputy_predictions is not None:
                accuracy = (deputy_predictions['tipoVoto'] == deputy_predictions['voto_previsto']).mean()
                st.metric("🎯 Previsibilidade do Deputado (Acurácia do Modelo)", f"{accuracy:.2%}")

                deputy_predictions['acerto'] = deputy_predictions.apply(
                    lambda row: '✅' if row['tipoVoto'] == row['voto_previsto'] else '❌',
                    axis=1
                )

                # --- MUDANÇA APLICADA AQUI ---
                # Adicionamos 'id_votacao' à lista de colunas e ao dicionário de renomeação
                st.dataframe(
                    deputy_predictions[
                        ['id_votacao', 'proposicao_ementa', 'tipoVoto', 'voto_previsto', 'acerto']].rename(columns={
                        'id_votacao': 'ID Votação',  # <-- NOVA LINHA
                        'proposicao_ementa': 'Ementa da Votação',
                        'tipoVoto': 'Voto Real',
                        'voto_previsto': 'Voto Previsto',
                        'acerto': 'Acerto do Modelo'
                    }),
                    use_container_width=True
                )
            else:
                st.error("Não foi possível gerar as previsões para este deputado.")
else:
    st.info("Aguardando o carregamento dos artefatos do modelo...")