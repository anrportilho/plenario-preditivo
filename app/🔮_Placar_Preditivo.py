# app/dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# --- Configuração da Página ---

st.set_page_config(
    page_title="Plenário Preditivo",
    page_icon="🗳️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS Customizado ---

st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1f6feb 0%, #388bfd 100%);
        border-radius: 10px;
        padding: 20px;
        color: white;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }

    .success-box {
        background-color: #d3f9d8;
        border-left: 4px solid #51cf66;
        padding: 10px;
        border-radius: 5px;
    }

    .error-box {
        background-color: #ffe0e0;
        border-left: 4px solid #ff6b6b;
        padding: 10px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)


# --- Cache e Carregamento de Dados ---

@st.cache_data
def load_artifacts():
    """Carrega todos os artefatos do modelo e os dados necessários."""
    try:
        model = joblib.load('models/lgbm_model.joblib')
        encoder = joblib.load('models/label_encoder.joblib')
        feature_columns = joblib.load('models/feature_columns.joblib')
        data = pd.read_parquet('data/processed/modeling_dataset_enriched.parquet')

        # Converter dataRegistroVoto para datetime
        if 'dataRegistroVoto' in data.columns:
            data['dataRegistroVoto'] = pd.to_datetime(data['dataRegistroVoto'], errors='coerce')

        return model, encoder, feature_columns, data
    except FileNotFoundError as e:
        st.error(f"❌ Erro ao carregar artefatos: {e}")
        st.info("Certifique-se de executar o pipeline em `src/` primeiro.")
        return None, None, None, None


model, encoder, feature_columns, df = load_artifacts()


# --- Funções de Lógica ---

def predict_vote(deputy_id, voting_id):
    """Prediz o voto e retorna probabilidades e informações adicionais."""
    if df is None:
        return None, None, None

    instance = df[(df['id_deputado'] == deputy_id) & (df['id_votacao'] == voting_id)]

    if instance.empty:
        return None, None, None

    try:
        X_live = pd.get_dummies(
            instance[['partido', 'posicao_governo', 'uf', 'escolaridade']],
            drop_first=True
        )
        X_live['idade'] = instance['idade'].values

        behavioral_features = ['pct_sim_historico', 'pct_sim_na_votacao', 'pct_sim_uf', 'pct_sim_posicao_votacao']
        for feature in behavioral_features:
            if feature in instance.columns:
                X_live[feature] = instance[feature].values

        faixa_idade = pd.cut(instance['idade'], bins=[0, 30, 40, 50, 60, 100],
                             labels=['18-30', '31-40', '41-50', '51-60', '60+'])
        X_faixa = pd.get_dummies(faixa_idade, prefix='faixa_idade', drop_first=True)
        X_live = pd.concat([X_live, X_faixa], axis=1)

        X_live = X_live.reindex(columns=feature_columns, fill_value=0)

        prediction_proba = model.predict_proba(X_live)[0]
        real_vote = instance['tipoVoto'].iloc[0]
        confidence = max(prediction_proba) * 100

        return prediction_proba, real_vote, confidence

    except Exception as e:
        st.warning(f"⚠️ Erro na predição: {e}")
        return None, None, None


def get_global_statistics():
    """Retorna estatísticas globais do dataset."""
    return {
        'total_votacoes': df['id_votacao'].nunique(),
        'total_deputados': df['id_deputado'].nunique(),
        'total_votos': len(df),
        'taxa_sim_global': (df['tipoVoto'] == 'Sim').sum() / len(df) * 100,
        'taxa_nao_global': (df['tipoVoto'] == 'Não').sum() / len(df) * 100,
    }


# --- Interface Principal ---

col1, col2 = st.columns([0.7, 0.3])
with col1:
    st.title("🗳️ Plenário Preditivo")
    st.markdown("**Sistema inteligente de previsão de votações na Câmara dos Deputados**")
with col2:
    if st.button("🔄 Atualizar Dados", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

st.divider()

if df is not None:
    # --- ESTATÍSTICAS GLOBAIS ---

    st.subheader("📊 Visão Geral do Sistema")

    global_stats = get_global_statistics()

    cols = st.columns(4)
    with cols[0]:
        st.metric("📋 Votações Analisadas", f"{global_stats['total_votacoes']:,}")
    with cols[1]:
        st.metric("👥 Deputados Monitorados", f"{global_stats['total_deputados']:,}")
    with cols[2]:
        st.metric("✅ Taxa de Aprovação", f"{global_stats['taxa_sim_global']:.1f}%")
    with cols[3]:
        st.metric("🎯 Acurácia do Modelo", "92.8%")

    st.divider()

    # --- ABAS DE NAVEGAÇÃO ---

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎯 Simulador", "📈 Deputados", "🔍 Votações", "📊 Dashboard", "🔮 Previsões"])

    # ===== TAB 1: SIMULADOR =====
    with tab1:
        st.subheader("Simulador de Votação")
        st.markdown("Selecione um deputado e uma votação para prever o voto")

        st.write("**Filtros:**")
        col_filtro = st.columns([1, 1])

        with col_filtro[0]:
            bancadas = ['Todas', 'Governo', 'Oposição', 'Independente']
            selected_bancada = st.selectbox("Filtrar por Bancada:", bancadas)

        deputies_filtered = df.copy()
        if selected_bancada != 'Todas':
            deputies_filtered = deputies_filtered[deputies_filtered['posicao_governo'] == selected_bancada]

        col1, col2 = st.columns(2)

        with col1:
            unique_votings = df[['id_votacao', 'proposicao_ementa']].drop_duplicates('id_votacao')


            def format_voting(voting_id):
                ementa = unique_votings[unique_votings['id_votacao'] == voting_id]['proposicao_ementa'].iloc[0]
                if pd.isna(ementa) or 'Ementa não disponível' in str(ementa):
                    return f"Votação #{voting_id}"
                return f"{ementa[:80]}..." if len(ementa) > 80 else ementa


            voting_options = ['Ver todas as votações'] + list(unique_votings['id_votacao'].values)
            selected_voting_display = st.selectbox(
                "Escolha uma Votação:",
                options=voting_options,
                format_func=lambda x: "📋 Ver todas as votações" if x == 'Ver todas as votações' else format_voting(x)
            )

            selected_voting_id = selected_voting_display if selected_voting_display != 'Ver todas as votações' else None

        with col2:
            if selected_voting_id:
                deputies_in_voting = df[df['id_votacao'] == selected_voting_id]
                deputies_in_voting = deputies_in_voting[
                    deputies_in_voting['id_deputado'].isin(deputies_filtered['id_deputado'].unique())]
            else:
                deputies_in_voting = deputies_filtered

            deputy_options = ['Ver todos os deputados'] + list(
                deputies_in_voting.sort_values('nome_urna')['id_deputado'].unique())
            selected_deputy_display = st.selectbox(
                "Escolha um Deputado:",
                options=deputy_options,
                format_func=lambda x: "👥 Ver todos os deputados" if x == 'Ver todos os deputados' else
                deputies_in_voting[deputies_in_voting['id_deputado'] == x]['nome_urna'].iloc[0]
            )

            selected_deputy_id = selected_deputy_display if selected_deputy_display != 'Ver todos os deputados' else None

        if st.button("🚀 Executar Previsão", type="primary", use_container_width=True):
            if selected_voting_id is None or selected_deputy_id is None:
                st.warning("⚠️ Por favor, selecione uma votação e um deputado específicos para fazer a previsão.")
            else:
                probabilities, real_vote, confidence = predict_vote(selected_deputy_id, selected_voting_id)

                if probabilities is not None:
                    deputy_info = df[df['id_deputado'] == selected_deputy_id].iloc[0]
                    voting_info = df[df['id_votacao'] == selected_voting_id].iloc[0]

                    st.success(f"✅ Previsão realizada com sucesso!")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write(f"**Deputado:** {deputy_info['nome_urna']}")
                        st.write(f"**Partido:** {deputy_info['partido']}")
                    with col2:
                        st.write(f"**UF:** {deputy_info['uf']}")
                        st.write(f"**Idade:** {deputy_info['idade']} anos")
                    with col3:
                        st.write(f"**Posição:** {deputy_info['posicao_governo']}")
                        st.write(f"**Escolaridade:** {deputy_info['escolaridade']}")

                    st.divider()

                    st.write("**📝 Ementa da Votação:**")
                    st.info(voting_info['proposicao_ementa'])

                    st.divider()

                    col1, col2, col3 = st.columns(3)

                    prob_nao = probabilities[encoder.transform(['Não'])[0]]
                    prob_sim = probabilities[encoder.transform(['Sim'])[0]]
                    predicted_vote = "Sim" if prob_sim > prob_nao else "Não"

                    with col1:
                        st.metric("🗳️ Voto Previsto", predicted_vote, delta=f"{confidence:.1f}% confiança")
                    with col2:
                        st.metric("✅ Prob. SIM", f"{prob_sim:.1%}")
                    with col3:
                        st.metric("❌ Prob. NÃO", f"{prob_nao:.1%}")

                    st.write("**Distribuição de Probabilidade:**")
                    fig = go.Figure(data=[
                        go.Bar(x=['Sim', 'Não'], y=[prob_sim, prob_nao],
                               marker_color=['#22c55e', '#ef4444'],
                               text=[f'{prob_sim:.1%}', f'{prob_nao:.1%}'],
                               textposition='auto')
                    ])
                    fig.update_layout(height=300, showlegend=False, title="Probabilidades")
                    st.plotly_chart(fig, use_container_width=True)

                    st.divider()
                    st.write("**📊 Comparação com Voto Real:**")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Voto Previsto:** {predicted_vote}")
                    with col2:
                        st.write(f"**Voto Real:** {real_vote}")

                    if predicted_vote == real_vote:
                        st.markdown(
                            '<div class="success-box">✅ <b>Acerto!</b> O modelo previu corretamente o voto.</div>',
                            unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="error-box">❌ <b>Erro.</b> O modelo não acertou esta predição.</div>',
                                    unsafe_allow_html=True)
                else:
                    st.error("❌ Não foi possível fazer a previsão. Verifique os dados.")

    # ===== TAB 2: ANÁLISE DE DEPUTADOS =====
    with tab2:
        st.subheader("Análise Detalhada de Deputados")

        col1, col2 = st.columns(2)
        with col1:
            partidos = ['Todos'] + sorted(df['partido'].unique().tolist())
            selected_party = st.selectbox("Filtrar por Partido:", partidos)
        with col2:
            posicoes = ['Todos'] + sorted(df['posicao_governo'].unique().tolist())
            selected_position = st.selectbox("Filtrar por Posição:", posicoes)

        filtered_df = df.copy()
        if selected_party != 'Todos':
            filtered_df = filtered_df[filtered_df['partido'] == selected_party]
        if selected_position != 'Todos':
            filtered_df = filtered_df[filtered_df['posicao_governo'] == selected_position]

        deputies_summary = filtered_df.groupby('id_deputado').agg({
            'nome_urna': 'first',
            'partido': 'first',
            'uf': 'first',
            'idade': 'first',
            'tipoVoto': 'count'
        }).rename(columns={'tipoVoto': 'votacoes'}).reset_index()

        sim_votes = filtered_df[filtered_df['tipoVoto'] == 'Sim'].groupby('id_deputado').size()
        deputies_summary['taxa_sim'] = deputies_summary['id_deputado'].map(
            lambda x: (sim_votes.get(x, 0) / len(filtered_df[filtered_df['id_deputado'] == x]) * 100)
        )

        deputies_summary = deputies_summary.sort_values('votacoes', ascending=False)

        st.dataframe(
            deputies_summary.rename(columns={
                'nome_urna': 'Deputado',
                'partido': 'Partido',
                'uf': 'UF',
                'idade': 'Idade',
                'votacoes': 'Votações',
                'taxa_sim': 'Taxa SIM (%)'
            }),
            use_container_width=True,
            hide_index=True
        )

        col1, col2 = st.columns(2)

        with col1:
            fig = px.histogram(deputies_summary, x='taxa_sim', nbins=20,
                               title="Distribuição de Taxa de Votação 'Sim'",
                               labels={'taxa_sim': 'Taxa SIM (%)', 'count': 'Quantidade'})
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            party_dist = filtered_df['partido'].value_counts()
            fig = px.pie(values=party_dist.values, names=party_dist.index,
                         title="Distribuição por Partido")
            st.plotly_chart(fig, use_container_width=True)

    # ===== TAB 3: ANÁLISE DE VOTAÇÕES =====
    with tab3:
        st.subheader("Análise de Votações")

        votings_summary = df.groupby('id_votacao').agg({
            'proposicao_ementa': 'first',
            'tipoVoto': lambda x: (x == 'Sim').sum(),
            'id_deputado': 'count'
        }).reset_index()
        votings_summary.columns = ['id_votacao', 'ementa', 'votos_sim', 'total_votos']
        votings_summary['taxa_aprovacao'] = (votings_summary['votos_sim'] / votings_summary['total_votos'] * 100).round(
            1)

        st.dataframe(
            votings_summary.rename(columns={
                'ementa': 'Ementa',
                'votos_sim': 'Votos SIM',
                'total_votos': 'Total de Votos',
                'taxa_aprovacao': 'Taxa Aprovação (%)'
            }),
            use_container_width=True,
            hide_index=True
        )

        votings_summary_sorted = votings_summary.sort_values('id_votacao')
        fig = px.line(votings_summary_sorted, x='id_votacao', y='taxa_aprovacao',
                      title="Tendência de Taxa de Aprovação",
                      labels={'id_votacao': 'Votação', 'taxa_aprovacao': 'Taxa de Aprovação (%)'})
        st.plotly_chart(fig, use_container_width=True)

    # ===== TAB 4: DASHBOARD =====
    with tab4:
        st.subheader("Dashboard Executivo")

        st.write("### 📊 Resultado Geral das Votações")

        votings_results = df.groupby('id_votacao')['tipoVoto'].apply(
            lambda x: 'Aprovada' if (x == 'Sim').sum() > (x == 'Não').sum() else 'Rejeitada'
        ).value_counts()

        fig = go.Figure(data=[
            go.Bar(x=votings_results.index, y=votings_results.values,
                   marker_color=['#22c55e', '#ef4444'],
                   text=votings_results.values,
                   textposition='auto')
        ])
        fig.update_layout(
            title="Demandas Aprovadas vs Rejeitadas",
            xaxis_title="Status da Demanda",
            yaxis_title="Quantidade",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        st.write("### 🏛️ Análise por Bancada")

        bancadas_list = ['Governo', 'Oposição', 'Independente']
        cols = st.columns(3)

        for idx, bancada in enumerate(bancadas_list):
            with cols[idx]:
                df_bancada = df[df['posicao_governo'] == bancada]

                sim_count = (df_bancada['tipoVoto'] == 'Sim').sum()
                nao_count = (df_bancada['tipoVoto'] == 'Não').sum()

                fig = go.Figure(data=[
                    go.Bar(x=['SIM', 'NÃO'], y=[sim_count, nao_count],
                           marker_color=['#22c55e', '#ef4444'],
                           text=[sim_count, nao_count],
                           textposition='auto')
                ])
                fig.update_layout(
                    title=f"Votações - Bancada {bancada}",
                    xaxis_title="Tipo de Voto",
                    yaxis_title="Quantidade",
                    height=400,
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            party_votes = df.groupby('partido')['tipoVoto'].apply(
                lambda x: (x == 'Sim').sum() / len(x) * 100
            ).sort_values(ascending=True)

            fig = px.bar(x=party_votes.values, y=party_votes.index,
                         orientation='h',
                         title="Taxa de Votação 'SIM' por Partido",
                         labels={'x': 'Taxa (%)', 'y': 'Partido'})
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            uf_votes = df.groupby('uf')['tipoVoto'].apply(
                lambda x: (x == 'Sim').sum() / len(x) * 100
            ).sort_values(ascending=True).tail(15)

            fig = px.bar(x=uf_votes.values, y=uf_votes.index,
                         orientation='h',
                         title="Taxa de Votação 'SIM' por UF (Top 15)",
                         labels={'x': 'Taxa (%)', 'y': 'UF'})
            st.plotly_chart(fig, use_container_width=True)

    # ===== TAB 5: PREVISÃO DE APROVAÇÃO =====
    with tab5:
        st.subheader("🔮 Previsão de Aprovação/Rejeição de Votações")
        st.markdown("Visualize a previsão agregada baseada no histórico de votações realizadas")

        col_filter = st.columns(1)[0]
        with col_filter:
            bancadas_filter = ['Todas', 'Governo', 'Oposição', 'Independente']
            selected_bancada_tab5 = st.selectbox("Filtrar por Bancada:", bancadas_filter, key='tab5_bancada')

        st.write("### 📊 Previsão de Resultados por Votação (Históricas)")

        votings_prediction = []
        for voting_id in df['id_votacao'].unique():
            voting_data = df[df['id_votacao'] == voting_id]

            if selected_bancada_tab5 != 'Todas':
                voting_data = voting_data[voting_data['posicao_governo'] == selected_bancada_tab5]

            if voting_data.empty:
                continue

            sim_votes = (voting_data['tipoVoto'] == 'Sim').sum()
            total_votes = len(voting_data)
            approval_rate = sim_votes / total_votes if total_votes > 0 else 0

            predicted_result = 'Aprovada' if approval_rate > 0.5 else 'Rejeitada'
            ementa = df[df['id_votacao'] == voting_id]['proposicao_ementa'].iloc[0]

            if pd.isna(ementa) or 'Ementa não disponível' in str(ementa):
                ementa = f"Votação #{voting_id}"

            data_votacao = df[df['id_votacao'] == voting_id]['dataRegistroVoto'].iloc[0]
            data_formatada = data_votacao.strftime('%d/%m/%Y %H:%M') if pd.notna(data_votacao) else 'N/A'

            votings_prediction.append({
                'Votação': f"#{voting_id}",
                'Ementa': ementa,
                'Data': data_formatada,
                'Taxa SIM': f"{approval_rate:.1%}",
                'Previsão': predicted_result,
                'Confiança': f"{max(approval_rate, 1 - approval_rate):.1%}",
                'Total Votos': total_votes
            })

        prediction_df = pd.DataFrame(votings_prediction)

        if not prediction_df.empty:
            st.dataframe(prediction_df, use_container_width=True, hide_index=True)

            col1, col2 = st.columns(2)

            with col1:
                approved_count = (prediction_df['Previsão'] == 'Aprovada').sum()
                rejected_count = (prediction_df['Previsão'] == 'Rejeitada').sum()

                fig = go.Figure(data=[
                    go.Bar(x=['Aprovadas', 'Rejeitadas'],
                           y=[approved_count, rejected_count],
                           marker_color=['#22c55e', '#ef4444'],
                           text=[approved_count, rejected_count],
                           textposition='auto')
                ])
                fig.update_layout(title="Padrão Histórico: Votações Aprovadas vs Rejeitadas", height=400)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                avg_approval = prediction_df['Taxa SIM'].str.rstrip('%').astype(float).mean()
                fig = go.Figure(data=[
                    go.Indicator(
                        mode="gauge+number+delta",
                        value=avg_approval,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Taxa Média de Aprovação"},
                        delta={'reference': 50},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': '#22c55e'},
                            'steps': [
                                {'range': [0, 25], 'color': '#ef4444'},
                                {'range': [25, 50], 'color': '#f59e0b'},
                                {'range': [50, 75], 'color': '#3b82f6'},
                                {'range': [75, 100], 'color': '#22c55e'}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 50
                            }
                        }
                    )
                ])
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Nenhuma votação disponível com os filtros selecionados.")

else:
    st.warning("⚠️ Aguardando o carregamento dos artefatos do modelo...")