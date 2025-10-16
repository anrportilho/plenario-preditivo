# src/feature_engineering/enrich_behavioral_features.py

import pandas as pd
from src.data_collection.api_client import save_to_parquet

if __name__ == "__main__":
    print("Enriquecendo dataset com features comportamentais...\n")

    try:
        df = pd.read_parquet('data/processed/modeling_dataset.parquet')
        print(f"Dataset original carregado: {len(df)} linhas\n")
    except FileNotFoundError:
        print("Erro: modeling_dataset.parquet não encontrado.")
        print("Certifique-se de que o script 'create_modeling_dataset.py' foi executado com sucesso.")
        exit()

    # --- CORREÇÃO APLICADA AQUI ---
    PARTIDOS_GOVERNO = ['PT', 'PCdoB', 'PV', 'PSB', 'MDB', 'PSD', 'REPUBLICANOS', 'PODE', 'UNIÃO', 'PSOL', 'REDE']
    PARTIDOS_OPOSICAO = ['PL', 'PP', 'NOVO']


    def define_posicao(partido):
        if partido in PARTIDOS_GOVERNO:
            return 'Governo'
        elif partido in PARTIDOS_OPOSICAO:
            return 'Oposição'  # <-- Corrigido para "Oposição" com "ç"
        else:
            return 'Independente'


    df['posicao_governo'] = df['partido'].apply(define_posicao)

    # (O restante do seu código de cálculo de features permanece o mesmo)
    deputy_vote_stats = df.groupby('id_deputado')['tipoVoto'].apply(
        lambda x: (x == 'Sim').sum() / len(x) if len(x) > 0 else 0.5).reset_index(name='pct_sim_historico')
    df = pd.merge(df, deputy_vote_stats, on='id_deputado', how='left')
    votacao_partido_stats = df.groupby(['id_votacao', 'partido'])['tipoVoto'].apply(
        lambda x: (x == 'Sim').sum() / len(x) if len(x) > 0 else 0.5).reset_index(name='pct_sim_na_votacao')
    df = pd.merge(df, votacao_partido_stats, on=['id_votacao', 'partido'], how='left')
    uf_vote_stats = df.groupby('uf')['tipoVoto'].apply(
        lambda x: (x == 'Sim').sum() / len(x) if len(x) > 0 else 0.5).reset_index(name='pct_sim_uf')
    df = pd.merge(df, uf_vote_stats, on='uf', how='left')
    votacao_posicao_stats = df.groupby(['id_votacao', 'posicao_governo'])['tipoVoto'].apply(
        lambda x: (x == 'Sim').sum() / len(x) if len(x) > 0 else 0.5).reset_index(name='pct_sim_posicao_votacao')
    df = pd.merge(df, votacao_posicao_stats, on=['id_votacao', 'posicao_governo'], how='left')

    file_path = 'data/processed/modeling_dataset_enriched.parquet'
    save_to_parquet(df, file_path)
    print(f"\n✓ Dataset enriquecido salvo em '{file_path}' com nomes corrigidos.")