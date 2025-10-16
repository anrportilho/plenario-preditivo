# src/feature_engineering/create_modeling_dataset.py

import pandas as pd
from src.data_collection.api_client import save_to_parquet

if __name__ == "__main__":
    print("Iniciando a criação do dataset de modelagem final...")
    try:
        deputies_df = pd.read_parquet('data/processed/deputies_master_table.parquet')
        votes_df = pd.read_parquet('data/raw/votes.parquet')
        votings_details_df = pd.read_parquet('data/processed/votings_details.parquet')
    except FileNotFoundError as e:
        print(f"Erro: Arquivo não encontrado - {e}.")
        exit()

    deputado_details = pd.json_normalize(votes_df['deputado_'])
    deputado_details = deputado_details.rename(columns={'id': 'id_deputado'})
    votes_df = pd.concat([votes_df.drop(columns=['deputado_']), deputado_details], axis=1)

    votes_df['id_votacao'] = votes_df['id_votacao'].astype(str)
    votings_details_df['id_votacao'] = votings_details_df['id_votacao'].astype(str)

    votes_with_details_df = pd.merge(votes_df, votings_details_df, on='id_votacao', how='left')
    modeling_df = pd.merge(votes_with_details_df, deputies_df, on='id_deputado', how='inner')

    modeling_df['tipoVoto'] = modeling_df['tipoVoto'].str.strip()
    valid_votes = ['Sim', 'Não']
    modeling_df = modeling_df[modeling_df['tipoVoto'].isin(valid_votes)].copy()
    modeling_df['proposicao_ementa'].fillna('Ementa não disponível', inplace=True)

    # --- MUDANÇA AQUI: Adicionamos a data às colunas finais ---
    final_columns = [
        'id_votacao', 'id_deputado', 'dataRegistroVoto',  # <-- COLUNA DE DATA ADICIONADA
        'proposicao_ementa', 'nome_urna', 'partido', 'uf',
        'idade', 'escolaridade', 'tipoVoto'
    ]
    # Filtra apenas colunas que realmente existem para evitar erros
    existing_cols = [col for col in final_columns if col in modeling_df.columns]
    modeling_df = modeling_df[existing_cols]

    file_path = 'data/processed/modeling_dataset.parquet'
    save_to_parquet(modeling_df, file_path)
    print(f"\n✓ Dataset de modelagem base salvo em '{file_path}'")