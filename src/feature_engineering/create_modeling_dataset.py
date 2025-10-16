# src/feature_engineering/create_modeling_dataset.py

import pandas as pd
from src.data_collection.api_client import save_to_parquet

if __name__ == "__main__":
    print("Iniciando a criação do dataset de modelagem final...")

    # 1. Carregar os datasets
    try:
        deputies_df = pd.read_parquet('data/processed/deputies_master_table.parquet')
        votes_df = pd.read_parquet('data/raw/votes.parquet')
        votings_details_df = pd.read_parquet('data/processed/votings_details.parquet')
        print("Datasets carregados com sucesso.")
    except FileNotFoundError as e:
        print(f"Erro: Arquivo não encontrado - {e}.")
        exit()

    # 2. Normalizar o DataFrame de votos
    deputado_details = pd.json_normalize(votes_df['deputado_'])
    deputado_details = deputado_details.rename(columns={'id': 'id_deputado'})
    votes_df = pd.concat([votes_df.drop(columns=['deputado_']), deputado_details], axis=1)

    # 3. Garantir tipos de dados consistentes para o merge
    votes_df['id_votacao'] = votes_df['id_votacao'].astype(str)
    votings_details_df['id_votacao'] = votings_details_df['id_votacao'].astype(str)

    # 4. Juntar votos com detalhes da votação. Usamos 'left' para manter todos os votos.
    modeling_df = pd.merge(votes_df, votings_details_df, on='id_votacao', how='left')

    # 5. Juntar com os dados dos deputados
    modeling_df = pd.merge(modeling_df, deputies_df, on='id_deputado', how='inner')
    print(f"Merge concluído. O dataset de modelagem tem {len(modeling_df)} linhas.")

    # 6. Limpar e filtrar o alvo de previsão
    modeling_df['tipoVoto'] = modeling_df['tipoVoto'].str.strip()
    valid_votes = ['Sim', 'Não']
    modeling_df = modeling_df[modeling_df['tipoVoto'].isin(valid_votes)].copy()
    print(f"Após o filtro por 'Sim' e 'Não', o dataset tem {len(modeling_df)} linhas.")

    # --- TRATAMENTO FINAL DA EMENTA ---
    # Preenche ementas nulas (que sabemos que existem) com um texto substituto.
    modeling_df['proposicao_ementa'].fillna('Ementa não disponível', inplace=True)
    print("Ementas nulas foram preenchidas com um texto padrão.")

    # 7. Selecionar colunas finais
    final_columns = [
        'id_votacao', 'id_deputado', 'proposicao_ementa',
        'nome_urna', 'partido', 'uf', 'idade', 'escolaridade', 'tipoVoto'
    ]
    modeling_df = modeling_df[final_columns]

    # 8. Salvar
    file_path = 'data/processed/modeling_dataset.parquet'
    save_to_parquet(modeling_df, file_path)

    print("\n--- Informações do Dataset de Modelagem Final ---")
    modeling_df.info()
    print("\n--- Amostra do Dataset ---")
    print(modeling_df[['id_votacao', 'partido', 'tipoVoto', 'proposicao_ementa']].head())