# src/feature_engineering/create_modeling_dataset.py

import pandas as pd
from src.data_collection.api_client import save_to_parquet  # Reutilizamos a função de salvar

if __name__ == "__main__":
    print("Iniciando a criação do dataset de modelagem...")

    # 1. Carregar os datasets
    try:
        deputies_df = pd.read_parquet('data/processed/deputies_master_table.parquet')
        votes_df = pd.read_parquet('data/raw/votes.parquet')
        print("Datasets carregados com sucesso.")
    except FileNotFoundError as e:
        print(f"Erro: Arquivo não encontrado - {e}.")
        print("Certifique-se de que todos os scripts da pasta 'data_collection' foram executados.")
        exit()

    # 2. Processar o DataFrame de votos
    deputado_details = pd.json_normalize(votes_df['deputado_'])
    deputado_details = deputado_details.rename(columns={'id': 'id_deputado'})
    votes_df = pd.concat([votes_df.drop(columns=['deputado_']), deputado_details], axis=1)
    print("DataFrame de votos normalizado.")

    # --- LINHA DE DEPURAÇÃO ---
    # Se o erro de KeyError ocorrer novamente, descomente a linha abaixo.
    # Ela mostrará o nome exato de todas as colunas disponíveis.
    # print("Colunas disponíveis no DataFrame de votos:", votes_df.columns)

    # 3. Juntar (merge) o dataset de deputados com o de votos
    print("Juntando datasets de deputados e votos...")
    modeling_df = pd.merge(votes_df, deputies_df, on='id_deputado', how='inner')
    print(f"Merge concluído. O dataset de modelagem inicial tem {len(modeling_df)} linhas.")

    # 4. Limpar o alvo de previsão (nossa variável 'y')
    valid_votes = ['Sim', 'Não']
    modeling_df = modeling_df[modeling_df['tipoVoto'].isin(valid_votes)]
    print(f"Após filtrar apenas por votos 'Sim' e 'Não', o dataset tem {len(modeling_df)} linhas.")

    # 5. Selecionar e reordenar colunas para clareza
    final_columns = [
        # IDs e informações da votação
        'id_votacao',
        'id_deputado',
        'dataRegistroVoto',  # <-- CORREÇÃO FINAL: O nome correto da coluna é este.

        # Features do Deputado (nossas variáveis 'X')
        'nome_urna',
        'partido',
        'uf',
        'idade',
        'escolaridade',

        # Alvo da Previsão (nossa variável 'y')
        'tipoVoto'
    ]
    modeling_df = modeling_df[final_columns]

    # 6. Salvar o dataset final
    file_path = 'data/processed/modeling_dataset.parquet'
    save_to_parquet(modeling_df, file_path)

    print("\n--- Informações do Dataset de Modelagem Final ---")
    modeling_df.info()
    print("\n--- Amostra do Dataset ---")
    print(modeling_df.head())