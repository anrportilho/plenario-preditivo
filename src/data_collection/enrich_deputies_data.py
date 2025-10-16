# src/data_collection/enrich_deputies_data.py

import pandas as pd
import requests
import time
from tqdm import tqdm
from api_client import BASE_URL, save_to_parquet  # Reutilizamos nossa URL base e função de salvar!


def fetch_deputy_details(deputy_id):
    """
    Busca os detalhes de um deputado específico na API da Câmara.

    Args:
        deputy_id (int): O ID do deputado.

    Returns:
        dict: Um dicionário com os dados detalhados do deputado, ou None em caso de erro.
    """
    endpoint = f"{BASE_URL}/deputados/{deputy_id}"
    try:
        response = requests.get(endpoint)
        response.raise_for_status()
        return response.json()['dados']
    except requests.exceptions.RequestException as e:
        print(f"Erro ao buscar detalhes para o deputado ID {deputy_id}: {e}")
        return None
    except KeyError:
        print(f"Erro: A chave 'dados' não foi encontrada para o deputado ID {deputy_id}.")
        return None


if __name__ == "__main__":
    # Carrega o dataset de deputados que já coletamos
    try:
        deputies_df = pd.read_parquet('data/raw/deputies.parquet')
    except FileNotFoundError:
        print("Erro: Arquivo 'data/raw/deputies.parquet' não encontrado.")
        print("Por favor, execute o script 'api_client.py' primeiro.")
        exit()

    print(f"Iniciando o enriquecimento dos dados para {len(deputies_df)} deputados.")

    all_details = []
    # Usamos tqdm para criar uma barra de progresso interativa
    for index, row in tqdm(deputies_df.iterrows(), total=deputies_df.shape[0], desc="Buscando detalhes"):
        deputy_id = row['id']
        details = fetch_deputy_details(deputy_id)

        if details:
            # Selecionamos apenas os campos que nos interessam para evitar poluir o dataset
            relevant_details = {
                'id': details.get('id'),
                'nomeCivil': details.get('nomeCivil'),
                'ultimoStatus_nomeEleitoral': details.get('ultimoStatus', {}).get('nomeEleitoral'),
                'ultimoStatus_data': details.get('ultimoStatus', {}).get('data'),
                'dataNascimento': details.get('dataNascimento'),
                'ufNascimento': details.get('ufNascimento'),
                'escolaridade': details.get('escolaridade')
            }
            all_details.append(relevant_details)

        # Pausa pequena para não sobrecarregar a API
        time.sleep(0.1)

    if all_details:
        details_df = pd.DataFrame(all_details)

        # Define o caminho para o novo arquivo. Note o uso da pasta "processed".
        file_path = 'data/processed/deputies_details.parquet'
        save_to_parquet(details_df, file_path)

        print("\n--- Informações do DataFrame de Detalhes Salvo ---")
        details_df.info()
        print("\n--- Amostra dos Dados ---")
        print(details_df.head())
    else:
        print("Nenhum dado detalhado foi coletado.")