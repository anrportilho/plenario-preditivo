# src/data_collection/api_client.py

import requests
import pandas as pd
import os


# URL base da API v2 da Câmara dos Deputados
BASE_URL = "https://dadosabertos.camara.leg.br/api/v2"


def find_next_url(links):
    """Função auxiliar para encontrar o link da próxima página na resposta da API."""
    for link in links:
        if link['rel'] == 'next':
            return link['href']
    return None


def fetch_all_deputies():
    """
    Busca a lista de TODOS os deputados em exercício na API da Câmara,
    navegando por todas as páginas de resultados.

    Returns:
        pandas.DataFrame: Um DataFrame com os dados de todos os deputados.
                          Retorna None em caso de erro.
    """
    all_deputies_list = []
    endpoint = f"{BASE_URL}/deputados"
    params = {
        'ordem': 'ASC',
        'ordenarPor': 'nome',
        'itens': 100  # Máximo de itens por página
    }

    next_url = endpoint
    page_number = 1

    print("Iniciando a busca de todos os deputados (navegando pelas páginas)...")

    try:
        while next_url:
            print(f"Buscando página: {page_number}...")
            # Na primeira iteração, params é usado. Nas seguintes, o next_url já tem os parâmetros.
            response = requests.get(next_url, params=params if page_number == 1 else None)
            response.raise_for_status()

            data = response.json()
            all_deputies_list.extend(data['dados'])

            # Procura pelo link da próxima página
            next_url = find_next_url(data['links'])
            page_number += 1

        # Converte a lista completa de dicionários em um DataFrame
        deputies_df = pd.DataFrame(all_deputies_list)
        print(f"\nSucesso! {len(deputies_df)} deputados encontrados no total.")
        return deputies_df

    except requests.exceptions.RequestException as e:
        print(f"Erro ao fazer a requisição à API: {e}")
        return None
    except KeyError:
        print("Erro: A chave 'dados' ou 'links' não foi encontrada no JSON de resposta.")
        return None

def save_to_parquet(df, file_path):
    """
    Salva um DataFrame em um arquivo Parquet, criando o diretório se não existir.

    Args:
        df (pd.DataFrame): O DataFrame a ser salvo.
        file_path (str): O caminho completo do arquivo (ex: 'data/raw/deputies.parquet').
    """
    try:
        # Extrai o diretório do caminho do arquivo
        directory = os.path.dirname(file_path)
        # Cria o diretório se ele não existir
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Diretório '{directory}' criado.")

        df.to_parquet(file_path, index=False)
        print(f"Dados salvos com sucesso em '{file_path}'")
    except Exception as e:
        print(f"Erro ao salvar o arquivo Parquet: {e}")


# Este bloco será executado apenas quando rodarmos o script diretamente
if __name__ == "__main__":
    deputies_dataframe = fetch_all_deputies()

    if deputies_dataframe is not None:
        # Define o caminho para salvar o arquivo
        file_path = 'data/raw/deputies.parquet'
        save_to_parquet(deputies_dataframe, file_path)

        print("\n--- Informações do DataFrame Salvo ---")
        deputies_dataframe.info()