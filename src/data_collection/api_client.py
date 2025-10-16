# src/data_collection/api_client.py

import requests
import pandas as pd

# URL base da API v2 da Câmara dos Deputados
BASE_URL = "https://dadosabertos.camara.leg.br/api/v2"

def fetch_all_deputies():
    """
    Busca a lista de todos os deputados em exercício na API da Câmara.

    Returns:
        pandas.DataFrame: Um DataFrame com os dados dos deputados.
                          Retorna None em caso de erro.
    """
    # O endpoint para deputados. Adicionamos parâmetros para ordenar e garantir
    # que pegamos um número suficiente na primeira chamada.
    endpoint = f"{BASE_URL}/deputados"
    params = {
        'ordem': 'ASC',
        'ordenarPor': 'nome',
        'itens': 100  # A API tem um limite por página, 100 é o máximo.
    }

    print("Buscando dados dos deputados na API...")

    try:
        response = requests.get(endpoint, params=params)
        # Lança um erro se a requisição falhou (ex: status code 404, 500)
        response.raise_for_status()

        data = response.json()

        # A API retorna os dados dentro de uma chave "dados"
        deputies_df = pd.DataFrame(data['dados'])
        print(f"Sucesso! {len(deputies_df)} deputados encontrados na primeira página.")
        return deputies_df

    except requests.exceptions.RequestException as e:
        print(f"Erro ao fazer a requisição à API: {e}")
        return None
    except KeyError:
        print("Erro: A chave 'dados' não foi encontrada no JSON de resposta.")
        return None

# Este bloco será executado apenas quando rodarmos o script diretamente
if __name__ == "__main__":
    deputies_dataframe = fetch_all_deputies()

    if deputies_dataframe is not None:
        print("\n--- Amostra dos Dados dos Deputados ---")
        # Mostra as 5 primeiras linhas do DataFrame
        print(deputies_dataframe.head())

        print("\n--- Informações do DataFrame ---")
        # Mostra informações como número de linhas, colunas e tipos de dados
        deputies_dataframe.info()