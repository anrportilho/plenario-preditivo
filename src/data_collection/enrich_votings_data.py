# src/data_collection/enrich_votings_data.py

import pandas as pd
import requests
import time
from tqdm import tqdm
import os

from src.data_collection.api_client import BASE_URL, save_to_parquet


def fetch_voting_details(voting_id):
    """
    Busca os detalhes de uma votação específica, focando na ementa da proposição.

    Args:
        voting_id (str): O ID da votação (ex: '2438687-76').

    Returns:
        dict: Um dicionário com os detalhes extraídos, ou None em caso de erro.
    """
    endpoint = f"{BASE_URL}/votacoes/{voting_id}"
    try:
        response = requests.get(endpoint)
        response.raise_for_status()
        details = response.json()['dados']

        # A ementa está dentro de um objeto 'proposicao'
        proposicao = details.get('proposicao', {})

        return {
            'id_votacao': details.get('id'),
            'data': details.get('data'),
            'descricao': details.get('descricao'),
            'proposicao_id': proposicao.get('id'),
            'proposicao_ementa': proposicao.get('ementa')
        }
    except requests.exceptions.RequestException:
        # Silencia o erro para o loop continuar
        return None


if __name__ == "__main__":
    INPUT_FILE = 'data/raw/votings_list.parquet'
    OUTPUT_FILE = 'data/processed/votings_details.parquet'

    # Carrega a lista de votações que precisamos processar
    try:
        votings_list_df = pd.read_parquet(INPUT_FILE)
    except FileNotFoundError:
        print(f"Erro: Arquivo '{INPUT_FILE}' não encontrado.")
        print("Por favor, execute 'fetch_votings_data.py' primeiro.")
        exit()

    # Lógica "resumível": verifica o que já foi processado
    existing_details_df = pd.DataFrame()
    if os.path.exists(OUTPUT_FILE):
        print(f"Arquivo de detalhes existente encontrado. Carregando...")
        existing_details_df = pd.read_parquet(OUTPUT_FILE)
        processed_ids = set(existing_details_df['id_votacao'])
        print(f"{len(processed_ids)} votações já foram enriquecidas.")
    else:
        processed_ids = set()

    # Filtra para buscar apenas as votações que ainda não foram processadas
    target_ids = set(votings_list_df['id'])
    new_ids_to_fetch = list(target_ids - processed_ids)

    if not new_ids_to_fetch:
        print("\nTodos os detalhes de votação já foram coletados. Nenhuma ação necessária.")
    else:
        print(f"\nDas {len(target_ids)} votações na lista, {len(new_ids_to_fetch)} novas serão processadas.")

        all_new_details = []
        for voting_id in tqdm(new_ids_to_fetch, desc="Buscando detalhes das votações"):
            details = fetch_voting_details(voting_id)
            if details:
                all_new_details.append(details)
            time.sleep(0.05)

        if all_new_details:
            new_details_df = pd.DataFrame(all_new_details)
            # Combina os resultados novos com os antigos, se houver
            combined_df = pd.concat([existing_details_df, new_details_df], ignore_index=True)

            save_to_parquet(combined_df, OUTPUT_FILE)

            print("\n--- Processamento Concluído ---")
            print(f"{len(new_details_df)} novos detalhes de votação foram coletados.")
            print(f"O arquivo final contém agora {len(combined_df)} registros.")
            print(f"Dados salvos em '{OUTPUT_FILE}'.")
        else:
            print("Nenhum detalhe novo foi coletado.")