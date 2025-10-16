# src/data_collection/fetch_votings_data.py

import requests
import pandas as pd
from datetime import datetime, timedelta
import time
from tqdm import tqdm
import os

from src.data_collection.api_client import BASE_URL, find_next_url, save_to_parquet


# As funções fetch_votings_list e fetch_votes_for_voting permanecem as mesmas.
def fetch_votings_list(start_date, end_date):
    # ... (código da função inalterado) ...
    all_votings_list = []
    endpoint = f"{BASE_URL}/votacoes"
    params = {'dataInicio': start_date, 'dataFim': end_date, 'ordem': 'DESC', 'ordenarPor': 'dataHoraRegistro',
              'itens': 100}
    next_url = endpoint
    page_number = 1
    # print(f"Buscando lista de votações entre {start_date} e {end_date}...") # Desativado para ser menos verboso
    try:
        while next_url:
            response = requests.get(next_url, params=params if page_number == 1 else None)
            response.raise_for_status()
            data = response.json()
            valid_votings = [v for v in data['dados'] if v.get('descricao')]
            all_votings_list.extend(valid_votings)
            next_url = find_next_url(data['links'])
            page_number += 1
        return pd.DataFrame(all_votings_list)
    except requests.exceptions.RequestException:
        # Silenciamos o erro para o loop principal continuar
        return None


def fetch_votes_for_voting(voting_id):
    # ... (código da função inalterado) ...
    endpoint = f"{BASE_URL}/votacoes/{voting_id}/votos"
    try:
        response = requests.get(endpoint)
        response.raise_for_status()
        votes_data = response.json()['dados']
        for vote in votes_data:
            vote['id_votacao'] = voting_id
        return votes_data
    except requests.exceptions.RequestException:
        return []


# --- LÓGICA PRINCIPAL REFEITA PARA MICRO-LOTES DIÁRIOS ---
if __name__ == "__main__":
    DAYS_TO_FETCH = 90  # Agora podemos usar um período longo!
    VOTES_FILE_PATH = 'data/raw/votes.parquet'

    # Carrega o progresso existente
    existing_votes_df = pd.DataFrame()
    if os.path.exists(VOTES_FILE_PATH):
        print(f"Arquivo de votos existente encontrado. Carregando...")
        existing_votes_df = pd.read_parquet(VOTES_FILE_PATH)
        processed_voting_ids = set(existing_votes_df['id_votacao'].unique())
        print(f"{len(processed_voting_ids)} votações já processadas.")
    else:
        processed_voting_ids = set()

    # Cria uma lista de datas para iterar, de hoje para trás
    date_range = [datetime.now() - timedelta(days=x) for x in range(DAYS_TO_FETCH)]

    all_new_votes = []

    # Loop principal que itera dia a dia
    print(f"\nIniciando busca de votos em micro-lotes diários para os últimos {DAYS_TO_FETCH} dias...")
    for single_date in tqdm(date_range, desc="Processando dias"):
        date_str = single_date.strftime('%Y-%m-%d')

        # 1. Busca votações APENAS para o dia atual
        votings_of_the_day_df = fetch_votings_list(start_date=date_str, end_date=date_str)

        if votings_of_the_day_df is None or votings_of_the_day_df.empty:
            continue  # Pula para o próximo dia se não houver votações ou ocorrer um erro

        # 2. Filtra IDs que ainda não foram processados
        target_voting_ids = set(votings_of_the_day_df['id'])
        new_voting_ids_to_fetch = list(target_voting_ids - processed_voting_ids)

        if not new_voting_ids_to_fetch:
            continue  # Pula se todas as votações deste dia já foram processadas

        # 3. Busca os votos para os novos IDs
        for voting_id in new_voting_ids_to_fetch:
            votes = fetch_votes_for_voting(voting_id)
            if votes:
                all_new_votes.extend(votes)
                processed_voting_ids.add(voting_id)  # Adiciona ao set para evitar reprocessamento na mesma rodada
            time.sleep(0.05)

    # 4. Salva o resultado final
    if all_new_votes:
        new_votes_df = pd.DataFrame(all_new_votes)
        combined_votes_df = pd.concat([existing_votes_df, new_votes_df], ignore_index=True)

        save_to_parquet(combined_votes_df, VOTES_FILE_PATH)

        print("\n--- Processamento Concluído ---")
        print(f"{len(all_new_votes)} novos registros de votos foram coletados.")
        print(f"O arquivo final contém agora {len(combined_votes_df)} registros no total.")
        print(f"Dados salvos em '{VOTES_FILE_PATH}'.")
    else:
        print("\nNenhum voto novo encontrado no período. O arquivo de dados está atualizado.")