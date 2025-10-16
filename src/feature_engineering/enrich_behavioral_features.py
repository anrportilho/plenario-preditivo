# src/feature_engineering/enrich_behavioral_features.py

import pandas as pd
import numpy as np
from src.data_collection.api_client import save_to_parquet

if __name__ == "__main__":
    print("Enriquecendo dataset com features comportamentais...\n")

    # 1. Carregar dados
    try:
        # --- CORREÇÃO AQUI ---
        # O caminho correto é relativo à raiz do projeto.
        df = pd.read_parquet('data/processed/modeling_dataset.parquet')
        print(f"Dataset original carregado: {len(df)} linhas\n")
    except FileNotFoundError:
        print("Erro: modeling_dataset.parquet não encontrado.")
        print("Certifique-se de que o script 'create_modeling_dataset.py' foi executado com sucesso.")
        exit()

    # Define os partidos de governo e oposição
    PARTIDOS_GOVERNO = ['PT', 'PCdoB', 'PV', 'PSB', 'MDB', 'PSD', 'REPUBLICANOS', 'PODE', 'UNIÃO', 'PSOL', 'REDE']
    PARTIDOS_OPOSICAO = ['PL', 'PP', 'NOVO']


    def define_posicao(partido):
        if partido in PARTIDOS_GOVERNO:
            return 'Governo'
        elif partido in PARTIDOS_OPOSICAO:
            return 'Oposicao'
        else:
            return 'Independente'


    # Adiciona a feature 'posicao_governo'
    df['posicao_governo'] = df['partido'].apply(define_posicao)

    # 2. Calcular histórico do deputado
    print("Calculando histórico de voto por deputado...")
    deputy_vote_stats = df.groupby('id_deputado')['tipoVoto'].apply(
        lambda x: (x == 'Sim').sum() / len(x) if len(x) > 0 else 0.5
    ).reset_index(name='pct_sim_historico')
    df = pd.merge(df, deputy_vote_stats, on='id_deputado', how='left')

    # 3. Calcular coesão partidária
    print("Calculando coesão partidária por votação...")
    votacao_partido_stats = df.groupby(['id_votacao', 'partido'])['tipoVoto'].apply(
        lambda x: (x == 'Sim').sum() / len(x) if len(x) > 0 else 0.5
    ).reset_index(name='pct_sim_na_votacao')
    df = pd.merge(df, votacao_partido_stats, on=['id_votacao', 'partido'], how='left')

    # 4. Calcular média de voto por UF
    print("Calculando padrão de voto por UF...")
    uf_vote_stats = df.groupby('uf')['tipoVoto'].apply(
        lambda x: (x == 'Sim').sum() / len(x) if len(x) > 0 else 0.5
    ).reset_index(name='pct_sim_uf')
    df = pd.merge(df, uf_vote_stats, on='uf', how='left')

    # 5. Calcular média de voto por posição na votação
    print("Calculando padrão de voto por posição governamental...")
    votacao_posicao_stats = df.groupby(['id_votacao', 'posicao_governo'])['tipoVoto'].apply(
        lambda x: (x == 'Sim').sum() / len(x) if len(x) > 0 else 0.5
    ).reset_index(name='pct_sim_posicao_votacao')
    df = pd.merge(df, votacao_posicao_stats, on=['id_votacao', 'posicao_governo'], how='left')

    # 6. Salvar
    file_path = 'data/processed/modeling_dataset_enriched.parquet'
    save_to_parquet(df, file_path)

    print(f"\n✓ Dataset enriquecido salvo em '{file_path}'")
    print(f"  Colunas adicionadas: pct_sim_historico, pct_sim_na_votacao, pct_sim_uf, pct_sim_posicao_votacao")