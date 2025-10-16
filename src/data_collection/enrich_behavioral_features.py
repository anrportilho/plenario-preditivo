# src/feature_engineering/enrich_behavioral_features.py

import pandas as pd
import numpy as np
from src.data_collection.api_client import save_to_parquet

if __name__ == "__main__":
    print("Enriquecendo dataset com features comportamentais...\n")

    # 1. Carregar dados brutos
    try:
        df = pd.read_parquet('data/processed/modeling_dataset.parquet')
    except FileNotFoundError:
        print("Erro: modeling_dataset.parquet não encontrado.")
        exit()

    print(f"Dataset original: {len(df)} linhas\n")

    # 2. Calcular histórico do deputado
    print("Calculando histórico de voto por deputado...")

    # Para cada deputado, calcular sua taxa histórica de "Sim"
    deputy_vote_stats = df.groupby('id_deputado').agg({
        'tipoVoto': [
            ('total_votos', 'count'),
            ('pct_sim', lambda x: (x == 'Sim').sum() / len(x))
        ]
    }).reset_index()

    deputy_vote_stats.columns = ['id_deputado', 'total_votos_historico', 'pct_sim_historico']

    # Juntar com o dataset principal
    df = pd.merge(df, deputy_vote_stats, on='id_deputado', how='left')

    print(f"  - Taxa de 'Sim' por deputado: min={df['pct_sim_historico'].min():.2f}, "
          f"max={df['pct_sim_historico'].max():.2f}, mean={df['pct_sim_historico'].mean():.2f}")

    # 3. Calcular coesão partidária
    print("\nCalculando coesão partidária...")

    # Para cada votação e partido, qual % votou "Sim"
    votacao_partido_stats = df.groupby(['id_votacao', 'partido']).agg({
        'tipoVoto': [
            ('total', 'count'),
            ('pct_sim_na_votacao', lambda x: (x == 'Sim').sum() / len(x))
        ]
    }).reset_index()

    votacao_partido_stats.columns = ['id_votacao', 'partido', 'num_deputados_votacao', 'pct_sim_na_votacao']

    # Juntar com o dataset principal
    df = pd.merge(df, votacao_partido_stats, on=['id_votacao', 'partido'], how='left')

    print(f"  - % de 'Sim' por votação/partido: min={df['pct_sim_na_votacao'].min():.2f}, "
          f"max={df['pct_sim_na_votacao'].max():.2f}, mean={df['pct_sim_na_votacao'].mean():.2f}")

    # 4. Calcular média de voto por UF
    print("\nCalculando padrão de voto por UF...")

    uf_vote_stats = df.groupby('uf').agg({
        'tipoVoto': lambda x: (x == 'Sim').sum() / len(x)
    }).reset_index()
    uf_vote_stats.columns = ['uf', 'pct_sim_uf']

    df = pd.merge(df, uf_vote_stats, on='uf', how='left')

    print(f"  - % de 'Sim' por UF: min={df['pct_sim_uf'].min():.2f}, "
          f"max={df['pct_sim_uf'].max():.2f}")

    # 5. Criar feature de "posição esperada"
    print("\nCriando feature de posição esperada...")

    df['posicao_governo_num'] = df['posicao_governo'].map({
        'Governo': 1,
        'Oposicao': 0,
        'Independente': 0.5
    })

    # Calcular % de "Sim" por posição governamental e votação
    votacao_posicao_stats = df.groupby(['id_votacao', 'posicao_governo']).agg({
        'tipoVoto': lambda x: (x == 'Sim').sum() / len(x) if len(x) > 0 else 0.5
    }).reset_index()
    votacao_posicao_stats.columns = ['id_votacao', 'posicao_governo', 'pct_sim_posicao_votacao']

    df = pd.merge(df, votacao_posicao_stats, on=['id_votacao', 'posicao_governo'], how='left')

    print(f"  - % de 'Sim' por posição: {df['pct_sim_posicao_votacao'].describe().to_dict()}")

    # 6. Salvar
    file_path = 'data/processed/modeling_dataset_enriched.parquet'
    save_to_parquet(df, file_path)

    print(f"\n✓ Dataset enriquecido salvo em '{file_path}'")
    print(f"  Colunas: {len(df.columns)}")
    print(f"\nNovas features criadas:")
    print(f"  - pct_sim_historico: taxa histórica de 'Sim' do deputado")
    print(f"  - pct_sim_na_votacao: % de 'Sim' do partido nessa votação")
    print(f"  - pct_sim_uf: % de 'Sim' da UF")
    print(f"  - pct_sim_posicao_votacao: % de 'Sim' da posição governamental nessa votação")