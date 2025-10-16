# src/analysis/analyze_ementa_bias.py

import pandas as pd

if __name__ == "__main__":
    print("Iniciando análise de viés da ementa no dataset...")

    # 1. Carregar o dataset final
    try:
        df = pd.read_parquet('data/processed/modeling_dataset_enriched.parquet')
        print(f"Dataset carregado com sucesso, contendo {len(df)} linhas.")
    except FileNotFoundError:
        print("Erro: Arquivo 'data/processed/modeling_dataset_enriched.parquet' não encontrado.")
        print("Certifique-se de que todo o pipeline de dados foi executado.")
        exit()

    # 2. Identificar os dois grupos
    placeholder_text = "Ementa não disponível"

    df_sem_ementa = df[df['proposicao_ementa'] == placeholder_text]
    df_com_ementa = df[df['proposicao_ementa'] != placeholder_text]

    print("\n" + "=" * 50)
    print("ANÁLISE DO DATASET")
    print("=" * 50)

    # 3. Analisar o grupo SEM ementa real
    print(f"\n--- Grupo 1: Votações SEM Ementa Real ---")
    if not df_sem_ementa.empty:
        total_sem_ementa = len(df_sem_ementa)
        print(f"Total de Votos: {total_sem_ementa}")

        distribuicao_sem_ementa = df_sem_ementa['tipoVoto'].value_counts()
        print("Distribuição de Votos:")
        print(distribuicao_sem_ementa)

        sim_pct = (distribuicao_sem_ementa.get('Sim', 0) / total_sem_ementa) * 100
        nao_pct = (distribuicao_sem_ementa.get('Não', 0) / total_sem_ementa) * 100
        print(f"  - % de Votos 'Sim': {sim_pct:.2f}%")
        print(f"  - % de Votos 'Não': {nao_pct:.2f}%")
    else:
        print("Nenhum voto encontrado sem ementa real.")

    # 4. Analisar o grupo COM ementa real
    print(f"\n--- Grupo 2: Votações COM Ementa Real ---")
    if not df_com_ementa.empty:
        total_com_ementa = len(df_com_ementa)
        print(f"Total de Votos: {total_com_enta}")

        distribuicao_com_ementa = df_com_ementa['tipoVoto'].value_counts()
        print("Distribuição de Votos:")
        print(distribuicao_com_ementa)

        sim_pct = (distribuicao_com_ementa.get('Sim', 0) / total_com_ementa) * 100
        nao_pct = (distribuicao_com_ementa.get('Não', 0) / total_com_ementa) * 100
        print(f"  - % de Votos 'Sim': {sim_pct:.2f}%")
        print(f"  - % de Votos 'Não': {nao_pct:.2f}%")
    else:
        print("Nenhum voto encontrado com ementa real.")

    print("\n" + "=" * 50)
    print("CONCLUSÃO DA ANÁLISE")
    print("=" * 50)
    print("Se a distribuição de votos 'Sim'/'Não' for muito diferente entre os dois grupos,")
    print("isso confirma que o modelo aprendeu um viés baseado na presença ou ausência da ementa,")
    print("em vez de aprender com o seu conteúdo.")