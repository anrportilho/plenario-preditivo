# src/feature_engineering/build_features.py

import pandas as pd
from datetime import datetime

# Importamos nossa função de salvar para reutilizá-la
from src.data_collection.api_client import save_to_parquet


def calculate_age(birth_date):
    """Calcula a idade a partir de uma data de nascimento."""
    if pd.isna(birth_date):
        return None
    birth_date = pd.to_datetime(birth_date)
    today = datetime.today()
    # Calcula a idade subtraindo o ano e ajustando se o aniversário ainda não passou
    age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
    return age


if __name__ == "__main__":
    print("Iniciando a criação da tabela mestra de deputados...")

    # Carrega os dois datasets que criamos
    try:
        deputies_basic_df = pd.read_parquet('data/raw/deputies.parquet')
        deputies_details_df = pd.read_parquet('data/processed/deputies_details.parquet')
    except FileNotFoundError as e:
        print(f"Erro: Arquivo não encontrado - {e}.")
        print("Certifique-se de que os scripts da pasta 'data_collection' foram executados com sucesso.")
        exit()

    # Junta os dois DataFrames pela coluna 'id'
    # Usamos 'inner' para garantir que apenas deputados presentes em ambos os arquivos sejam mantidos
    print("Juntando datasets básico e de detalhes...")
    master_df = pd.merge(deputies_basic_df, deputies_details_df, on='id')

    # --- Engenharia de Features ---
    print("Realizando a engenharia de features...")

    # 1. Converter colunas de data
    master_df['dataNascimento'] = pd.to_datetime(master_df['dataNascimento'])
    master_df['ultimoStatus_data'] = pd.to_datetime(master_df['ultimoStatus_data'])

    # 2. Calcular a idade
    master_df['idade'] = master_df['dataNascimento'].apply(calculate_age)

    # 3. Selecionar e renomear colunas para clareza
    final_columns = {
        'id': 'id_deputado',
        'nome': 'nome_urna',
        'nomeCivil': 'nome_civil',
        'siglaPartido': 'partido',
        'siglaUf': 'uf',
        'idade': 'idade',
        'ufNascimento': 'uf_nascimento',
        'escolaridade': 'escolaridade',
        'dataNascimento': 'data_nascimento',
        'ultimoStatus_data': 'data_status',
        'idLegislatura': 'id_legislatura'
    }
    master_df = master_df[final_columns.keys()].rename(columns=final_columns)

    print("Tabela mestra criada com sucesso!")

    # Salva o resultado final
    file_path = 'data/processed/deputies_master_table.parquet'
    save_to_parquet(master_df, file_path)

    print("\n--- Informações da Tabela Mestra ---")
    master_df.info()
    print("\n--- Amostra dos Dados ---")
    print(master_df.head())