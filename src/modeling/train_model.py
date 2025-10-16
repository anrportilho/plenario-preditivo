# src/modeling/train_model.py

import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

if __name__ == "__main__":
    print("Iniciando o pipeline de treinamento do modelo...")

    # 1. Carregar o dataset de modelagem
    try:
        df = pd.read_parquet('data/processed/modeling_dataset.parquet')
        print("Dataset de modelagem carregado com sucesso.")
    except FileNotFoundError:
        print("Erro: Arquivo 'data/processed/modeling_dataset.parquet' não encontrado.")
        print("Certifique-se de que o script 'create_modeling_dataset.py' foi executado.")
        exit()

    # 2. Preparar os dados para o modelo
    # Definir quais colunas são features (X) e qual é o alvo (y)
    features = ['partido', 'uf', 'idade', 'escolaridade']
    target = 'tipoVoto'

    X = df[features]
    y = df[target]

    # O LightGBM consegue lidar com colunas categóricas diretamente, mas elas precisam
    # ser do tipo 'category' no pandas.
    for col in ['partido', 'uf', 'escolaridade']:
        X[col] = X[col].astype('category')

    # A variável alvo ('Sim', 'Não') precisa ser convertida para números (0, 1)
    # Usaremos o LabelEncoder do scikit-learn para isso.
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    # Vamos guardar as classes para referência futura
    # le.classes_ nos dará ['Não', 'Sim'], então 0 = Não, 1 = Sim
    print(f"A variável alvo foi codificada: {list(le.classes_)} -> {list(range(len(le.classes_)))}")

    # 3. Dividir os dados em conjuntos de Treino e Teste
    # Usaremos 80% dos dados para treinar o modelo e 20% para testar seu desempenho.
    # stratify=y_encoded garante que a proporção de 'Sim' e 'Não' seja a mesma nos dois conjuntos.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    print(f"Dados divididos em treino ({len(X_train)} linhas) e teste ({len(X_test)} linhas).")

    # 4. Treinar o modelo LightGBM
    print("\nTreinando o modelo LightGBM...")
    model = lgb.LGBMClassifier(random_state=42)
    model.fit(X_train, y_train)
    print("Treinamento concluído.")

    # 5. Fazer previsões no conjunto de teste
    y_pred = model.predict(X_test)

    # 6. Avaliar a performance do modelo
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=le.classes_)

    print("\n--- Performance do Modelo no Conjunto de Teste ---")
    print(f"Acurácia: {accuracy:.4f}")
    print("\nRelatório de Classificação:")
    print(report)