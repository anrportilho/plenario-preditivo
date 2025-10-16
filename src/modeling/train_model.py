# src/modeling/train_model.py

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

if __name__ == "__main__":
    print("Iniciando o pipeline de treinamento do modelo (otimizado, sem NLP)...")

    # 1. Carregar o dataset enriquecido
    try:
        df = pd.read_parquet('data/processed/modeling_dataset_enriched.parquet')
        print(f"Dataset enriquecido carregado: {len(df)} linhas.\n")
    except FileNotFoundError:
        print("Erro: Arquivo 'data/processed/modeling_dataset_enriched.parquet' não encontrado.")
        print("Execute o script 'enrich_behavioral_features.py' primeiro.")
        exit()

    # 2. Preparar os dados
    target = 'tipoVoto'
    df['escolaridade'] = df['escolaridade'].fillna('Não Informado')

    # --- Feature Engineering (Apenas Features Comportamentais e Demográficas) ---
    print("Preparando features para o modelo...")
    X = pd.get_dummies(
        df[['partido', 'posicao_governo', 'uf', 'escolaridade']],
        drop_first=True
    )
    X['idade'] = df['idade'].values
    X['pct_sim_historico'] = df['pct_sim_historico'].values
    X['pct_sim_na_votacao'] = df['pct_sim_na_votacao'].values
    X['pct_sim_uf'] = df['pct_sim_uf'].values
    X['pct_sim_posicao_votacao'] = df['pct_sim_posicao_votacao'].values

    faixa_idade = pd.cut(df['idade'], bins=[0, 30, 40, 50, 60, 100], labels=['18-30', '31-40', '41-50', '51-60', '60+'])
    X_faixa = pd.get_dummies(faixa_idade, prefix='faixa_idade', drop_first=True)
    X = pd.concat([X, X_faixa], axis=1)
    print(f"Total de features utilizadas: {X.shape[1]}\n")

    # 3. Codificar a variável alvo
    le = LabelEncoder()
    y_encoded = le.fit_transform(df[target])

    # 4. Dividir os dados
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # 5. Treinar o modelo
    print("Treinando o modelo LightGBM final...")
    model = lgb.LGBMClassifier(random_state=42, class_weight='balanced', verbose=-1)
    model.fit(X_train, y_train)
    print("Treinamento concluído.\n")

    # 6. Salvar artefatos (sem o TfidfVectorizer)
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/lgbm_model.joblib')
    joblib.dump(le, 'models/label_encoder.joblib')
    joblib.dump(X.columns.tolist(), 'models/feature_columns.joblib')
    print("Modelo e artefatos salvos na pasta 'models/'.\n")

    # 7. Avaliar
    y_pred_test = model.predict(X_test)
    print("=" * 60, "\nRESULTADOS DO MODELO FINAL\n", "=" * 60)
    print(f"Acurácia no Teste:  {accuracy_score(y_test, y_pred_test):.4f}\n")
    print(classification_report(y_test, y_pred_test, target_names=le.classes_))