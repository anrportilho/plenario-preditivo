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
    print("Iniciando o pipeline de treinamento do modelo...")

    # 1. Carregar o dataset
    try:
        # Tentar carregar o dataset enriquecido; se não existir, usar o básico
        try:
            df = pd.read_parquet('data/processed/modeling_dataset_enriched.parquet')
            print(f"Dataset enriquecido carregado: {len(df)} linhas, {df.shape[1]} colunas.\n")
            use_enriched = True
        except FileNotFoundError:
            df = pd.read_parquet('data/processed/modeling_dataset.parquet')
            print(f"Dataset básico carregado: {len(df)} linhas, {df.shape[1]} colunas.")
            print("(Para melhor desempenho, execute: python src/feature_engineering/enrich_behavioral_features.py)\n")
            use_enriched = False
    except FileNotFoundError:
        print("Erro: Nenhum dataset encontrado.")
        exit()

    # 2. Preparar os dados
    target = 'tipoVoto'

    # Garantir que escolaridade está preenchida
    df['escolaridade'] = df['escolaridade'].fillna('Não Informado')

    # --- Feature Engineering ---
    print("Realizando engenharia de features...")

    # Se temos dataset enriquecido, adicionar as features comportamentais
    if use_enriched:
        X = pd.get_dummies(
            df[['partido', 'posicao_governo', 'uf', 'escolaridade']],
            drop_first=True
        )
        X['idade'] = df['idade'].values
        X['pct_sim_historico'] = df['pct_sim_historico'].values
        X['pct_sim_na_votacao'] = df['pct_sim_na_votacao'].values
        X['pct_sim_uf'] = df['pct_sim_uf'].values
        X['pct_sim_posicao_votacao'] = df['pct_sim_posicao_votacao'].values
    else:
        # Dataset básico: apenas features demográficas
        X = pd.get_dummies(
            df[['partido', 'posicao_governo', 'uf', 'escolaridade']],
            drop_first=True
        )
        X['idade'] = df['idade'].values

    # Criar bins de idade para capturar padrões etários
    faixa_idade = pd.cut(df['idade'],
                         bins=[0, 30, 40, 50, 60, 100],
                         labels=['18-30', '31-40', '41-50', '51-60', '60+'])
    X_faixa = pd.get_dummies(faixa_idade, prefix='faixa_idade', drop_first=True)
    X = pd.concat([X, X_faixa], axis=1)

    print(f"Total de features: {X.shape[1]}")
    if use_enriched:
        print(f"Features: 4 comportamentais + {X.shape[1] - 4} demográficas\n")
    else:
        print(f"Features: apenas demográficas\n")

    # 3. Codificar a variável alvo
    le = LabelEncoder()
    y_encoded = le.fit_transform(df[target])
    print(f"Classes: {list(le.classes_)} -> {list(range(len(le.classes_)))}")
    print(f"Distribuição: {np.bincount(y_encoded)}\n")

    # 4. Dividir os dados (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    print(f"Treino: {X_train.shape[0]} linhas | Teste: {X_test.shape[0]} linhas\n")

    # 5. Treinar o modelo
    print("Treinando LightGBM...")
    model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=7,
        num_leaves=31,
        random_state=42,
        class_weight='balanced',
        verbose=-1
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], callbacks=[lgb.early_stopping(10)])
    print("Treinamento concluído.\n")

    # 6. Salvar artefatos
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/lgbm_model.joblib')
    joblib.dump(le, 'models/label_encoder.joblib')
    joblib.dump(X.columns.tolist(), 'models/feature_columns.joblib')
    print("Modelo salvo em 'models/'\n")

    # 7. Avaliar
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    accuracy_train = accuracy_score(y_train, y_pred_train)
    accuracy_test = accuracy_score(y_test, y_pred_test)

    print("=" * 60)
    print("RESULTADOS")
    print("=" * 60)
    print(f"Acurácia no Treino: {accuracy_train:.4f}")
    print(f"Acurácia no Teste:  {accuracy_test:.4f}\n")

    print("--- Relatório de Classificação ---")
    print(classification_report(y_test, y_pred_test, target_names=le.classes_))

    # 8. Matriz de Confusão
    cm = confusion_matrix(y_test, y_pred_test)
    print("\n--- Matriz de Confusão ---")
    print(f"                Predito: Não  Predito: Sim")
    print(f"Real: Não          {cm[0, 0]:4d}          {cm[0, 1]:4d}")
    print(f"Real: Sim          {cm[1, 0]:4d}          {cm[1, 1]:4d}\n")

    # 9. Feature Importance
    print("--- Top 10 Features Mais Importantes ---")
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(importance_df.head(10).to_string(index=False))