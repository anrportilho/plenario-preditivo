# src/modeling/train_model.py

import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import joblib

# --- NOVA IMPORTAÇÃO ---
# Importamos o NLTK para buscar a lista de stop words em português
import nltk
from nltk.corpus import stopwords

if __name__ == "__main__":
    print("Iniciando o pipeline de treinamento do modelo com features de NLP...")

    # 1. Carregar o dataset
    try:
        df = pd.read_parquet('data/processed/modeling_dataset.parquet')
        print(f"Dataset de modelagem carregado com sucesso, contendo {len(df)} linhas.")
    except FileNotFoundError:
        print("Erro: Arquivo 'data/processed/modeling_dataset.parquet' não encontrado.")
        exit()

    # 2. Preparar os dados
    categorical_features = ['partido', 'uf', 'escolaridade']
    text_feature = 'proposicao_ementa'
    target = 'tipoVoto'

    # Correção do FutureWarning: atribuímos o resultado de volta à coluna
    df['escolaridade'] = df['escolaridade'].fillna('Não Informado')

    # --- Pipeline de NLP (CORRIGIDO) ---
    print("Processando a feature de texto com TF-IDF...")

    # Carrega a lista de stop words em português a partir do NLTK
    portuguese_stopwords = stopwords.words('portuguese')

    # Passamos a LISTA de palavras para o parâmetro stop_words
    tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words=portuguese_stopwords)
    X_text = tfidf_vectorizer.fit_transform(df[text_feature])

    # --- Pipeline de Features Categóricas ---
    X_categorical = pd.get_dummies(df[categorical_features], drop_first=True)

    # --- Combinando as Features ---
    X_combined = hstack([X_text, X_categorical.astype(float)])
    print("Features de texto e categóricas foram combinadas.")

    # Codifica a variável alvo
    le = LabelEncoder()
    y_encoded = le.fit_transform(df[target])

    # 3. Dividir os dados
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    print(f"Dados divididos em treino ({X_train.shape[0]} linhas) e teste ({X_test.shape[0]} linhas).")

    # 4. Treinar o modelo
    print("\nTreinando o modelo LightGBM...")
    model = lgb.LGBMClassifier(random_state=42)
    model.fit(X_train, y_train)
    print("Treinamento concluído.")

    # 5. Salvar o modelo e os transformadores
    # (código de salvamento inalterado)
    joblib.dump(model, 'models/lgbm_model.joblib')
    joblib.dump(tfidf_vectorizer, 'models/tfidf_vectorizer.joblib')
    joblib.dump(le, 'models/label_encoder.joblib')
    joblib.dump(X_categorical.columns.to_list(), 'models/categorical_columns.joblib')
    print("Modelo e transformadores foram salvos na pasta 'models/'.")

    # 6. Fazer previsões e avaliar
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=le.classes_)

    print("\n--- Performance do Modelo (COM NLP) no Conjunto de Teste ---")
    print(f"Acurácia: {accuracy:.4f}")
    print("\nRelatório de Classificação:")
    print(report)