# src/analysis/test_single_voting.py

import requests
import pprint  # Usaremos pprint para uma visualização bonita do resultado

# Importamos a função que queremos testar
from src.data_collection.enrich_votings_data import fetch_voting_details

if __name__ == "__main__":
    # O ID da votação da Reforma Tributária (PEC 45/2019) que encontramos.
    VOTING_ID_ALVO = "2438687-76"

    print(f"Iniciando teste de coleta de detalhes para a votação ID: {VOTING_ID_ALVO}\n")

    # Chama a função que queremos testar
    details = fetch_voting_details(VOTING_ID_ALVO)

    print("=" * 50)
    print("RESULTADO DA COLETA")
    print("=" * 50)

    if details:
        # Imprime o dicionário completo de forma legível
        pprint.pprint(details)

        print("\n" + "=" * 50)
        print("VEREDITO DA ANÁLISE")
        print("=" * 50)

        ementa = details.get('proposicao_ementa')

        if ementa and "Ementa não disponível" not in ementa:
            print("\n✅ SUCESSO! A ementa foi coletada corretamente.")
            print(f"\nTexto da Ementa: '{ementa[:150]}...'")
        else:
            print("\n⚠️ FALHA PARCIAL. A votação foi encontrada, mas a ementa está vazia ou é processual.")
            print(
                "   Isso confirma que a função de coleta funciona, mas esta votação específica não tem ementa de mérito na API.")

    else:
        print("\n❌ FALHA TOTAL. Não foi possível coletar os detalhes para este ID.")
        print("   Verifique se o ID está correto ou se a API está online.")