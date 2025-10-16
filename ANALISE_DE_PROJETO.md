# Relatório de Projeto: Plenar.io Preditivo

**Data:** 16 de Outubro de 2025
**Autor:** Anderson Portilho
**Versão:** 1.0

## 1. Sumário Executivo

Este documento detalha a jornada de desenvolvimento do projeto "Plenar.io Preditivo", desde sua concepção até o produto final. O objetivo inicial de prever o resultado de votações futuras com base no texto de proposições (NLP) foi estrategicamente pivotado devido a limitações críticas na API de dados abertos da Câmara dos Deputados. O projeto evoluiu para uma poderosa plataforma de **análise comportamental e simulação de cenários políticos**, alcançando uma acurácia de **92.8%** através de uma robusta engenharia de features. Este relatório documenta os desafios encontrados, a estratégia adaptativa adotada e as principais lições aprendidas.

## 2. Objetivos Iniciais

O projeto foi concebido com uma visão ambiciosa e centrada na previsão de eventos futuros. Os objetivos primários eram:

* **Previsão de Votações Futuras:** O principal objetivo era criar uma ferramenta onde um usuário pudesse inserir o texto da ementa de uma proposição que ainda não foi votada. O modelo de Machine Learning, com forte componente de Processamento de Linguagem Natural (NLP), deveria então prever o placar da votação.
* **Análise de Perfil:** Identificar o perfil de votação de cada parlamentar, entendendo sua tendência ideológica com base em seu histórico.
* **Entrega de Valor via Dashboard:** Consolidar todas as funcionalidades em um dashboard interativo com Streamlit, permitindo a exploração dos dados e previsões por jornalistas, analistas e cidadãos.

## 3. Desafios e Limitações da API

A execução do projeto revelou que a realidade dos dados disponíveis impunha desafios significativos, que foram o principal motor para a evolução da estratégia.

* **A Limitação Crítica - Ausência da Ementa:** A descoberta mais impactante foi que o endpoint da API `/votacoes/{id}` **não retorna o texto da ementa**, mesmo para votações de mérito de alta relevância, como a PEC 45/2019 (Reforma Tributária). Nossa investigação cirúrgica com o script `test_single_voting.py` provou que o campo `proposicao_ementa` retornava `None`. Isso invalidou a premissa central do nosso objetivo de previsão baseado em NLP.
* **Dominância de Votações Processuais:** A análise do dataset `modeling_dataset_enriched.parquet` com o script `analyze_ementa_bias.py` revelou que 100% dos dados coletados em um período inicial de 90 dias correspondiam a votações sem texto de ementa, tratando-se, em sua maioria, de pautas processuais.
* **Instabilidade da API:** Durante a coleta de dados, enfrentamos repetidos erros de `504 Server Error: Gateway Timeout` ao tentar buscar dados em janelas de tempo moderadas (30-90 dias). Isso indicou que a API é sensível ao volume de dados requisitado, forçando a adoção de estratégias de coleta mais resilientes.
* **Inconsistências nos Dados:** Depuramos múltiplos problemas de qualidade de dados, como a necessidade de padronizar os tipos de dados das chaves de merge (ex: `id_votacao`) e limpar valores de texto que continham inconsistências.

## 4. A Estratégia Adotada: O Pivô Estratégico

Diante da constatação de que a previsão baseada em NLP não era viável com a fonte de dados disponível, o projeto passou por um **pivô estratégico** consciente e deliberado.

* **De:** Ferramenta de Previsão de Novas Pautas com NLP.
* **Para:** **Plataforma de Análise Comportamental e Simulação de Cenários Políticos.**

Esta nova estratégia focou na verdadeira força do nosso dataset: os padrões de comportamento. As ações foram:

1.  **Foco na Engenharia de Features Comportamentais:** Em vez de depender do texto, desenvolvemos features de alto impacto que capturam a dinâmica política real, como:
    * `pct_sim_na_votacao` (coesão do partido na votação)
    * `pct_sim_historico` (tendência individual do deputado)
    * `pct_sim_posicao_votacao` (alinhamento do bloco político)
    * `posicao_governo` (alinhamento com o governo)

    Estas features foram responsáveis pelo salto de performance do modelo de uma acurácia inicial de ~56% para **92.8%**.

2.  **Desenvolvimento de Scripts de Coleta Resilientes:** Para combater a instabilidade da API, o script `fetch_votings_data.py` foi refatorado para operar em "micro-lotes diários" e ser "resumível", garantindo que a coleta de dados pudesse ser executada por longos períodos sem perda de progresso.

3.  **Refinamento do Produto:** O dashboard foi reestruturado para refletir o novo foco:
    * Removemos a página de previsão de novas pautas, que não era mais suportada pelos dados.
    * Aprimoramos a página principal para funcionar como um poderoso **simulador de cenários**, onde o usuário entende que está usando o comportamento de uma votação passada como referência.
    * Criamos páginas dedicadas à análise granular por votação e por perfil de parlamentar.

## 5. Lições Aprendidas

* **Valide Suas Premissas Antes de Escalar:** A lição mais importante. O teste cirúrgico com `test_single_voting.py` nos economizou horas de processamento e nos deu uma resposta definitiva em minutos. Nunca presuma que uma API fornecerá os dados que você espera.
* **A Engenharia de Features Supera o Algoritmo:** O salto de performance de mais de 35 pontos percentuais não veio da troca do algoritmo, mas da criação de features inteligentes que capturaram a essência do problema.
* **Esteja Preparado para Pivotar:** Um projeto de dados bem-sucedido não é aquele que segue um plano rígido, mas aquele que se adapta às evidências encontradas nos próprios dados. Saber redefinir o objetivo do projeto é uma habilidade sênior.
* **A Importância da Análise de Erros:** Analisar as falhas do modelo (o baixo recall para o voto 'Não') foi crucial para identificar o desbalanceamento de classes e corrigi-lo com o parâmetro `class_weight='balanced'`, resultando em um modelo mais justo e equilibrado.
* **O Valor da Depuração Iterativa:** Cada `KeyError`, `ModuleNotFoundError` ou DataFrame vazio foi um passo essencial no processo de compreensão dos dados e na construção de um pipeline de ETL robusto.