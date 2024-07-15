# Análise de Sentimento com Pandas e BigQuery

Pessoa, este projeto realiza uma análise de sentimento em um conjunto de comentários gerados sinteticamente usando a biblioteca Faker. A ideia é ilustrar como podemos usar o Pandas e suas bibliotecas com as técnicas de Processamento de Linguagem Natural (PLN) para limpar e analisar os comentários, e então treina um modelo de regressão logística para prever a polaridade (positiva ou negativa) dos comentários. Estas polaridades poderiam em aplicações reais, como por exemplo análise de sentimentos sobre um determinado produto ou serviço, trazer benefícios para a CIA, de forma preditiva, instruindo inclusive à engenharia para melhorar um determinado produto ou serviços em função da polaridade negativa.Os resultados são visualizados através de mapas de palavras e gráficos, e os dados são persistidos no Google BigQuery para futuras análises.

## Passos do Código

1. **Importação de Bibliotecas Necessárias**:

   - As bibliotecas utilizadas incluem `pandas`, `Faker`, `sklearn`, `matplotlib`, `seaborn`, `wordcloud`, `re`, `nltk` e `google-cloud-bigquery`.

2. **Configurações do BigQuery**:

   - Configura as variáveis `project_id` e `dataset_id` para conexão com o Google BigQuery.

3. **Função para Salvar DataFrames no BigQuery**:

   - A função `save_to_bigquery` salva um DataFrame no BigQuery utilizando a API do `google-cloud-bigquery`.

4. **Geração de Dados Sintéticos**:

   - Utiliza a biblioteca Faker para gerar 6000 comentários sintéticos, sendo 2000 positivos e 4000 negativos.

5. **Funções para Limpeza e Tratamento de Texto**:

   - `handle_negation`: Trata negações no texto, adicionando contexto apropriado.
   - `clean_text`: Converte o texto para minúsculas, remove pontuações e aplica a função de tratamento de negações.

6. **Preparação e Limpeza dos Dados**:

   - Aplica as funções de limpeza de texto aos comentários gerados.

7. **Extração de Características com TF-IDF**:

   - Utiliza `TfidfVectorizer` do `sklearn` para transformar os textos em uma matriz TF-IDF.

8. **Divisão dos Dados em Treino e Teste**:

   - Divide os dados em conjuntos de treino e teste utilizando `train_test_split`.

9. **Treinamento do Modelo de Regressão Logística**:

   - Treina um modelo de regressão logística com os dados de treino.

10. **Avaliação do Modelo**:

    - Prediz os rótulos dos dados de teste e calcula a acurácia e o log loss do modelo.

11. **Visualização dos Resultados**:

    - Cria gráficos de barras para visualizar a acurácia e o log loss do modelo.
    - Gera mapas de palavras para visualizar as palavras mais relevantes nos comentários positivos e negativos.
    - Cria gráficos de barras para mostrar as top 10 palavras positivas e negativas.

12. **Persistência dos Dados**:
    - Salva os DataFrames gerados como arquivos CSV.
    - Persiste os DataFrames no Google BigQuery utilizando a função `save_to_bigquery`.

## Requisitos

Certifique-se de ter as seguintes bibliotecas instaladas:

```bash
pip install pandas faker scikit-learn matplotlib seaborn wordcloud nltk google-cloud-bigquery
```
