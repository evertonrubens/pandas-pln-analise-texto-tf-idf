import pandas as pd
from faker import Faker
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from google.cloud import bigquery

nltk.download('stopwords')
nltk.download('punkt')

# Configurações do BigQuery
project_id = "Id_Project_GCP"  # Substitua pelo seu ID do projeto
dataset_id = "Seu-DataSet_id_Aqui"  # Substitua pelo seu ID do dataset

# Função para salvar DataFrame no BigQuery
def save_to_bigquery(df, table_name):
    client = bigquery.Client(project=project_id)
    table_id = f"{project_id}.{dataset_id}.{table_name}"
    job = client.load_table_from_dataframe(df, table_id)
    job.result()
    print(f"DataFrame salvo na tabela {table_name} do BigQuery.")

# Gerar dados sintéticos
fake = Faker()
Faker.seed(42)

# Função para gerar frases positivas e negativas
def generate_sentence(sentiment):
    positive_phrases = [
        "O sistema de gestão de APIs é excelente.",
        "A interface é muito intuitiva e fácil de usar.",
        "O desempenho do sistema é excepcional.",
        "Estou muito satisfeito com as funcionalidades oferecidas.",
        "O suporte ao cliente é rápido e eficiente.",
        "As integrações são perfeitas e sem falhas.",
        "A documentação é clara e detalhada.",
        "O sistema melhora nossa produtividade.",
        "A segurança do sistema é robusta.",
        "Recomendo este sistema para todos."
    ]
    
    negative_phrases = [
        "O sistema de gestão de APIs é péssimo.",
        "A interface é confusa e difícil de usar.",
        "O desempenho do sistema é decepcionante.",
        "Estou muito insatisfeito com as funcionalidades oferecidas.",
        "O suporte ao cliente é lento e ineficiente.",
        "As integrações são problemáticas e cheias de falhas.",
        "A documentação é confusa e pouco detalhada.",
        "O sistema reduz nossa produtividade.",
        "A segurança do sistema é fraca.",
        "Não recomendo este sistema para ninguém."
    ]
    
    if sentiment == 1:
        return fake.sentence(ext_word_list=positive_phrases)
    else:
        return fake.sentence(ext_word_list=negative_phrases)

# Criar DataFrame com 6.000 comentários (2.000 positivos e 4.000 negativos)
data = {
    'text': [generate_sentence(1) if i < 2000 else generate_sentence(0) for i in range(6000)],
    'category': [1 if i < 2000 else 0 for i in range(6000)]
}

df = pd.DataFrame(data)
print("DataFrame inicial:")
print(df.head())
print("\nDataFrame com 6.000 comentários:")
print(df)

# Separar frases positivas e negativas em DataFrames diferentes
positive_phrases = df[df['category'] == 1].reset_index(drop=True)
negative_phrases = df[df['category'] == 0].reset_index(drop=True)

print("\nDataFrame de frases positivas:")
print(positive_phrases.head())
print("\nDataFrame de frases negativas:")
print(negative_phrases.head())

# Função para tratar negações e ajustar o contexto
def handle_negation(text):
    negations = ["não", "nunca", "jamais", "nem"]
    tokens = word_tokenize(text)
    new_tokens = []
    negate = False
    for token in tokens:
        if token in negations:
            negate = True
        elif token in ['.', '!', '?']:
            negate = False
        elif negate:
            if token in ["interface", "produtividade", "funcionalidades", "gestão", "integrações", "usar"]:
                if token == "interface":
                    new_tokens.append("não tem boa interface")
                elif token == "produtividade":
                    new_tokens.append("não é produtiva")
                elif token == "funcionalidades":
                    new_tokens.append("não tem boas funcionalidades")
                elif token == "gestão":
                    new_tokens.append("não tem gestão")
                elif token == "integrações":
                    new_tokens.append("difícil integração")
                elif token == "usar":
                    new_tokens.append("difícil usar")
            else:
                new_tokens.append("não é " + token)
        else:
            new_tokens.append(token)
    return " ".join(new_tokens)

# Limpeza de texto e tratamento de negações
def clean_text(text):
    text = text.lower()  # Converte para minúsculas
    text = re.sub(r'[^\w\s]', '', text)  # Remove pontuações
    text = handle_negation(text)
    return text

df['cleaned_text'] = df['text'].apply(clean_text)
print("\nDataFrame após limpeza de texto:")
print(df.head())

# Extração de características com TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['cleaned_text'])
print("\nMatriz TF-IDF:")
print(X.toarray())

# Divisão dos dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, df['category'], test_size=0.3, random_state=42)

# Treinamento de um modelo de regressão logística
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predição e avaliação do modelo
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
loss = log_loss(y_test, model.predict_proba(X_test))
print(f"\nAcurácia do modelo: {accuracy}")
print(f"Log Loss do modelo: {loss}")

# Plotar gráficos de acurácia e log loss
fig, ax = plt.subplots(1, 2, figsize=(14, 5))

# Acurácia
ax[0].barh(['Acurácia'], [accuracy], color='blue')
ax[0].set_xlim(0, 1)
ax[0].set_title('Acurácia do Modelo')
ax[0].set_xlabel('Acurácia')
ax[0].set_ylabel('')

# Log Loss
ax[1].barh(['Log Loss'], [loss], color='red')
ax[1].set_xlim(0, 1)
ax[1].set_title('Log Loss do Modelo')
ax[1].set_xlabel('Log Loss')
ax[1].set_ylabel('')

plt.tight_layout()
plt.show()

# Extrair as palavras mais relevantes
feature_names = vectorizer.get_feature_names_out()
coefficients = model.coef_.flatten()
word_importance = pd.DataFrame({'word': feature_names, 'importance': coefficients})
word_importance = word_importance.sort_values(by='importance', ascending=False)

# Separar as palavras mais importantes por categoria
positive_words = word_importance[word_importance['importance'] > 0]
negative_words = word_importance[word_importance['importance'] < 0]

print("\nPalavras Positivas:")
print(positive_words.head(20))
print("\nPalavras Negativas:")
print(negative_words.head(20))

# Gerar um mapa de palavras para as palavras mais relevantes
wordcloud_positive = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(dict(zip(positive_words['word'], positive_words['importance'])))
wordcloud_negative = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(dict(zip(negative_words['word'], negative_words['importance'])))

# Visualizar o mapa de palavras para palavras positivas
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_positive, interpolation='bilinear')
plt.axis('off')
plt.title('Mapa de Palavras Positivas')
plt.show()

# Visualizar o mapa de palavras para palavras negativas
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_negative, interpolation='bilinear')
plt.axis('off')
plt.title('Mapa de Palavras Negativas')
plt.show()

# Criar tabelas visuais com os resultados
positive_counts = positive_words.copy()
positive_counts['count'] = positive_counts['importance'].abs()

negative_counts = negative_words.copy()
negative_counts['count'] = negative_counts['importance'].abs()

positive_counts = positive_counts.sort_values(by='count', ascending=False).head(10)
negative_counts = negative_counts.sort_values(by='count', ascending=False).head(10)

print("\nTabela de Palavras Positivas:")
print(positive_counts)
print("\nTabela de Palavras Negativas:")
print(negative_counts)

# Visualizar as tabelas de palavras mais relevantes
fig, axs = plt.subplots(2, 1, figsize=(10, 10))
sns.barplot(x='count', y='word', data=positive_counts, ax=axs[0], palette='viridis')
axs[0].set_title('Top 10 Palavras Positivas')
axs[0].set_xlabel('Importância')
axs[0].set_ylabel('Palavra')

sns.barplot(x='count', y='word', data=negative_counts, ax=axs[1], palette='viridis')
axs[1].set_title('Top 10 Palavras Negativas')
axs[1].set_xlabel('Importância')
axs[1].set_ylabel('Palavra')

plt.tight_layout()
plt.show()

# Salvar DataFrames como arquivos CSV e no BigQuery
df.to_csv('src/dataset/dataframe_inicial.csv', index=False)
positive_phrases.to_csv('src/dataset/dataframe_frases_positivas.csv', index=False)
negative_phrases.to_csv('src/dataset/dataframe_frases_negativas.csv', index=False)
positive_counts.to_csv('src/dataset/dataframe_palavras_positivas.csv', index=False)
negative_counts.to_csv('src/dataset/dataframe_palavras_negativas.csv', index=False)

# Salvar DataFrames no BigQuery
save_to_bigquery(df, "dataframe_inicial")
save_to_bigquery(positive_phrases, "dataframe_frases_positivas")
save_to_bigquery(negative_phrases, "dataframe_frases_negativas")
save_to_bigquery(positive_counts, "dataframe_palavras_positivas")
save_to_bigquery(negative_counts, "dataframe_palavras_negativas")