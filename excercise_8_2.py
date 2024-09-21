import pandas as pd
import nltk
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from nltk.corpus import stopwords

# Descargar recursos de NLTK
nltk.download('stopwords')
nltk.download('punkt')

# Cargar el dataset
data = pd.read_csv('sdata.csv', nrows=5000)

# Convertir el conjunto de palabras vacías en una lista
stop_words = list(stopwords.words('english'))

# Función para obtener los n-gramas más frecuentes
def get_top_n_words(corpus, n=20, ngram_range=(1, 1)):
    # Ajustar el rango de n-gramas (bigrama, trigrama, etc.)
    vec = CountVectorizer(stop_words=stop_words, ngram_range=ngram_range).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:n]

# Dividir las críticas en "buenas" (>3 estrellas) y "malas" (<=3 estrellas)
good_reviews = data[data['stars'] > 3]['text']
bad_reviews = data[data['stars'] <= 3]['text']

# Obtener los 20 bigramas más frecuentes en "buenas críticas"
top_good_bigrams = get_top_n_words(good_reviews, 20, ngram_range=(2, 2))

# Obtener los 20 bigramas más frecuentes en "malas críticas"
top_bad_bigrams = get_top_n_words(bad_reviews, 20, ngram_range=(2, 2))

# Obtener los 20 trigramas más frecuentes en "buenas críticas"
top_good_trigrams = get_top_n_words(good_reviews, 20, ngram_range=(3, 3))

# Obtener los 20 trigramas más frecuentes en "malas críticas"
top_bad_trigrams = get_top_n_words(bad_reviews, 20, ngram_range=(3, 3))

# Crear DataFrames para cada conjunto de n-gramas
df_good_bigrams = pd.DataFrame(top_good_bigrams, columns=['Bigram', 'Frequency'])
df_bad_bigrams = pd.DataFrame(top_bad_bigrams, columns=['Bigram', 'Frequency'])
df_good_trigrams = pd.DataFrame(top_good_trigrams, columns=['Trigram', 'Frequency'])
df_bad_trigrams = pd.DataFrame(top_bad_trigrams, columns=['Trigram', 'Frequency'])

# Gráfica top de 20 bigramas en buenas críticas
plt.figure(figsize=(10, 6))
df_good_bigrams.groupby('Bigram').sum()['Frequency'].sort_values(ascending=False).plot(
    kind='bar', title='Top 20 Bigramas "Buenas críticas"')
plt.xticks(rotation=45)
plt.show()

# Gráfica top de 20 bigramas en malas críticas
plt.figure(figsize=(10, 6))
df_bad_bigrams.groupby('Bigram').sum()['Frequency'].sort_values(ascending=False).plot(
    kind='bar', title='Top 20 Bigramas "Malas críticas"')
plt.xticks(rotation=45)
plt.show()

# Gráfica top de 20 trigramas en buenas críticas
plt.figure(figsize=(10, 6))
df_good_trigrams.groupby('Trigram').sum()['Frequency'].sort_values(ascending=False).plot(
    kind='bar', title='Top 20 Trigramas "Buenas críticas"')
plt.xticks(rotation=45)
plt.show()

# Gráfica top de 20 trigramas en malas críticas
plt.figure(figsize=(10, 6))
df_bad_trigrams.groupby('Trigram').sum()['Frequency'].sort_values(ascending=False).plot(
    kind='bar', title='Top 20 Trigramas "Malas críticas"')
plt.xticks(rotation=45)
plt.show()


# ¿Son los resultados útiles?
# Bigramas y trigramas:
# Los bigramas y trigramas pueden proporcionar contexto adicional al mostrar combinaciones comunes de palabras que aparecen juntas, 
# como "great service" o "bad experience".

# En buenas críticas, es probable que encuentres combinaciones positivas como "friendly staff" o "delicious food".

# En malas críticas, podrías encontrar bigramas o trigramas negativos como "bad service" o "wait long time".

# Al comparar estos resultados con las palabras individuales más comunes, los bigramas y trigramas ofrecen una 
# visión más detallada del contexto y pueden ayudar a detectar frases recurrentes tanto en las experiencias 
# positivas como en las negativas.