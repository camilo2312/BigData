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

# Filtrar palabras vacías
stop_words = list(stopwords.words('english'))

# Función para obtener las palabras más frecuentes
def get_top_n_words(corpus, n=20):
    vec = CountVectorizer(stop_words=stop_words).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:n]

# Dividir las críticas en buneas y malas buenas (>3 estrellas) y malas (<=3 estrellas)
good_reviews = data[data['stars'] > 3]['text']
bad_reviews = data[data['stars'] <= 3]['text']

# Obtener las 20 palabras más frecuentes en "buenas críticas"
top_good_words = get_top_n_words(good_reviews, 20)

# Obtener las 20 palabras más frecuentes en "malas críticas"
top_bad_words = get_top_n_words(bad_reviews, 20)

# Crear DataFrames para cada conjunto de palabras
df_good = pd.DataFrame(top_good_words, columns=['Word', 'Frequency'])
df_bad = pd.DataFrame(top_bad_words, columns=['Word', 'Frequency'])

# Grafica de las mejores 20 palabras
plt.figure(figsize=(10, 6))
df_good.groupby('Word').sum()['Frequency'].sort_values(ascending=False).plot(
    kind='bar', title='Top 20 palabras "Buenas Criticas"')
plt.xticks(rotation=45)
plt.show()

# Grafica de las peores 20 palabras
plt.figure(figsize=(10, 6))
df_bad.groupby('Word').sum()['Frequency'].sort_values(ascending=False).plot(
    kind='bar', title='Top 20 palarbas "Malas criticas"')
plt.xticks(rotation=45)
plt.show()
