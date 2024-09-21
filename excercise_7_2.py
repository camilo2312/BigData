import pandas as pd
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

# Descargar los recursos de NLTK (si no están descargados ya)
nltk.download('stopwords')
nltk.download('punkt')

# Obtener las stopwords en inglés
stop_words = set(stopwords.words('english'))

# Función modificada para obtener los k n-gramas superiores sin palabras vacías
def top_k_ngrams(word_tokens, n, k):
    # Filtrar las palabras vacías de los tokens
    filtered_tokens = [word for word in word_tokens if word not in stop_words]
    
    # Transformar los tokens filtrados en un solo texto
    word_text = ' '.join(filtered_tokens)
    
    # Usar CountVectorizer para contar los n-gramas de las palabras vacías
    vec = CountVectorizer(ngram_range=(n, n)).fit([word_text])
    bag_of_words = vec.transform([word_text])
    sum_words = bag_of_words.sum(axis=0)
    
    # Obtener los n-gramas y sus frecuencias
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    
    # Imprimir por consola los k n-gramas más frecuentes
    print(f"Top {k} {n}-grams sin palabras vacías:")
    for word, freq in words_freq[:k]:
        print(f"{word}: {freq}")
    print("\n")

# Ejemplo con 1000 datos de cómo usar la función con las primeras 1000 revisiones
data = pd.read_csv('sdata.csv', nrows=1000)
AllReviews = data['text']

# Tokenización y preparación de palabras
all_text = ' '.join(AllReviews.astype(str))
words = nltk.word_tokenize(all_text.lower())

# Obtener los 10 mejores 1-gramas, 2-gramas y 3-gramas, eliminando palabras vacías
top_k_ngrams(words, 1, 10)  # 1-gramas
top_k_ngrams(words, 2, 10)  # 2-gramas
top_k_ngrams(words, 3, 10)  # 3-gramas
