import pandas as pd
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

# Descargar los recursos de NLTK
nltk.download('stopwords')
nltk.download('punkt')

# Cargar los datos (sdata.csv) y obtener las primeras 1000 revisiones
data = pd.read_csv('sdata.csv', nrows=1000)

# Extraer las primeras 1000 reseñas
AllReviews = data['text']

# Tokenizar y limpiar las reseñas (eliminar stopwords)
stop_words = set(stopwords.words('english'))
all_text = ' '.join(AllReviews.astype(str))
words = nltk.word_tokenize(all_text.lower())
filtered_words = [word for word in words if word.isalnum() and word not in stop_words]

# Función para obtener los k n-gramas superiores
def top_k_ngrams(word_tokens, n, k):
    # Convertir los tokens en un solo texto
    word_text = ' '.join(word_tokens)
    
    # Usar CountVectorizer para contar los n-gramas
    vec = CountVectorizer(ngram_range=(n, n)).fit([word_text])
    bag_of_words = vec.transform([word_text])
    sum_words = bag_of_words.sum(axis=0)
    
    # Obtener los n-gramas y sus frecuencias
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    
    # Imprimir los k n-gramas más frecuentes
    print(f"Top {k} {n}-grams:")
    for word, freq in words_freq[:k]:
        print(f"{word}: {freq}")
    print("\n")

# Aplicar la función para obtener los 10 mejores 1-gramas, 2-gramas y 3-gramas
top_k_ngrams(filtered_words, 1, 10)  # 1-gramas
top_k_ngrams(filtered_words, 2, 10)  # 2-gramas
top_k_ngrams(filtered_words, 3, 10)  # 3-gramas