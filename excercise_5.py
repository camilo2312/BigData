import pandas as pd
import nltk
from collections import Counter
import numpy as np

# Descargar los recursos de NLTK
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords

# Cargar los datos (sdata.csv)
data = pd.read_csv('sdata.csv')

# Filtrar la columna de texto
AllReviews = data['text']

# Paso 1: Tokenizar todas las reseñas y eliminamos las stopwords
stop_words = set(stopwords.words('english'))

# Combinar todas las reseñas en un solo bloque de texto
all_text = ' '.join(AllReviews.astype(str))

# Tokenizar en palabras y eliminar las stopwords
words = nltk.word_tokenize(all_text.lower())
filtered_words = [word for word in words if word.isalnum() and word not in stop_words]

# Paso 2: Contar la frecuencia de cada palabra
word_counts = Counter(filtered_words)

# Total de palabras
total_words = sum(word_counts.values())

# Paso 3: Encontrar el 1% más alto y el 1% más bajo
# Ordenar las palabras por frecuencia
sorted_word_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

# Calcular el umbral del 1%
top_1_percent_index = int(len(sorted_word_counts) * 0.01)
low_1_percent_index = int(len(sorted_word_counts) * 0.99)

# Paso 4: Extraer palabras de alta y baja frecuencia
top_1_percent_words = sorted_word_counts[:top_1_percent_index]
low_1_percent_words = sorted_word_counts[low_1_percent_index:]

# Imprimir las palabras de alta y baja frecuencia
print("Palabras de alta frecuencia (1% más alto):")
for word, count in top_1_percent_words:
    print(f"{word}: {count}")

print("\nPalabras de baja frecuencia (1% más bajo):")
for word, count in low_1_percent_words:
    print(f"{word}: {count}")