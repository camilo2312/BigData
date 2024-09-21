import nltk
from nltk.corpus import stopwords
import pandas as pd

# Descargar recursos de NLTK
nltk.download('stopwords')
nltk.download('punkt')

# Cargar los datos (sdata.csv)
data = pd.read_csv('sdata.csv', nrows=5000)

# Obtener la primera revisión
first_review = data['text'][0]

# Tokenizar la primera revisión en palabras individuales
words_in_review = nltk.word_tokenize(first_review.lower())

# Obtener las palabras vacías en inglés
stop_words = set(stopwords.words('english'))

# Filtrar palabras vacías y no vacías
stopwords_in_review = [word for word in words_in_review if word in stop_words]
filtered_review = [word for word in words_in_review if word not in stop_words]

# Imprimir las palabras vacías encontradas
print("Palabras vacías encontradas en la primera revisión:")
print(stopwords_in_review)

# Imprimir la revisión sin las palabras vacías
print("\nPrimera revisión sin palabras vacías:")
print(' '.join(filtered_review))
