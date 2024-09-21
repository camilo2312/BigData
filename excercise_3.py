import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# Función para construir la nube de palabras
def word_cloud_rating(data, star_value):
    # Filtrar los datos por el valor de las estrellas
    filtered_data = data[data['stars'] == star_value]
    
    # Combinar todas las reseñas filtradas en un solo texto
    word_cloud_text = ' '.join(filtered_data['text'].astype(str))

    # Contruye la nube de palabras
    # Eliminar stopwords en inglés
    stop_words = set(stopwords.words('english'))
    
    wordcloud = WordCloud(stopwords=stop_words, max_font_size=100, 
                          max_words=100, background_color="white", 
                          scale=10, width=800, height=400).generate(word_cloud_text)
    
    # Imprime o muestra en pantalla la nube de palabras
    plt.figure(figsize=(10,5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()

# Cargar los datos (sdata.csv)
data = pd.read_csv('sdata.csv')

# Visualizar la nube de palabras para reseñas de 1 estrella
word_cloud_rating(data, 1)