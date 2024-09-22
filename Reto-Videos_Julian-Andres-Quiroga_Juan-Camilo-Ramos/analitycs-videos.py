import pandas as pd
import matplotlib.pyplot as plt
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tkinter as tk
from tkinter import ttk

# Cargar el dataset
data_videos = pd.read_csv('CAVideos.csv', nrows=5000)

# Filtrar títulos vacíos o que son muy cortos, y adicionalmente eliminar duplicados
data_videos = data_videos[data_videos['title'].str.len() > 5].drop_duplicates(subset=['title'])

# Obtener los títulos de tods los videos
all_titles = data_videos['title'].astype(str)

# Función para obtener el título seleccionado y cerrar la ventana
def on_select(event):
    global title_to_compare
    title_to_compare = combobox.get()  # Obtener el título seleccionado
    root.quit()  # Cerrar la ventana emergente

# Crear ventana emergente con tkinter
root = tk.Tk()
root.title("Seleccionar Título")

# Etiqueta para indicar la acción al usuario
label = ttk.Label(root, text="Seleccione un título para comparar su similitud:")
label.pack(pady=10)

# Crear lista desplegable con los primeros 50 títulos
combobox = ttk.Combobox(root, values=[f"{i + 1}: {title}" for i, title in enumerate(all_titles[:50])], width=80)
combobox.pack(pady=10)
combobox.bind("<<ComboboxSelected>>", on_select)  # Evento para detectar cuando se selecciona un título

# Botón para confirmar la selección
button = ttk.Button(root, text="Seleccionar", command=on_select)
button.pack(pady=10)

# Iniciar ventana
root.mainloop()

# Verificar si el usuario seleccionó un título
if 'title_to_compare' not in globals():
    print("No se seleccionó ningún título.")
    exit()

# Imprimir el título seleccionado
print(f"Título seleccionado: {title_to_compare}")

# Remover el índice del título seleccionado
title_to_compare = title_to_compare.split(": ", 1)[1]  # Solo dejar el título sin el índice

# Definir stopwords
stop_words = list(stopwords.words('english'))

# Inicializar el vectorizador TF-IDF
vectorizer_tfidf = TfidfVectorizer(stop_words=stop_words)

# Calcular la matriz TF-IDF de todos los títulos
tfidf_matrix = vectorizer_tfidf.fit_transform(all_titles)

# Transformar el título seleccionado en su vector TF-IDF
title_vector = vectorizer_tfidf.transform([title_to_compare])

# Calcular similitudes de coseno entre el título seleccionado y los demás títulos
cosine_similarities = cosine_similarity(title_vector, tfidf_matrix).flatten()

# Obtener los índices de los 10 títulos más similares, excluyendo el título seleccionado
top_indices = cosine_similarities.argsort()[::-1]  # Ordenar por similitud descendente
top_indices = [i for i in top_indices if all_titles.iloc[i] != title_to_compare]  # Excluir el título base

# Seleccionar los 10 títulos más similares
top_10_indices = top_indices[:10]

# Obtener los 10 títulos más recomendados y sus similitudes
recommended_titles = all_titles.iloc[top_10_indices]
recommended_similarities = cosine_similarities[top_10_indices]

# Imprimir los 10 títulos más similares
print("\nTop 10 títulos más similares:")
print(recommended_titles)

# Crear una gráfica de barras con los 10 títulos más similares y sus puntuaciones
plt.figure(figsize=(10, 6))
plt.barh(recommended_titles[::-1], recommended_similarities[::-1], color='skyblue')
plt.xlabel('Similitud de Coseno')
plt.title('Top 10 Títulos más Similares')
plt.tight_layout()
plt.show()
