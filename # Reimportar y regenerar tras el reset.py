# Reimportar y regenerar tras el reset
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from sklearn.cluster import KMeans
import os

# Ruta de imagen
image_path = "C:/Users/salas/Downloads/img.jpg"
img = Image.open(image_path)

# Redimensionar a un máximo de 100 píxeles por lado para crochet
max_size = 100
img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

# Convertir a array y aplicar reducción de colores (10 tonos)
img_np = np.array(img)
h, w, _ = img_np.shape
img_flat = img_np.reshape(-1, 3)
kmeans = KMeans(n_clusters=10, random_state=42).fit(img_flat)
quantized_flat = kmeans.cluster_centers_[kmeans.labels_].astype('uint8')
quantized_img = quantized_flat.reshape((h, w, 3))

# Carpeta donde está el script
output_dir = os.path.dirname(os.path.abspath(__file__))

# Rutas absolutas para guardar las imágenes
image_output_path = os.path.join(output_dir, "patron_gatito10.png")
numbered_pattern_path = os.path.join(output_dir, "patron_gatito_numerado10.png")

# Crear patrón cuadriculado con leyenda
fig, ax = plt.subplots(figsize=(12, 12))
ax.set_title("Patrón de Crochet - Gato Durmiendo (10 colores)", fontsize=16)
ax.imshow(quantized_img)
ax.set_xticks(np.arange(w))
ax.set_yticks(np.arange(h))
ax.set_xticklabels(np.arange(1, w + 1))
ax.set_yticklabels(np.arange(1, h + 1))
ax.grid(which='both', color='gray', linestyle='-', linewidth=0.5)
ax.tick_params(axis='both', which='both', length=0)
plt.xticks(rotation=90)

# Crear leyenda de colores
unique_colors = np.unique(quantized_img.reshape(-1, 3), axis=0)
patches = [mpatches.Patch(color=np.array(c)/255, label=f'{i+1}') for i, c in enumerate(unique_colors)]
plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', title="Colores")

# Guardar como imagen
plt.savefig(image_output_path, bbox_inches='tight', dpi=300)
plt.close()

# Extraer dimensiones y colores únicos
h, w, _ = quantized_img.shape
unique_colors = np.unique(quantized_img.reshape(-1, 3), axis=0)

# Crear imagen con cuadricula y números
fig, ax = plt.subplots(figsize=(15, 10))
ax.imshow(quantized_img)
ax.set_xticks(np.arange(w))
ax.set_yticks(np.arange(h))
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.tick_params(axis='both', which='both', length=0)
ax.grid(True, which='both', color='lightgray', linestyle='-', linewidth=0.5)

# Crear mapa de colores únicos a números
color_map = {tuple(color): idx + 1 for idx, color in enumerate(unique_colors)}

# Poner el número en cada celda según su color
for y in range(h):
    for x in range(w):
        pixel_color = tuple(quantized_img[y, x])
        color_number = color_map[pixel_color]
        ax.text(x, y, str(color_number), ha='center', va='center', fontsize=4, color='black')

# Agregar leyenda de colores
patches = [mpatches.Patch(color=np.array(c)/255, label=f'{i+1}') for i, c in enumerate(unique_colors)]
plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', title="Colores")

# Guardar la imagen numerada
plt.savefig(numbered_pattern_path, bbox_inches='tight', dpi=300)
plt.close()