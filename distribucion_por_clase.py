# carga original

# def load_dataset(image_directory, mask_directory):
#     images = []
#     masks = []
#     for image_filename in os.listdir(image_directory):
#         if image_filename.endswith('.jpg') or image_filename.endswith('.png'):
#             # Formar la ruta del archivo de la imagen
#             img_path = os.path.join(image_directory, image_filename)
#             img = load_img(img_path)
#             img_array = img_to_array(img)
#             images.append(img_array)

#             # Formar el nombre del archivo de la máscara
#             image_name_without_extension = image_filename.rsplit('.', 1)[0]  # Eliminar la última extensión
#             mask_filename = image_name_without_extension + '_mask.png'  # Agregar sufijo "_mask.png"
#             mask_path = os.path.join(mask_directory, mask_filename)
#             if os.path.exists(mask_path):  # Verificar si el archivo de máscara existe
#                 mask = load_img(mask_path, color_mode='grayscale')
#                 mask_array = img_to_array(mask)
#                 masks.append(mask_array)
#             else:
#                 print("No se encontró la máscara para la imagen:", image_filename)

#     return np.array(images), np.array(masks)

import os
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array

def load_dataset(image_directory, mask_directory):
    images = []
    masks = []

    for image_filename in os.listdir(image_directory):
        if image_filename.lower().endswith(('.jpg', '.png')):
            # Cargar imagen
            img_path = os.path.join(image_directory, image_filename)
            img = load_img(img_path)
            img_array = img_to_array(img)
            images.append(img_array)

            # Determinar nombre de la máscara
            name_wo_ext = os.path.splitext(image_filename)[0]
            mask_filename = f"{name_wo_ext}_mask.png"
            mask_path = os.path.join(mask_directory, mask_filename)

            if os.path.exists(mask_path):
                # Cargar máscara en modo grayscale
                mask = load_img(mask_path, color_mode='grayscale')
                mask_array = img_to_array(mask)               # shape: (H, W, 1)
                mask_array = mask_array.astype(np.int32)      # asegurar dtype entero
                mask_array = np.squeeze(mask_array, axis=-1)  # shape: (H, W)
                masks.append(mask_array)
            else:
                print(f"No se encontró la máscara para la imagen: {image_filename}")

    # Convertir a arrays
    images = np.array(images)
    masks = np.array(masks)  # shape: (N, H, W)

    # --- Cálculo de porcentaje de píxeles por clase ---
    flat = masks.flatten()
    unique_vals, counts = np.unique(flat, return_counts=True)
    total_pixels = flat.size

    print("\n--- Porcentaje de píxeles por clase ---")
    for cls, cnt in zip(unique_vals, counts):
        pct = cnt / total_pixels * 100
        print(f"Clase {cls}: {pct:.2f}% ({cnt} píxeles)")

    return images, masks

# Ejemplo de uso:
images, masks = load_dataset("Balanced/train/images", "Balanced/train/masks")
