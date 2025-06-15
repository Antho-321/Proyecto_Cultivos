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
from typing import Union, List, Dict    # <-- IMPORTAR AQUÍ

def load_dataset(image_directory, mask_directory):
    images = []
    masks  = []

    for image_filename in os.listdir(image_directory):
        if image_filename.lower().endswith(('.jpg', '.png')):
            # Cargar imagen
            img_path = os.path.join(image_directory, image_filename)
            img = load_img(img_path)
            images.append(img_to_array(img))

            # Cargar máscara asociada
            name_wo_ext   = os.path.splitext(image_filename)[0]
            mask_filename = f"{name_wo_ext}_mask.png"
            mask_path     = os.path.join(mask_directory, mask_filename)

            if os.path.exists(mask_path):
                mask = load_img(mask_path, color_mode='grayscale')
                m = img_to_array(mask).astype(np.int32)
                masks.append(np.squeeze(m, axis=-1))  # (H, W)
            else:
                print(f"No se encontró la máscara para: {image_filename}")

    # Convertimos YA aquí en arrays y devolvemos
    images = np.array(images)     # shape (N, H, W, C)
    masks  = np.array(masks)      # shape (N, H, W)
    return images, masks

def print_pixel_percentage(images_list, masks_list):
    """
    Convierte las listas de imágenes y máscaras en arrays de NumPy,
    calcula e imprime el porcentaje de píxeles por clase en las máscaras,
    y devuelve los arrays listos para usar.

    Parámetros
    ----------
    images_list : list of np.ndarray
        Lista de imágenes (H, W, C).
    masks_list : list of np.ndarray
        Lista de máscaras (H, W) con valores de clase enteros.

    Retorna
    -------
    images : np.ndarray, shape (N, H, W, C)
    masks : np.ndarray, shape (N, H, W)
    """
    # 1) Convertir a arrays
    images = np.array(images_list)
    masks  = np.array(masks_list)  # shape: (N, H, W)

    # 2) Cálculo de porcentaje de píxeles por clase
    flat = masks.flatten()
    unique_vals, counts = np.unique(flat, return_counts=True)
    total_pixels = flat.size

    print("\n--- Porcentaje de píxeles por clase ---")
    for cls, cnt in zip(unique_vals, counts):
        pct = cnt / total_pixels * 100
        print(f"Clase {cls}: {pct:.2f}% ({cnt} píxeles)")

    return images, masks

def calculate_class_weights(masks: Union[np.ndarray, List[np.ndarray]], verbose: bool = True) -> Dict[int, float]:
    """
    Calcula los pesos de clase basados en la frecuencia inversa de los píxeles.

    Args:
        masks (np.ndarray or list of np.ndarray): Máscara o lista de máscaras de segmentación 
                                                  con forma (H, W) o (N, H, W).
                                                  Los valores son los índices de clase.
        verbose (bool): Si es True, imprime un resumen de los pesos calculados.

    Returns:
        dict: Un diccionario donde las claves son los ID de clase y los valores son sus pesos.
              Ej: {0: 1.2, 1: 0.8, 2: 3.5}
    """
    # Si es lista, convertimos a un único array de numpy
    masks_arr = np.array(masks)  # Forma resultante: (N, H, W) o (H, W)
    
    # Aplanamos el array para tener una lista 1D de todos los píxeles de clase
    flat_pixels = masks_arr.flatten()
    
    # Contamos la aparición de cada clase
    classes, counts = np.unique(flat_pixels, return_counts=True)
    
    # Determinamos el número total de clases (basado en el ID de clase más alto)
    num_classes = int(flat_pixels.max()) + 1
    total_pixels = flat_pixels.size

    if verbose:
        print("\n--- Pesos por clase (calculados por frecuencia inversa) ---")

    weights = {}
    # Calculamos el peso para las clases que sí aparecen en las máscaras
    for cls, count in zip(classes, counts):
        # Fórmula de peso: Inverso de la frecuencia de la clase
        weight = total_pixels / (num_classes * count)
        weights[int(cls)] = weight
        if verbose:
            percentage = (count / total_pixels) * 100
            print(f"Clase {cls}: peso = {weight:.4f} ({count} píxeles, {percentage:.2f}%)")

    # Asignamos peso 0 a las clases que no aparecen en ninguna máscara
    present_classes = set(classes)
    all_possible_classes = set(range(num_classes))
    missing_classes = all_possible_classes - present_classes
    
    for cls in missing_classes:
        weights[cls] = 0.0
        if verbose:
            print(f"Clase {cls}: peso = 0.0000 (clase ausente)")

    return weights

# # Cargo las listas
# images_list, masks_list = load_dataset("Balanced/train/images", "Balanced/train/masks")
# weights_dict_silent = calculate_class_weights(masks_list, verbose=True)