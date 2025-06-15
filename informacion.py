import numpy as np
from typing import Dict, List, Union, Optional
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def load_dataset(image_directory, mask_directory):
    images = []
    masks = []

    for image_filename in os.listdir(image_directory):
        if image_filename.lower().endswith(('.jpg', '.png')):
            # Cargar imagen
            img_path = os.path.join(image_directory, image_filename)
            img = load_img(img_path)
            images.append(img_to_array(img))

            # Cargar máscara asociada
            name_wo_ext = os.path.splitext(image_filename)[0]
            mask_filename = f"{name_wo_ext}_mask.png"
            mask_path = os.path.join(mask_directory, mask_filename)

            if os.path.exists(mask_path):
                mask = load_img(mask_path, color_mode='grayscale')
                m = img_to_array(mask).astype(np.int32)
                masks.append(np.squeeze(m, axis=-1))  # (H, W)
            else:
                print(f"No se encontró la máscara para: {image_filename}")

    # Convertimos YA aquí en arrays y devolvemos
    images = np.array(images)     # shape (N, H, W, C)
    masks = np.array(masks)       # shape (N, H, W)  return images, masks
    return images, masks

def calculate_pixel_difference(
    masks: Union[np.ndarray, List[np.ndarray]],
    class_names: Optional[Dict[int, str]] = None
) -> Dict[int, int]:
    """
    Para cada clase, calcula la diferencia entre el número de píxeles
    de la clase 0 (fondo) y el número de píxeles de dicha clase.

    Parámetros
    ----------
    masks : np.ndarray or list
        Array de máscaras de forma (N, H, W) o una lista de máscaras (H, W).
    class_names : dict, opcional
        Diccionario para mapear IDs de clase a nombres legibles para la impresión.
        Ej: {0: 'Fondo', 1: 'Edificio'}

    Retorna
    -------
    dict
        Un diccionario donde las claves son los IDs de clase (mayores que 0)
        y los valores son el resultado de (píxeles_clase_0 - píxeles_clase_X).
    """
    # 1. Aplanar todas las máscaras para obtener una lista 1D de píxeles
    masks_arr = np.array(masks)
    flat_pixels = masks_arr.flatten()

    # 2. Contar la ocurrencia de cada valor de píxel (clase)
    unique_classes, counts = np.unique(flat_pixels, return_counts=True)
    pixel_counts = dict(zip(unique_classes, counts))

    # 3. Obtener el recuento de píxeles de la clase 0 (fondo)
    background_pixel_count = pixel_counts.get(0, 0)

    if background_pixel_count == 0:
        print("Advertencia: La clase 0 no se encontró en las máscaras. No se pueden calcular las diferencias.")
        return {}

    # 4. Calcular y mostrar las diferencias
    differences = {}
    print("\n--- Diferencia de Píxeles vs. Fondo (Clase 0) ---")
    print(f"Píxeles de Fondo (Clase 0) para referencia: {background_pixel_count}")

    for cls_id, count in pixel_counts.items():
        if cls_id == 0:
            continue  # Omitir la comparación de la clase 0 consigo misma

        # Calcular la diferencia
        diff = background_pixel_count - count
        differences[int(cls_id)] = diff

        # Imprimir el resultado de forma legible
        name_str = f" ({class_names[cls_id]})" if class_names and cls_id in class_names else ""
        print(f"Clase {int(cls_id)}{name_str}: {background_pixel_count} - {count} = {diff} píxeles de diferencia")

    return differences

# Carga original de tus datos
# images_list, masks_list = load_dataset("Balanced/train/images", "Balanced/train/masks")

# Ejemplo de cómo llamar a la nueva función
# Supongamos que masks_list ya está cargado y contiene tus máscaras

# Opcional: define nombres para tus clases para un informe más claro
class_names = {
    0: 'Fondo',
    1: 'Clase 1',
    2: 'Clase 2',
    3: 'Clase 3',
    4: 'Clase 4',
    5: 'Clase 5'
}

images_list, masks_list = load_dataset("Balanced/train/images", "Balanced/train/masks")
# Llama a la función con tu lista de máscaras
pixel_diff_dict = calculate_pixel_difference(masks_list, class_names=class_names)

# El diccionario 'pixel_diff_dict' contendrá los resultados calculados
print("\nDiccionario de diferencias devuelto:")
print(pixel_diff_dict)