import numpy as np
import os

# --- Tu función original para cargar los datos ---
# (Se asume que librerías como tensorflow o PIL están instaladas para load_img)
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def load_dataset(image_directory, mask_directory):
    images = []
    masks = []
    for image_filename in os.listdir(image_directory):
        if image_filename.endswith('.jpg') or image_filename.endswith('.png'):
            img_path = os.path.join(image_directory, image_filename)
            img = load_img(img_path)
            img_array = img_to_array(img)
            images.append(img_array)

            image_name_without_extension = image_filename.rsplit('.', 1)[0]
            mask_filename = image_name_without_extension + '_mask.png'
            mask_path = os.path.join(mask_directory, mask_filename)
            if os.path.exists(mask_path):
                mask = load_img(mask_path, color_mode='grayscale')
                mask_array = img_to_array(mask)
                masks.append(mask_array)
            else:
                print("No se encontró la máscara para la imagen:", image_filename)

    return np.array(images), np.array(masks)

# --- Función para calcular e imprimir el promedio de área por clase ---

def print_average_class_area(masks_array, class_pixels=[0, 1, 2, 3, 4, 5]):
    """
    Calcula e imprime el área promedio (en píxeles) que ocupa cada clase
    en un conjunto de máscaras de segmentación.

    Args:
        masks_array (np.array): Un array de NumPy que contiene todas las máscaras.
                                La forma esperada es (num_mascaras, alto, ancho, 1).
        class_pixels (list): Una lista de los valores de píxel que definen cada clase.
    """
    # Asegurarse de que el array no esté vacío
    if masks_array.size == 0:
        print("El array de máscaras está vacío.")
        return

    num_masks = masks_array.shape[0]
    print(f"📊 Analizando {num_masks} máscaras...")
    print("-" * 40)

    for class_value in class_pixels:
        # Contar el número total de píxeles para la clase actual en TODAS las máscaras
        total_class_pixels = np.sum(masks_array == class_value)

        # Calcular el promedio dividiendo por el número de máscaras
        average_area = total_class_pixels / num_masks

        print(f"Clase {class_value}: Área promedio = {average_area:.2f} píxeles")

    print("-" * 40)


# --- Ejemplo de uso ---

# 2. Llamar a la función con los datos de ejemplo
# En tu caso, la llamarías así:
images, masks = load_dataset('Balanced/train/images', 'Balanced/train/masks')
# print_average_class_area(masks)

print("--- Ejecutando con datos de ejemplo ---")
print_average_class_area(masks)