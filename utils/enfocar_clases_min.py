import numpy as np

def crop_around_classes(
    image: np.ndarray,
    mask: np.ndarray,
    classes_to_find: list[int] = [1, 2, 3, 4, 5],
    margin: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Recorta un rectángulo alrededor de todos los píxeles que pertenecen a las
    clases especificadas en la máscara.

    Args:
        image (H×W×C np.ndarray): Imagen original.
        mask (H×W×1 np.ndarray): Máscara con valores de clase.
        classes_to_find (list[int]): Lista de IDs de clase a encontrar.
        margin (int): Píxeles de margen extra alrededor del bounding box.

    Returns:
        cropped_image (h×w×C np.ndarray)
        cropped_mask (h×w×1 np.ndarray)

    Si no se encuentra ningún píxel de las clases, devuelve la imagen y
    máscara originales.
    """
    # Crea una máscara booleana donde True son los píxeles de las clases de interés
    is_class_present = np.isin(mask, classes_to_find)

    # Encuentra las coordenadas de estos píxeles
    # El [:,:,0] es para asegurar que la máscara sea 2D si tiene una dimensión de canal
    ys, xs = np.where(is_class_present[:,:,0])

    if ys.size == 0:
        # No hay píxeles de las clases de interés: devolvemos original
        return image, mask

    # Calcula los límites del bounding box
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()

    # Aplica el margen sin salirse de los límites de la imagen
    y0 = max(0, y_min - margin)
    y1 = min(mask.shape[0], y_max + margin + 1)
    x0 = max(0, x_min - margin)
    x1 = min(mask.shape[1], x_max + margin + 1)

    # Recorta la imagen y la máscara
    cropped_image = image[y0:y1, x0:x1]
    cropped_mask = mask[y0:y1, x0:x1]

    return cropped_image, cropped_mask

import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split

def load_and_crop_dataset(
    image_directory: str,
    mask_directory: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> tuple:
    """
    Carga imágenes y máscaras, recorta cada par alrededor de las clases [1-5],
    y divide el dataset en conjuntos de entrenamiento y validación.

    Args:
        image_directory (str): Ruta al directorio de imágenes.
        mask_directory (str): Ruta al directorio de máscaras.
        test_size (float): Proporción del dataset a reservar para validación.
        random_state (int): Semilla para la aleatoriedad en la división.

    Returns:
        Una tupla con los cuatro conjuntos de datos:
        (X_train, X_val, y_train, y_val)
    """
    images_processed = []
    masks_processed = []

    print("Iniciando la carga y recorte de imágenes...")
    for image_filename in sorted(os.listdir(image_directory)):
        if image_filename.lower().endswith(('.jpg', '.png')):
            # --- Carga la imagen ---
            img_path = os.path.join(image_directory, image_filename)
            img = load_img(img_path)
            img_array = img_to_array(img)

            # --- Carga la máscara correspondiente ---
            image_name_no_ext = os.path.splitext(image_filename)[0]
            mask_filename = f"{image_name_no_ext}_mask.png" # Asume el formato de tu función original
            mask_path = os.path.join(mask_directory, mask_filename)

            if os.path.exists(mask_path):
                mask = load_img(mask_path, color_mode='grayscale')
                mask_array = img_to_array(mask)

                # --- Recorta la imagen y la máscara ---
                img_cropped, mask_cropped = crop_around_classes(img_array, mask_array)

                images_processed.append(img_cropped)
                masks_processed.append(mask_cropped)
            else:
                print(f"Advertencia: No se encontró la máscara para la imagen: {image_filename}")

    if not images_processed:
        print("No se procesaron imágenes. Verifica las rutas y los nombres de archivo.")
        return None

    print(f"\nSe procesaron {len(images_processed)} pares de imagen/máscara.")
    print("Dividiendo en conjuntos de entrenamiento y validación...")

    # ADVERTENCIA: Los arrays tienen tamaños diferentes. Se guardan como arrays de objetos.
    # Esto es adecuado para el almacenamiento pero puede requerir pre-procesamiento
    # adicional (ej. padding/resize) antes de alimentar un modelo de red neuronal.
    images_np = np.array(images_processed, dtype=object)
    masks_np = np.array(masks_processed, dtype=object)

    # --- Divide los datos ---
    X_train, X_val, y_train, y_val = train_test_split(
        images_np, masks_np, test_size=test_size, random_state=random_state
    )

    print("División completada.")
    print(f"Tamaño del conjunto de entrenamiento: {len(X_train)} imágenes")
    print(f"Tamaño del conjunto de validación: {len(X_val)} imágenes")

    return X_train, X_val, y_train, y_val

# # --- Ejemplo de Uso ---
# if __name__ == '__main__':

#     IMAGE_DIR = "Balanced/train/images"
#     MASK_DIR = "Balanced/train/masks"

#     # Llama a la función para obtener los datasets divididos
#     X_train, X_val, y_train, y_val = load_and_crop_dataset(IMAGE_DIR, MASK_DIR)

#     # Ahora puedes usar X_train, X_val, y_train, y_val para entrenar tu modelo
#     if X_train is not None:
#         print(f"\nEjemplo de dimensiones de la primera imagen de entrenamiento: {X_train[0].shape}")
#         print(f"Ejemplo de dimensiones de la primera máscara de entrenamiento: {y_train[0].shape}")