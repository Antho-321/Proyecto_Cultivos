from PIL import Image
import numpy as np
import os
from keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split


# =========================================================================
# Utilidades
# =========================================================================
def ensure_channel_dim(arr: np.ndarray) -> np.ndarray:
    """Añade un canal extra si la imagen es 2-D (H, W → H, W, 1)."""
    if arr.ndim == 2:
        return arr[..., np.newaxis]
    return arr


# =========================================================================
# 1. Recorte alrededor de las clases de interés
# =========================================================================
def crop_around_classes(
    image: np.ndarray,
    mask: np.ndarray,
    classes_to_find: list[int] = [1, 2, 3, 4, 5],
    margin: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    mask = ensure_channel_dim(mask)                     # (H, W, 1)
    is_class_present = np.isin(mask.squeeze(), classes_to_find)

    ys, xs = np.where(is_class_present)
    if ys.size == 0:                                    # no hay clases → sin recorte
        return image, mask

    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()

    y0 = max(0, y_min - margin)
    y1 = min(mask.shape[0], y_max + margin + 1)
    x0 = max(0, x_min - margin)
    x1 = min(mask.shape[1], x_max + margin + 1)

    cropped_image = image[y0:y1, x0:x1]
    cropped_mask  = mask[y0:y1, x0:x1, :]
    return cropped_image, cropped_mask


# =========================================================================
# 2. Carga y división del dataset
# =========================================================================
def load_and_crop_dataset(
    image_directory: str,
    mask_directory: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    images_processed, masks_processed = [], []

    for image_filename in sorted(os.listdir(image_directory)):
        if not image_filename.lower().endswith((".jpg", ".png")):
            continue

        img_path  = os.path.join(image_directory, image_filename)
        img_array = img_to_array(load_img(img_path))

        mask_filename = os.path.splitext(image_filename)[0] + "_mask.png"
        mask_path     = os.path.join(mask_directory, mask_filename)
        if not os.path.exists(mask_path):
            print(f"Advertencia: falta {mask_filename}")
            continue

        mask_array = ensure_channel_dim(np.array(Image.open(mask_path)))
        img_crop, mask_crop = crop_around_classes(img_array, mask_array)

        images_processed.append(img_crop)
        masks_processed.append(mask_crop)

    if not images_processed:
        raise RuntimeError("No se procesaron imágenes: revisa rutas y nombres.")

    X = np.array(images_processed, dtype=object)
    y = np.array(masks_processed,  dtype=object)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


# =========================================================================
# 3. Conteo de máscaras sin recorte
# =========================================================================
def load_img_to_array(path: str) -> np.ndarray:
    """Devuelve la imagen con forma (H, W, C)."""
    return ensure_channel_dim(np.array(Image.open(path)))


def count_uncropped(
    image_directory: str,
    mask_directory: str,
    classes_to_find: list[int] = [1, 2, 3, 4, 5],
    margin: int = 10,
) -> int:
    no_crop = 0

    for img_name in sorted(os.listdir(image_directory)):
        if not img_name.lower().endswith((".jpg", ".png")):
            continue

        img_path  = os.path.join(image_directory, img_name)
        mask_name = os.path.splitext(img_name)[0] + "_mask.png"
        mask_path = os.path.join(mask_directory, mask_name)
        if not os.path.exists(mask_path):
            continue

        img  = load_img_to_array(img_path)   # (H, W, C)
        mask = load_img_to_array(mask_path)  # (H, W, 1)

        _, cropped_mask = crop_around_classes(
            image=img,
            mask=mask,
            classes_to_find=classes_to_find,
            margin=margin,
        )

        if cropped_mask.shape == mask.shape:
            no_crop += 1

    return no_crop


def conteo_clases_en_sin_recorte(
    image_directory: str,
    mask_directory: str,
    classes_to_find: list[int] = [1, 2, 3, 4, 5],
    margin: int = 10,
) -> dict[int, int]:
    """
    Devuelve (y muestra por pantalla) cuántas imágenes sin recorte contienen
    al menos un píxel de cada clase indicada.

    Ej. salida ⇒ {1: 32, 2: 18, 3: 10, 4: 5, 5: 0}
    """
    # Inicializa contador por clase
    clases_contador = {c: 0 for c in classes_to_find}

    for img_name in sorted(os.listdir(image_directory)):
        if not img_name.lower().endswith((".jpg", ".png")):
            continue

        img_path  = os.path.join(image_directory, img_name)
        mask_name = os.path.splitext(img_name)[0] + "_mask.png"
        mask_path = os.path.join(mask_directory, mask_name)
        if not os.path.exists(mask_path):
            continue

        # Carga imágenes
        img  = load_img_to_array(img_path)   # (H, W, C)
        mask = load_img_to_array(mask_path)  # (H, W, 1)

        # Comprueba si la máscara se recortaría
        _, cropped_mask = crop_around_classes(
            image=img,
            mask=mask,
            classes_to_find=classes_to_find,
            margin=margin,
        )

        # Solo analiza las que NO se recortan
        if cropped_mask.shape == mask.shape:
            # Elimina la dimensión canal para buscar clases
            mask_2d = mask.squeeze()

            # Marca presencia de cada clase
            for c in classes_to_find:
                if (mask_2d == c).any():
                    clases_contador[c] += 1

    # Muestra resultados
    print("Imágenes sin recorte por clase:")
    for c in classes_to_find:
        print(f"  Clase {c}: {clases_contador[c]} imágenes")

    return clases_contador


def ejemplos_clase_sin_recorte(
    image_directory: str,
    mask_directory: str,
    classes_to_find: list[int] = [1, 2, 3, 4, 5],
    margin: int = 10,
) -> dict[int, str | None]:
    """
    Para cada clase indicada, imprime y devuelve la ruta de la primera imagen
    sin recorte que contenga al menos un píxel de dicha clase.

    Retorna un diccionario {clase: ruta_imagen_primera | None}.
    """
    # --- contadores y primeras rutas ---
    contador_clase  = {c: 0     for c in classes_to_find}
    primera_ruta    = {c: None  for c in classes_to_find}

    for img_name in sorted(os.listdir(image_directory)):
        if not img_name.lower().endswith((".jpg", ".png")):
            continue

        img_path  = os.path.join(image_directory, img_name)
        mask_name = os.path.splitext(img_name)[0] + "_mask.png"
        mask_path = os.path.join(mask_directory, mask_name)
        if not os.path.exists(mask_path):
            continue

        img  = load_img_to_array(img_path)   # (H, W, C)
        mask = load_img_to_array(mask_path)  # (H, W, 1)

        _, cropped_mask = crop_around_classes(
            image=img,
            mask=mask,
            classes_to_find=classes_to_find,
            margin=margin,
        )

        # Solo analizamos las que NO se recortan
        if cropped_mask.shape == mask.shape:
            mask_2d = mask.squeeze()

            for c in classes_to_find:
                if (mask_2d == c).any():
                    contador_clase[c] += 1
                    if primera_ruta[c] is None:             # guarda solo la primera
                        primera_ruta[c] = img_path

        # Terminamos antes si ya tenemos ejemplo para todas las clases
        if all(r is not None for r in primera_ruta.values()):
            break

    # --- salida por pantalla ---
    print("Imágenes sin recorte por clase y su primer ejemplo:")
    for c in classes_to_find:
        print(f"  Clase {c}: {contador_clase[c]} imágenes"
              f" | primer ejemplo: {primera_ruta[c]}")

    return primera_ruta


# -------------------------------------------------------------------------
# Ejemplo de uso
# -------------------------------------------------------------------------
if __name__ == "__main__":
    IMAGE_DIR = "Balanced/train/images"
    MASK_DIR  = "Balanced/train/masks"

    rutas_ejemplo = ejemplos_clase_sin_recorte(IMAGE_DIR, MASK_DIR)