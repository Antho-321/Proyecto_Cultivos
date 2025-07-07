from PIL import Image
import numpy as np
import os
from keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split

# =========================================================================
# Utilidades
# =========================================================================
def ensure_channel_dim(arr: np.ndarray) -> np.ndarray:
    return arr[..., np.newaxis] if arr.ndim == 2 else arr


# =========================================================================
# 1. Recorte alrededor de las clases
# =========================================================================
def crop_around_classes(
    image: np.ndarray,
    mask: np.ndarray,
    classes_to_find: list[int] = [1, 2, 3, 4, 5],
    margin: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    mask = ensure_channel_dim(mask)
    is_class_present = np.isin(mask.squeeze(), classes_to_find)

    ys, xs = np.where(is_class_present)
    if ys.size == 0:
        return image, mask

    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()

    y0 = max(0, y_min - margin)
    y1 = min(mask.shape[0], y_max + margin + 1)
    x0 = max(0, x_min - margin)
    x1 = min(mask.shape[1], x_max + margin + 1)

    return image[y0:y1, x0:x1], mask[y0:y1, x0:x1, :]


# =========================================================================
# 2. Carga y división del dataset
# =========================================================================
def load_and_crop_dataset(
    image_directory: str,
    mask_directory: str,
    test_size: float = 0.2,
    random_state: int = 42,
):
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
# 3. Conteo de máscaras sin recorte (sin cambios)
# =========================================================================
def load_img_to_array(path: str) -> np.ndarray:
    return ensure_channel_dim(np.array(Image.open(path)))


# =========================================================================
# 4. Ejemplos únicos por clase sin recorte  (FUNCIÓN ACTUALIZADA)
# =========================================================================
def ejemplos_clase_sin_recorte(
    image_directory: str,
    mask_directory: str,
    classes_to_find: list[int] = [1, 2, 3, 4, 5],
    margin: int = 10,
) -> dict[int, str | None]:
    """
    Devuelve, para cada clase, la ruta de la primera imagen SIN recorte que
    contenga esa clase y que no haya sido usada ya para otra clase.
    """
    primera_ruta = {c: None for c in classes_to_find}
    usadas: set[str] = set()

    for img_name in sorted(os.listdir(image_directory)):
        if not img_name.lower().endswith((".jpg", ".png")):
            continue

        img_path  = os.path.join(image_directory, img_name)
        mask_path = os.path.join(
            mask_directory, os.path.splitext(img_name)[0] + "_mask.png"
        )
        if not os.path.exists(mask_path):
            continue

        img  = load_img_to_array(img_path)
        mask = load_img_to_array(mask_path)

        _, cropped_mask = crop_around_classes(img, mask, classes_to_find, margin)

        # solo consideramos las que NO se recortan
        if cropped_mask.shape != mask.shape:
            continue

        mask_2d = mask.squeeze()

        # intenta asignar esta imagen a la primera clase que la necesite
        for c in classes_to_find:
            if primera_ruta[c] is None and (mask_2d == c).any() and img_path not in usadas:
                primera_ruta[c] = img_path
                usadas.add(img_path)          # marca como usada
                break                         # no se asigna a más clases

        # terminamos si ya tenemos un ejemplo para cada clase
        if all(primera_ruta[c] is not None for c in classes_to_find):
            break

    # salida resumen
    print("Imágenes sin recorte por clase y su primer ejemplo único:")
    for c in classes_to_find:
        print(f"  Clase {c} | ejemplo: {primera_ruta[c]}")

    return primera_ruta

# Asegúrate de que estas funciones estén definidas o importadas en el mismo módulo:
# from tu_modulo import ensure_channel_dim, crop_around_classes

def contar_imagenes_sin_recorte(
    image_directory: str,
    mask_directory: str,
    classes_to_find: list[int] = [1, 2, 3, 4, 5],
    margin: int = 10,
) -> int:
    """
    Cuenta cuántas imágenes mantienen las mismas dimensiones tras aplicar crop_around_classes.
    """
    contador = 0

    for image_filename in sorted(os.listdir(image_directory)):
        if not image_filename.lower().endswith((".jpg", ".png")):
            continue

        img_path  = os.path.join(image_directory, image_filename)
        mask_path = os.path.join(
            mask_directory,
            os.path.splitext(image_filename)[0] + "_mask.png"
        )
        if not os.path.exists(mask_path):
            continue

        # cargamos la imagen y la máscara originales
        img_array  = img_to_array(load_img(img_path))
        mask_array = ensure_channel_dim(np.array(Image.open(mask_path)))

        # aplicamos el recorte
        img_crop, mask_crop = crop_around_classes(
            img_array,
            mask_array,
            classes_to_find,
            margin
        )

        # si no cambió el tamaño, incrementamos contador
        if img_crop.shape == img_array.shape and mask_crop.shape == mask_array.shape:
            contador += 1

    print(f"Número de imágenes sin recorte: {contador}")
    return contador


if __name__ == "__main__":
    IMAGE_DIR = "Balanced/train/images"
    MASK_DIR  = "Balanced/train/masks"
    contar_imagenes_sin_recorte(IMAGE_DIR, MASK_DIR)
