# pixel_level_augment_gpu.py
import os, uuid, math, random, shutil
from pathlib import Path
from typing import Tuple, List

# --- Cambios para la GPU ---
# Se importa CuPy. Debe instalarse con el kit de herramientas CUDA correcto.
# Ejemplo: pip install cupy-cuda11x
import cupy as cp
# -------------------------

import numpy as np # NumPy sigue siendo útil para operaciones en la CPU
from PIL import Image


def _safe_crop_coords(cy: int, cx: int,
                      h: int, w: int,
                      ch: int, cw: int) -> Tuple[int, int]:
    """
    Devuelve la coordenada superior izquierda (y1, x1) de un recorte centrado en (cy, cx),
    ajustada para que el recorte quepa dentro de la imagen original.
    Esta función sigue usando tipos de Python nativos y NumPy, ya que la lógica es simple
    y no requiere aceleración en la GPU.
    """
    y1 = int(np.clip(cy - ch // 2, 0, h - ch))
    x1 = int(np.clip(cx - cw // 2, 0, w - cw))
    return y1, x1


def _save_patch(img_arr: np.ndarray,
                msk_arr: np.ndarray,
                out_img_dir: Path,
                out_msk_dir: Path,
                suffix: str) -> int:
    """
    Guarda el parche en el disco usando un UUID aleatorio + sufijo.
    Los arrays de entrada DEBEN ser arrays de NumPy (en la CPU).
    Retorna el número de píxeles de primer plano (todos ≠0) en el parche de la máscara.
    """
    name = f"{uuid.uuid4().hex}_{suffix}.png"
    Image.fromarray(img_arr).save(out_img_dir / name)
    Image.fromarray(msk_arr).save(out_msk_dir / (name.replace('.png', '_mask.png')))
    return int((msk_arr != 0).sum())


def pixel_level_augment(image_dir: str,
                        mask_dir: str,
                        out_img_dir: str,
                        out_mask_dir: str,
                        target_class: int,
                        extra_pixels: int,
                        crop_size: Tuple[int, int] = (96, 96),
                        min_pixels_in_crop: int = 40,
                        max_trials_per_image: int = 200
                        ) -> None:
    """
    Crea pares adicionales de imagen/máscara para que *aproximadamente* `extra_pixels`
    píxeles adicionales de `target_class` se escriban en `out_img_dir` / `out_mask_dir`.
    Esta versión está optimizada para ejecutarse en una GPU NVIDIA usando CuPy.
    """
    # ------------------------------------------------------------------ #
    # Comprobaciones de seguridad y preparación de carpetas
    # ------------------------------------------------------------------ #
    if not (0 <= target_class <= 255):
        raise ValueError("`target_class` must be a valid pixel value (0-255)")
    image_dir, mask_dir = Path(image_dir), Path(mask_dir)
    out_img_dir, out_mask_dir = Path(out_img_dir), Path(out_mask_dir)
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)

    img_files: List[Path] = sorted(
        p for p in image_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if not img_files:
        raise RuntimeError(f"No images found in {image_dir}")

    # ------------------------------------------------------------------ #
    # Bucle principal
    # ------------------------------------------------------------------ #
    h_crop, w_crop = crop_size
    cumulative_pixels = 0
    
    # El generador de números aleatorios de NumPy sigue siendo adecuado para la CPU.
    rng = np.random.default_rng()

    print(f"Iniciando aumento en la GPU para la clase {target_class}...")
    while cumulative_pixels < extra_pixels:
        # Elige una imagen al azar
        img_path: Path = rng.choice(img_files)
        base_name = img_path.stem
        msk_path: Path = mask_dir / f"{base_name}_mask.png"
        
        if not msk_path.exists():
            print(f"Advertencia: No se encontró la máscara {msk_path}, saltando imagen.")
            continue

        # Carga los arrays en la CPU primero usando Pillow/NumPy
        img_arr_cpu = np.asarray(Image.open(img_path).convert("RGB"))
        msk_arr_cpu = np.asarray(Image.open(msk_path).convert("L"), dtype=np.uint8)

        # --- MODIFICACIÓN GPU: Mueve los arrays a la GPU ---
        img_arr_gpu = cp.asarray(img_arr_cpu)
        msk_arr_gpu = cp.asarray(msk_arr_cpu)
        
        # --- MODIFICACIÓN GPU: Realiza la búsqueda en la GPU ---
        ys_gpu, xs_gpu = cp.where(msk_arr_gpu == target_class)
        
        # .size funciona tanto en NumPy como en CuPy
        if ys_gpu.size == 0:
            # No hay píxeles de esta clase en la imagen; elige otra.
            continue

        h, w = msk_arr_gpu.shape[:2]

        # Intenta varias semillas en esta imagen para obtener un buen recorte
        for _ in range(max_trials_per_image):
            # Elige un índice aleatorio de un píxel objetivo
            idx = rng.integers(low=0, high=len(ys_gpu))

            # --- MODIFICACIÓN GPU: Obtiene las coordenadas del píxel semilla ---
            # .get() transfiere un único valor de la GPU a la CPU
            cy, cx = int(ys_gpu[idx].get()), int(xs_gpu[idx].get())

            y1, x1 = _safe_crop_coords(cy, cx, h, w, h_crop, w_crop)
            
            # --- MODIFICACIÓN GPU: Recorta el parche directamente en la GPU ---
            img_patch_gpu = img_arr_gpu[y1:y1 + h_crop, x1:x1 + w_crop, :]
            msk_patch_gpu = msk_arr_gpu[y1:y1 + h_crop, x1:x1 + w_crop]

            # --- MODIFICACIÓN GPU: Cuenta los píxeles en la GPU ---
            # .sum() en CuPy devuelve un array de 0 dimensiones.
            n_target_pixels = int((msk_patch_gpu == target_class).sum())
            
            if n_target_pixels >= min_pixels_in_crop:
                # --- MODIFICACIÓN GPU: Mueve el parche de vuelta a la CPU para guardarlo ---
                img_patch_cpu = cp.asnumpy(img_patch_gpu)
                msk_patch_cpu = cp.asnumpy(msk_patch_gpu)

                _ = _save_patch(
                    img_patch_cpu,
                    msk_patch_cpu,
                    out_img_dir,
                    out_mask_dir,
                    suffix=f"cls{target_class}"
                )
                cumulative_pixels += n_target_pixels
                
                # Opcional: Imprime el progreso
                print(f"\rProgreso: {cumulative_pixels / extra_pixels * 100:.2f}% ({cumulative_pixels:,}/{extra_pixels:,} píxeles añadidos)...", end="")
                
                break  # Vuelve al bucle while para elegir una nueva imagen
        # fin del bucle for
    # fin del bucle while

    print(f"\n✔ Se añadieron ~{cumulative_pixels:,} píxeles de la clase {target_class} "
          f"en {out_img_dir.relative_to(Path.cwd())}")


# ------------------------- Ejemplo de uso ------------------------- #
if __name__ == "__main__":
    # Asegúrate de que las rutas y los parámetros son correctos para tu caso de uso.
    # Este es solo un ejemplo.
    try:
        pixel_level_augment(
            image_dir="Balanced/train/images",
            mask_dir="Balanced/train/masks",
            out_img_dir="Balanced_augmented/train/images", # Es mejor escribir en una nueva carpeta
            out_mask_dir="Balanced_augmented/train/masks",
            target_class=1,
            extra_pixels=50000,
            crop_size=(256, 256),
            min_pixels_in_crop=100
        )
    except FileNotFoundError:
        print("Error: Los directorios de entrada no existen. Por favor, ajusta las rutas en el bloque `if __name__ == '__main__'`.")
    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")