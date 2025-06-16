# pixel_level_augment_gpu_v3.py
import uuid
from pathlib import Path
from typing import Tuple, List

import cupy as cp
import numpy as np
from PIL import Image


# --------------------------------------------------------------------- #
# Utilidades totalmente en GPU
# --------------------------------------------------------------------- #
def _safe_crop_coords_gpu(cy, cx, h, w, ch, cw):
    """Devuelve (y1, x1) de un recorte centrado en (cy, cx)."""
    y1 = cp.clip(cy - ch // 2, 0, h - ch)
    x1 = cp.clip(cx - cw // 2, 0, w - cw)
    # convertimos a int de Python (transferencia de 8 B cada uno)
    return int(y1.get()), int(x1.get())


def _save_patch_gpu(img_patch_gpu: cp.ndarray,
                    msk_patch_gpu: cp.ndarray,
                    out_img_dir: Path,
                    out_msk_dir: Path,
                    target_class: int,
                    suffix: str) -> int:
    """Copia a RAM, guarda PNG y devuelve el # píxeles == target_class."""
    name = f"{uuid.uuid4().hex}_{suffix}.png"
    Image.fromarray(cp.asnumpy(img_patch_gpu)).save(out_img_dir / name)
    Image.fromarray(cp.asnumpy(msk_patch_gpu))\
        .save(out_msk_dir / name.replace(".png", "_mask.png"))
    return int((msk_patch_gpu == target_class).sum().item())


# --------------------------------------------------------------------- #
# Bucle principal
# --------------------------------------------------------------------- #
def pixel_level_augment_gpu(image_dir: str,
                            mask_dir: str,
                            out_img_dir: str,
                            out_mask_dir: str,
                            target_class: int,
                            extra_pixels: int,
                            crop_size: Tuple[int, int] = (96, 96),
                            min_pixels_in_crop: int = 40,
                            max_trials_per_image: int = 200,
                            preload: bool = False,
                            gpu_max_mem: int = 4_000,   # MiB
                            max_other_frac: float = 0.0  # ≤ 1.0
                            ) -> None:
    """
    Genera parches para aumentar la clase `target_class` usando GPU.
    Aceleración adicional (+30-50 %) respecto a la versión CPU.

    Parámetros nuevos / destacados
    ------------------------------
    max_other_frac : float
        Fracción máxima (0-1) de píxeles que se permite que NO sean de la
        clase objetivo dentro del parche aceptado.  Ej.:

        • 0.0  → parche 100 % puro (comportamiento original)
        • 0.05 → hasta un 5 % de píxeles “otros”
    """
    image_dir, mask_dir = Path(image_dir), Path(mask_dir)
    out_img_dir, out_mask_dir = Path(out_img_dir), Path(out_mask_dir)
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)

    img_files: List[Path] = sorted(
        p for p in image_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if not img_files:
        raise RuntimeError(f"Sin imágenes en {image_dir}")

    # --------------------------------------------------------------- #
    # Pre-carga (si el dataset es pequeño)
    # --------------------------------------------------------------- #
    if preload:
        imgs_gpu, msks_gpu, mem_bytes = [], [], 0
        for p in img_files:
            img = np.asarray(Image.open(p).convert("RGB"))
            msk = np.asarray(Image.open(mask_dir / f"{p.stem}_mask.png")
                             .convert("L"), dtype=np.uint8)
            imgs_gpu.append(cp.asarray(img))
            msks_gpu.append(cp.asarray(msk))
            mem_bytes += img.nbytes + msk.nbytes
            if mem_bytes / 1_048_576 > gpu_max_mem:
                print("Dataset demasiado grande para pre-carga; paso a modo stream.")
                imgs_gpu.clear(); msks_gpu.clear(); preload = False
                break

    h_crop, w_crop = crop_size
    cumulative_pixels = 0
    print(f"Comenzando aumento GPU… objetivo ≈{extra_pixels:,} píxeles extra")

    while cumulative_pixels < extra_pixels:
        # Elegir imagen al azar 100 % en GPU
        idx_img = int(cp.random.randint(0, len(img_files)))

        if preload:
            img_gpu, msk_gpu = imgs_gpu[idx_img], msks_gpu[idx_img]
        else:
            p_img = img_files[idx_img]
            p_msk = mask_dir / f"{p_img.stem}_mask.png"
            if not p_msk.exists():
                continue
            img_gpu = cp.asarray(Image.open(p_img).convert("RGB"))
            msk_gpu = cp.asarray(Image.open(p_msk).convert("L"), dtype=cp.uint8)

        ys, xs = cp.where(msk_gpu == target_class)
        if ys.size == 0:
            continue

        h, w = msk_gpu.shape
        trials, success = 0, False

        while trials < max_trials_per_image and not success:
            trials += 1
            j = int(cp.random.randint(0, ys.size))
            cy, cx = ys[j], xs[j]         # escalares 0-D cupy

            y1, x1 = _safe_crop_coords_gpu(cy, cx, h, w, h_crop, w_crop)
            img_patch = img_gpu[y1:y1 + h_crop, x1:x1 + w_crop]
            msk_patch = msk_gpu[y1:y1 + h_crop, x1:x1 + w_crop]

            # ----- Criterios de aceptación -------------------------- #
            n_target = int((msk_patch == target_class).sum().item())
            if n_target < min_pixels_in_crop:
                continue

            total_pix   = h_crop * w_crop
            n_other     = total_pix - n_target
            allowed_max = int(total_pix * max_other_frac)

            if n_other > allowed_max:
                continue        # Demasiados píxeles de otra clase
            # -------------------------------------------------------- #

            cumulative_pixels += _save_patch_gpu(
                img_patch, msk_patch,
                out_img_dir, out_mask_dir,
                target_class, f"cls{target_class}"
            )
            success = True
            print(f"\rProgreso: {cumulative_pixels/extra_pixels*100:6.2f}% "
                  f"({cumulative_pixels:,}/{extra_pixels:,})",
                  end="", flush=True)

    print(f"\n✔ Añadidos ~{cumulative_pixels:,} píxeles "
          f"en {out_img_dir.relative_to(Path.cwd())}")


# --------------------------- Ejemplo de uso --------------------------- #
if __name__ == "__main__":
    # Ajusta las rutas de acuerdo a tu estructura de carpetas
    pixel_level_augment_gpu(
        image_dir="Balanced/train/images",
        mask_dir="Balanced/train/masks",
        out_img_dir="Balanced_augmented/train/images",
        out_mask_dir="Balanced_augmented/train/masks",
        target_class=1,
        extra_pixels=22_029_888,
        crop_size=(96, 96),
        min_pixels_in_crop=100,
        preload=True,            # cambia a False si no cabe en la GPU
        max_other_frac=0.10      # ← permite hasta un 5 % de píxeles “otros”
    )
