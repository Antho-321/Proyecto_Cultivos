# pixel_level_augment.py
import os, uuid, math, random, shutil
from pathlib import Path
from typing import Tuple, List

import numpy as np
from PIL import Image


def _safe_crop_coords(cy: int, cx: int,
                      h: int, w: int,
                      ch: int, cw: int) -> Tuple[int, int]:
    """
    Returns the top-left coordinate (y1, x1) of a crop centred at (cy, cx),
    clipped so the crop fits inside the original image.
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
    Saves the patch to disk using a random UUID + suffix.
    Returns the number of *foreground* pixels (all ≠0) in the mask patch
    — the caller may use this for bookkeeping if desired.
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
    Create extra image/mask pairs so that *approximately* `extra_pixels`
    additional pixels of `target_class` are written to `out_img_dir`
    / `out_mask_dir`.

    Parameters
    ----------
    image_dir, mask_dir
        Existing dataset folders (original size / resolution).
    out_img_dir, out_mask_dir
        Where to store augmented samples.  
        *Tip:* if these are the same as `image_dir`/`mask_dir`, the function
        will simply append new files into the original dataset.
    target_class : int
        Desired pixel value to oversample (0–5).
    extra_pixels : int
        How many *pixels* (not images) of `target_class` you want *added*.
        The function keeps sampling until the running sum ≥ this value.
    crop_size : (H, W)
        The patch size to harvest around each seed pixel.
    min_pixels_in_crop : int
        Reject crops that contain fewer than this number of pixels of
        `target_class` (avoids writing almost-empty patches).
    max_trials_per_image : int
        Safety stop to avoid infinite loops with extremely rare classes.
    """
    # ------------------------------------------------------------------ #
    # Sanity checks & folder prep
    # ------------------------------------------------------------------ #
    if target_class < 0 or target_class > 5:
        raise ValueError("`target_class` must be between 0 and 5")
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
    # Main loop
    # ------------------------------------------------------------------ #
    h_crop, w_crop = crop_size
    cumulative_pixels = 0

    rng = np.random.default_rng()

    while cumulative_pixels < extra_pixels:
        # Randomly choose an image; we will attempt to mine one patch from it.
        img_path: Path = rng.choice(img_files)
        base_name = img_path.stem
        msk_path: Path = mask_dir / f"{base_name}_mask.png"

        # Load arrays (Pillow → numpy, *no* resize!)
        img_arr = np.asarray(Image.open(img_path).convert("RGB"))
        msk_arr = np.asarray(Image.open(msk_path).convert("L"), dtype=np.uint8)

        ys, xs = np.where(msk_arr == target_class)
        if ys.size == 0:
            # No pixels of this class in this image; pick another image.
            continue

        h, w = msk_arr.shape[:2]

        # Try a few seeds in this image to get a crop with enough target pixels
        for _ in range(max_trials_per_image):
            idx = rng.integers(low=0, high=len(ys))
            cy, cx = int(ys[idx]), int(xs[idx])
            y1, x1 = _safe_crop_coords(cy, cx, h, w, h_crop, w_crop)
            img_patch = img_arr[y1:y1 + h_crop, x1:x1 + w_crop, :]
            msk_patch = msk_arr[y1:y1 + h_crop, x1:x1 + w_crop]

            n_target_pixels = int((msk_patch == target_class).sum())
            if n_target_pixels >= min_pixels_in_crop:
                _ = _save_patch(
                    img_patch,
                    msk_patch,
                    out_img_dir,
                    out_mask_dir,
                    suffix=f"cls{target_class}"
                )
                cumulative_pixels += n_target_pixels
                break  # go back to the while-loop / pick new image
        # end for
    # end while

    print(f"✔ Added ~{cumulative_pixels:,} pixels of class {target_class} "
          f"in {out_img_dir.relative_to(Path.cwd())}")


# ------------------------- Usage example ------------------------- #
if __name__ == "__main__":
    # Add ~50 000 extra pixels of class-4
    pixel_level_augment(
        image_dir="Balanced/train/images",
        mask_dir="Balanced/train/masks",
        out_img_dir="Balanced/train/images",   # append to same folder
        out_mask_dir="Balanced/train/masks",
        target_class=1,
        extra_pixels=22029888,
        crop_size=(96, 96)                     # match your CropAroundClass4
    )
