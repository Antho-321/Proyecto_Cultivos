# train.py  — versión refactorizada con las recomendaciones
# ================================================================
#  • Recorte guiado antes de las aug             (crop_around_classes)
#  • Sampler estratificado con WeightedRandomSampler
#  • Early‑Stopping (paciencia 10)
#  • ReduceLROnPlateau + TF32 + torch.compile «max‑autotune»
#  • Histórico de mIoU y plot automático al final
# ================================================================

from __future__ import annotations
import os
import numpy as np
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from model import CloudDeepLabV3Plus
from utils import (
    imprimir_distribucion_clases_post_augmentation,
    crop_around_classes,
    save_performance_plot,
)
from config import Config

# ---------------------------------------------
# Aceleración hardware
# ---------------------------------------------
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

# ==============================================================================
# 1.   DATASET
# ==============================================================================
class CloudDataset(torch.utils.data.Dataset):
    """Dataset que recorta alrededor de las clases raras antes de la aug."""

    _IMG_EXTENSIONS: tuple[str, ...] = (".jpg", ".png")

    def __init__(
        self,
        image_dir: str | Path,
        mask_dir: str | Path,
        transform: A.Compose | None = None,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform

        self.images: List[Path] = [
            p for p in self.image_dir.iterdir() if p.suffix.lower() in self._IMG_EXTENSIONS
        ]

        # Pre‑calcular pesos por imagen (para Balanced Sampler)
        self.img_weights: list[float] = []
        for img_path in self.images:
            mask_path = self._mask_path_from_image_name(img_path)
            mask = np.array(Image.open(mask_path))
            # Si contiene alguna clase minoritaria (3,4) => peso alto
            if np.isin(mask, [3, 4]).any():
                self.img_weights.append(3.0)
            else:
                self.img_weights.append(1.0)

    def _mask_path_from_image_name(self, img_path: Path) -> Path:
        return self.mask_dir / f"{img_path.stem}_mask.png"

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img_path = self.images[idx]
        mask_path = self._mask_path_from_image_name(img_path)
        if not mask_path.exists():
            raise FileNotFoundError(f"Máscara no encontrada: {mask_path}")

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))

        # --- recorte alrededor de las clases minoritarias antes de la aug ---
        mask_3d = np.expand_dims(mask, -1)
        image_crop, mask_crop_3d = crop_around_classes(image, mask_3d)
        mask_crop = mask_crop_3d.squeeze()
        # -------------------------------------------------------------------

        if self.transform:
            aug = self.transform(image=image_crop, mask=mask_crop)
            image, mask = aug["image"], aug["mask"]

        return image, mask

# ==============================================================================
# 2.   FUNCIONES DE ENTRENAMIENTO / VALIDACIÓN
# ==============================================================================

def train_fn(loader, model, optimizer, loss_fn, scaler, num_classes: int = 6):
    model.train()
    loop = tqdm(loader, leave=True)
    device = Config.DEVICE

    tp = torch.zeros(num_classes, device=device)
    fp = torch.zeros(num_classes, device=device)
    fn = torch.zeros(num_classes, device=device)

    for imgs, targets in loop:
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).long()

        with autocast(device_type=device, dtype=torch.float16):
            logits = model(imgs)
            logits = logits[0] if isinstance(logits, tuple) else logits
            loss = loss_fn(logits, targets)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        preds = logits.argmax(dim=1)
        for c in range(num_classes):
            tp[c] += ((preds == c) & (targets == c)).sum()
            fp[c] += ((preds == c) & (targets != c)).sum()
            fn[c] += ((preds != c) & (targets == c)).sum()

        loop.set_postfix(loss=f"{loss.item():.3f}")

    eps = 1e-6
    iou = tp / (tp + fp + fn + eps)
    dice = (2 * tp) / (2 * tp + fp + fn + eps)
    miou = torch.nanmean(iou)

    print("\nÉpoca de entrenamiento finalizada:")
    print("  - Dice por clase:", dice.cpu().numpy())
    print("  - IoU  por clase:", iou.cpu().numpy())
    print(f"  - mIoU: {miou:.4f}")

    return miou


def check_metrics(loader, model, n_classes: int = 6, device: str = "cuda"):
    eps = 1e-8
    conf = torch.zeros((n_classes, n_classes), dtype=torch.float64, device=device)
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True).long()
            logits = model(x)
            logits = logits[0] if isinstance(logits, tuple) else logits
            preds = logits.argmax(dim=1)
            idx = (preds * n_classes + y).view(-1).float()
            conf += torch.histc(idx, bins=n_classes * n_classes, min=0, max=n_classes ** 2 - 1).view(
                n_classes, n_classes
            )

    inter = torch.diag(conf)
    pred_sum = conf.sum(1)
    true_sum = conf.sum(0)
    union = pred_sum + true_sum - inter

    iou = (inter + eps) / (union + eps)
    dice = (2 * inter + eps) / (pred_sum + true_sum + eps)

    miou, mdice = iou.mean(), dice.mean()

    print("IoU por clase :", iou.cpu().numpy())
    print("Dice por clase:", dice.cpu().numpy())
    print(f"mIoU macro = {miou:.4f} | Dice macro = {mdice:.4f}")

    model.train()
    return miou, mdice


def validate_fn(loader, model, loss_fn, device=Config.DEVICE):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True).long()
            logits = model(x)
            logits = logits[0] if isinstance(logits, tuple) else logits
            total += loss_fn(logits, y).item() * x.size(0)
    model.train()
    return total / len(loader.dataset)

# ==============================================================================
# 3.   EARLY‑STOPPING CALLBACK
# ==============================================================================
class EarlyStopping:
    def __init__(self, patience: int = 10, delta: float = 1e-4):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best = None
        self.early_stop = False

    def __call__(self, metric: float):
        if self.best is None or metric > self.best + self.delta:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# ==============================================================================
# 4.   MAIN
# ==============================================================================

def main():
    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(0.95)

    print(f"Using device: {Config.DEVICE}")

    # --------------- Augmentations ---------------
    train_tf = A.Compose([
        A.Resize(Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH),
        A.Rotate(limit=35, p=0.7),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Normalize(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255),
        ToTensorV2(),
    ])
    val_tf = A.Compose([
        A.Resize(Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH),
        A.Normalize(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255),
        ToTensorV2(),
    ])

    train_ds = CloudDataset(Config.TRAIN_IMG_DIR, Config.TRAIN_MASK_DIR, transform=train_tf)
    val_ds   = CloudDataset(Config.VAL_IMG_DIR, Config.VAL_MASK_DIR,   transform=val_tf)

    # --------- Weighted sampler para sobre-representar clases raras ---------
    sampler = WeightedRandomSampler(train_ds.img_weights, num_samples=len(train_ds.img_weights), replacement=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        sampler=sampler,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    imprimir_distribucion_clases_post_augmentation(train_loader, 6, "Distribución de clases en ENTRENAMIENTO (post-aug)")

    # ---------------- Model & optimizer ----------------
    model = CloudDeepLabV3Plus(num_classes=6).to(Config.DEVICE)
    print("Compiling model …")
    torch._inductor.config.triton.unique_kernel_names = True
    torch._inductor.config.epilogue_fusion = "max"
    model = torch.compile(model, mode="max-autotune", fullgraph=True, dynamic=False)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.3, patience=20, min_lr=1e-7, verbose=True)
    scaler = GradScaler()
    stopper = EarlyStopping(patience=10)

    best_miou = -1.0
    train_hist, val_hist = [], []

    for epoch in range(Config.NUM_EPOCHS):
        if stopper.early_stop:
            print("\n⚠️  Early stopping activado – se alcanzó la paciencia máxima.")
            break

        print(f"\n—— Epoch {epoch + 1}/{Config.NUM_EPOCHS} ——")
        print("Calculando métricas de entrenamiento…")
        train_miou = train_fn(train_loader, model, optimizer, loss_fn, scaler)
        val_loss = validate_fn(val_loader, model, loss_fn)

        print("Calculando métricas de validación…")
        val_miou, val_dice = check_metrics(val_loader, model, n_classes=6, device=Config.DEVICE)

        scheduler.step(val_loss)
        stopper(val_miou)

        train_hist.append(train_miou.item())
        val_hist.append(val_miou.item())

        if val_miou > best_miou:
            best_miou = val_miou
            print(f"🔹 Nuevo mejor mIoU: {best_miou:.4f} | Dice: {val_dice:.4f}  →  guardando modelo…")
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_mIoU": best_miou,
            }, Config.MODEL_SAVE_PATH)

    # --------- Curva de aprendizaje ---------
    save_performance_plot(train_hist, val_hist, "/content/drive/MyDrive/colab/rendimiento_miou.png")

    # --------- Evaluación final ---------
    print("\nEvaluando el mejor checkpoint…")
    ckpt = torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    best_miou, best_dice = check_metrics(val_loader, model, n_classes=6, device=Config.DEVICE)
    print(f"🏁 mIoU final del mejor modelo: {best_miou:.4f} | Dice: {best_dice:.4f}")


if __name__ == "__main__":
    main()
