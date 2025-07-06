# =============================
#  train.py
# =============================

import os
import numpy as np
from PIL import Image
from typing import Tuple, List

import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2

from model import CloudDeepLabV3Plus  # ↳ tu arquitectura
from utils import (
    imprimir_distribucion_clases_post_augmentation,
    crop_around_classes,
    save_performance_plot,
)
from config import Config  # ↳ mismo Config, solo añade MAX_LR si quieres tunearlo

# ---------------------------------------------
# 1. Focal Loss (versión multi-clase, γ=2)
# ---------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        logpt = F.log_softmax(input, dim=1)
        pt    = torch.exp(logpt)
        logpt = (1 - pt) ** self.gamma * logpt  # focal factor
        loss  = F.nll_loss(logpt, target, weight=self.weight, reduction=self.reduction)
        return loss

# -----------------------------------------------------
# 2. Dataset: igual que antes (recorte antes de aug)
# -----------------------------------------------------
class CloudDataset(torch.utils.data.Dataset):
    _IMG_EXTENSIONS = (".jpg", ".png")

    def __init__(self, image_dir: str, mask_dir: str, transform: A.Compose | None = None):
        self.image_dir = image_dir
        self.mask_dir  = mask_dir
        self.transform = transform
        self.images    = [f for f in os.listdir(image_dir) if f.lower().endswith(self._IMG_EXTENSIONS)]

    def __len__(self):
        return len(self.images)

    def _mask_path_from_image_name(self, image_filename: str) -> str:
        name_wo_ext = image_filename.rsplit(".", 1)[0]
        return os.path.join(self.mask_dir, f"{name_wo_ext}_mask.png")

    def __getitem__(self, idx: int):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = self._mask_path_from_image_name(img_name)
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Máscara no encontrada para {img_name} → {mask_path}")

        image = np.array(Image.open(img_path).convert("RGB"))
        mask  = np.array(Image.open(mask_path).convert("L"))

        # --- recorte alrededor de las clases antes de Aug ---
        img_c, mask_c_3d = crop_around_classes(image, np.expand_dims(mask, -1))
        mask_c = mask_c_3d.squeeze()

        if self.transform:
            aug = self.transform(image=img_c, mask=mask_c)
            image, mask = aug["image"], aug["mask"]

        return image, mask

# --------------------------------------------------------
# 3. Función de entrenamiento (scheduler step-wise y loss compuesta,
#    plus vectorized metrics)
# --------------------------------------------------------
def make_loss_fn(class_weights: torch.Tensor) -> nn.Module:
    ce    = nn.CrossEntropyLoss(weight=class_weights)
    focal = FocalLoss(gamma=2.0, weight=class_weights)

    class ComboLoss(nn.Module):
        def __init__(self, ce, focal):
            super().__init__()
            self.ce, self.focal = ce, focal
        def forward(self, inp, tgt):
            return 0.5 * self.ce(inp, tgt) + 0.5 * self.focal(inp, tgt)
    return ComboLoss(ce, focal)


def train_fn(loader, model, optimizer, scheduler, loss_fn, scaler, num_classes: int = 6):
    model.train()
    tp = torch.zeros(num_classes, device=Config.DEVICE)
    fp = torch.zeros_like(tp)
    fn = torch.zeros_like(tp)

    loop = tqdm(loader, leave=False)
    for data, targets in loop:
        data    = data.to(Config.DEVICE, non_blocking=True)
        targets = targets.to(Config.DEVICE, non_blocking=True).long()

        with autocast(dtype=torch.float16):
            logits = model(data)
            logits = logits[0] if isinstance(logits, tuple) else logits
            loss   = loss_fn(logits, targets)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()  # one-cycle actualiza cada iteración

        preds = torch.argmax(logits, dim=1)
        # 📈 Métricas vectorizadas con bincount
        conf = (
            torch.bincount(
                (preds * num_classes + targets).view(-1),
                minlength=num_classes**2,
            )
            .view(num_classes, num_classes)
            .to(tp.dtype)
        )
        tp += torch.diag(conf)
        fp += conf.sum(0) - torch.diag(conf)
        fn += conf.sum(1) - torch.diag(conf)

        loop.set_postfix(loss=loss.item(), lr=scheduler.get_last_lr()[0])

    iou  = tp / (tp + fp + fn + 1e-6)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-6)
    print(f"\n· Train Dice por clase: {dice.cpu().numpy()}")
    print(f"· Train IoU  por clase: {iou.cpu().numpy()}")
    print(f"· mIoU: {iou.mean():.4f} | mDice: {dice.mean():.4f}")
    return iou.mean()


# --------------------------------------------------------
# 4. Validación  (sin cambios salvo dtype consistente)
# --------------------------------------------------------
@torch.no_grad()
def check_metrics(loader, model, n_classes: int = 6):
    eps = 1e-8
    conf = torch.zeros((n_classes, n_classes), dtype=torch.float16, device=Config.DEVICE)
    model.eval()
    for x, y in loader:
        x = x.to(Config.DEVICE, non_blocking=True)
        y = y.to(Config.DEVICE).long()
        logits = model(x)
        logits = logits[0] if isinstance(logits, tuple) else logits
        preds = torch.argmax(logits, 1)
        conf += torch.bincount(
            (preds * n_classes + y).view(-1),
            minlength=n_classes**2
        ).view(n_classes, n_classes).to(conf.dtype)

    inter = torch.diag(conf)
    pred_sum, true_sum = conf.sum(1), conf.sum(0)
    union = pred_sum + true_sum - inter
    iou  = (inter + eps) / (union + eps)
    dice = (2*inter + eps) / (pred_sum + true_sum + eps)
    print("IoU val :", iou.cpu().numpy())
    print("Dice val:", dice.cpu().numpy())
    print(f"mIoU: {iou.mean():.4f} | mDice: {dice.mean():.4f}")
    return iou.mean(), dice.mean()


# --------------------------------------------------------
# 5. Main
# --------------------------------------------------------
def main():
    # ✅ Más flags de backend
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    print(f"Using device: {Config.DEVICE}")

    # ---------- Transforms ----------
    train_tf = A.Compose([
        A.Resize(Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH),
        A.Rotate(limit=35, p=0.7),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Normalize(mean=(0,0,0), std=(1,1,1), max_pixel_value=255),
        ToTensorV2(),
    ])
    val_tf = A.Compose([
        A.Resize(Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH),
        A.Normalize(mean=(0,0,0), std=(1,1,1), max_pixel_value=255),
        ToTensorV2(),
    ])

    train_ds = CloudDataset(Config.TRAIN_IMG_DIR, Config.TRAIN_MASK_DIR, train_tf)
    val_ds   = CloudDataset(Config.VAL_IMG_DIR,   Config.VAL_MASK_DIR,   val_tf)

    # 🔄 DataLoader optimizado
    train_ld = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        pin_memory_device=Config.DEVICE,
        persistent_workers=True,
        prefetch_factor=4,
    )
    val_ld = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        pin_memory_device=Config.DEVICE,
        persistent_workers=True,
        prefetch_factor=4,
    )

    # ---------- Distribución y pesos ----------
    class_weights = torch.tensor(
        [0.0549, 1.1832, 1.2697, 1.0435, 2.0081, 0.4406],
        device=Config.DEVICE
    )

    imprimir_distribucion_clases_post_augmentation(
        train_ld, 6, "Distribución de clases en ENTRENAMIENTO (post-aug)"
    )

    # 🏗 Modelo con memory_format y compilación
    model = CloudDeepLabV3Plus(num_classes=6)
    model = model.to(Config.DEVICE, memory_format=torch.channels_last)
    model = torch.compile(
        model,
        mode="max-autotune",
        fullgraph=True,
        dynamic=False
    )

    # ⚙️ Optimizador fusionado
    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
        fused=True
    )

    total_steps = len(train_ld) * Config.NUM_EPOCHS
    scheduler = OneCycleLR(
        optimizer,
        max_lr=getattr(Config, "MAX_LR", 5e-4),
        total_steps=total_steps,
        pct_start=0.2,
        div_factor=25,
        final_div_factor=1e3,
        anneal_strategy="cos",
    )

    loss_fn = make_loss_fn(class_weights)
    # 🎛 GradScaler con growth_interval
    scaler = GradScaler(growth_interval=2000)

    best_miou = -1.0
    train_hist, val_hist = [], []

    for epoch in range(Config.NUM_EPOCHS):
        print(f"\n—— Epoch {epoch+1}/{Config.NUM_EPOCHS} ——")
        train_miou = train_fn(train_ld, model, optimizer, scheduler, loss_fn, scaler)
        val_miou, val_dice = check_metrics(val_ld, model)

        train_hist.append(train_miou.item())
        val_hist.append(val_miou.item())

        if val_miou > best_miou:
            best_miou = val_miou
            torch.save({
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "miou": best_miou,
                "epoch": epoch,
            }, Config.MODEL_SAVE_PATH)
            print(f"🔹 Nuevo mejor mIoU {best_miou:.4f} – modelo guardado.")

    save_performance_plot(train_hist, val_hist, "/content/drive/MyDrive/colab/rendimiento_miou.png")

    # ------- Cargar y evaluar el mejor modelo --------
    ckpt = torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    best_miou, best_dice = check_metrics(val_ld, model)
    print(f"🏁 Mejor mIoU: {best_miou:.4f} | Dice: {best_dice:.4f}")


if __name__ == "__main__":
    main()
