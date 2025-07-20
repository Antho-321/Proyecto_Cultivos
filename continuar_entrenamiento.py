# ==================================================================================================
# 0. IMPORTS Y DESCARGA DEL MODELO PRE-ENTRENADO
# ==================================================================================================
import os
import contextlib
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2

from model2 import CloudDeepLabV3Plus
from utils import imprimir_distribucion_clases_post_augmentation
from config import Config

# ==================================================================================================
# 1. TRANSFORMACIONES
# ==================================================================================================
base_tf = A.Compose([
    A.Resize(Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH),
    A.Rotate(limit=35, p=0.7),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    # photometric
    A.RandomBrightnessContrast(p=0.4),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15,
                         val_shift_limit=10, p=0.3),
    # normalización + tensor
    A.Normalize(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255.0),
    ToTensorV2(),
], additional_targets={'mask': 'mask'})

extra_tf = A.Compose([
    # sólo para máscaras binarias {0,4}
    A.RandomResizedCrop(Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH,
                        scale=(0.5, 1.0), ratio=(0.8, 1.2), p=0.6),
    A.ElasticTransform(alpha=40, sigma=50, alpha_affine=20, p=0.4),
    A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
    A.GaussianBlur(blur_limit=3, p=0.25),
    A.CoarseDropout(max_holes=8, max_height=32, max_width=32,
                    fill_value=0, mask_fill_value=0, p=0.3),
])


# ==================================================================================================
# 2. DATASET
# ==================================================================================================
class CloudDataset(torch.utils.data.Dataset):
    _IMG_EXTENSIONS = (".jpg", ".png")

    def __init__(self, image_dir: str, mask_dir: str,
                 base_tf: A.Compose, extra_tf: A.Compose | None = None):
        self.image_dir, self.mask_dir = image_dir, mask_dir
        self.base_tf, self.extra_tf   = base_tf, extra_tf
        self.images = [f for f in os.listdir(image_dir)
                       if f.lower().endswith(self._IMG_EXTENSIONS)]

    def __len__(self) -> int:
        return len(self.images)

    def _mask_path_from_image_name(self, image_filename: str) -> str:
        stem = image_filename.rsplit(".", 1)[0]
        return os.path.join(self.mask_dir, f"{stem}_mask.png")

    def __getitem__(self, idx: int):
        img_name  = self.images[idx]
        img_path  = os.path.join(self.image_dir, img_name)
        mask_path = self._mask_path_from_image_name(img_name)

        image = np.array(Image.open(img_path).convert("RGB"))
        mask  = np.array(Image.open(mask_path).convert("L"))

        uniq       = np.unique(mask)
        only_0_4   = (len(uniq) <= 2) and set(uniq).issubset({0, 4})
        transform  = self.base_tf

        if only_0_4 and self.extra_tf is not None:
            # evita duplicar Normalize + ToTensorV2:
            transform = A.Compose(self.extra_tf.transforms +
                                  self.base_tf.transforms[-2:])

        augmented = transform(image=image, mask=mask)
        return augmented["image"], augmented["mask"]


# ==================================================================================================
# 3. ENTRENAMIENTO Y VALIDACIÓN
# ==================================================================================================
def train_fn(loader, model, optimizer, loss_fn, scaler, num_classes: int = 6):
    model.train()
    loop = tqdm(loader, leave=False)
    tp = fp = fn = torch.zeros(num_classes, device=Config.DEVICE)

    for data, targets in loop:
        data, targets = data.to(Config.DEVICE, non_blocking=True), targets.to(Config.DEVICE, non_blocking=True).long()
        with autocast(device_type="cuda", dtype=torch.float16):
            logits = model(data)[0] if isinstance(model(data), tuple) else model(data)
            loss   = loss_fn(logits, targets)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        preds = logits.argmax(1)
        for c in range(num_classes):
            tp[c] += ((preds == c) & (targets == c)).sum()
            fp[c] += ((preds == c) & (targets != c)).sum()
            fn[c] += ((preds != c) & (targets == c)).sum()

        loop.set_postfix(loss=loss.item())

    eps = 1e-6
    iou  = tp / (tp + fp + fn + eps)
    dice = (2 * tp) / (2 * tp + fp + fn + eps)
    print(f"IoU  {torch.nanmean(iou):.4f}  |  Dice  {torch.nanmean(dice):.4f}")


@torch.no_grad()
def check_metrics(loader, model, n_classes: int = 6, device: str = "cuda", use_amp: bool = True):
    model.eval().to(device)
    conf = torch.zeros((n_classes, n_classes), device=device, dtype=torch.int64)
    amp_ctx = autocast(device_type="cuda") if use_amp else contextlib.nullcontext()

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True).int()
        with amp_ctx:
            logits = model(x)[0] if isinstance(model(x), tuple) else model(x)
        preds = logits.argmax(1)
        conf += torch.bincount((preds * n_classes + y).view(-1),
                               minlength=n_classes**2).view(n_classes, n_classes)

    inter = conf.diag().float()
    union = conf.sum(1) + conf.sum(0) - inter + 1e-6
    miou  = (inter / union).mean()
    mdice = ((2 * inter) / (conf.sum(1) + conf.sum(0) + 1e-6)).mean()
    return float(miou), float(mdice)


# ==================================================================================================
# 4. MAIN
# ==================================================================================================
def main(resume: bool = True):
    torch.backends.cudnn.benchmark = True
    print("Device:", Config.DEVICE)

    train_ds = CloudDataset(Config.TRAIN_IMG_DIR, Config.TRAIN_MASK_DIR,
                            base_tf=base_tf, extra_tf=extra_tf)
    val_ds   = CloudDataset(Config.VAL_IMG_DIR,   Config.VAL_MASK_DIR,
                            base_tf=base_tf)

    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True,
                              num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY)
    val_loader   = DataLoader(val_ds,   batch_size=Config.BATCH_SIZE, shuffle=False,
                              num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY)

    imprimir_distribucion_clases_post_augmentation(train_loader, 6,
        "Distribución de clases en ENTRENAMIENTO (post-aug)")

    model     = CloudDeepLabV3Plus(num_classes=6).to(Config.DEVICE)
    torch._inductor.config.triton.cudagraphs = True
    model     = torch.compile(model)
    loss_fn   = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scaler    = GradScaler()

    start_epoch, best_miou = 0, -1.0
    if resume and os.path.isfile(Config.MODEL_SAVE_PATH):
        ckpt = torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE)
        model.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_miou   = ckpt.get("best_mIoU", best_miou)
        print(f"🔄 Reanudado en epoch {start_epoch} | best mIoU {best_miou:.4f}")

    for epoch in range(start_epoch, Config.NUM_EPOCHS):
        print(f"\n— Epoch {epoch+1}/{Config.NUM_EPOCHS}")
        train_fn(train_loader, model, optimizer, loss_fn, scaler)
        miou, mdice = check_metrics(val_loader, model, n_classes=6, device=Config.DEVICE)
        print(f"Val mIoU {miou:.4f} | Dice {mdice:.4f}")

        if miou > best_miou:
            best_miou = miou
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_mIoU": best_miou,
            }, Config.MODEL_SAVE_PATH)
            print("✅ Nuevo checkpoint guardado")

    print("\nEvaluando mejor checkpoint…")
    ckpt = torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    miou, mdice = check_metrics(val_loader, model, n_classes=6, device=Config.DEVICE)
    print(f"🏆 Best checkpoint ⇒ mIoU {miou:.4f} | Dice {mdice:.4f}")


if __name__ == "__main__":
    main(resume=True)
