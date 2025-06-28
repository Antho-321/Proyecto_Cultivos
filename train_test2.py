import os
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import CloudDeepLabV3Plus
from utils import (
    imprimir_distribucion_clases_post_augmentation,
    crop_around_classes,
    save_performance_plot,
)
from config import Config
import kornia.augmentation as K
from kornia.augmentation import AugmentationSequential
from kornia.augmentation.auto import RandAugment


def mascara_contiene_solo_0_y_4(mask: np.ndarray) -> bool:
    """
    Verifica si una máscara contiene únicamente píxeles con valor 0 y 4.
    """
    return set(np.unique(mask)) == {0, 4}


class CloudDataset(torch.utils.data.Dataset):
    _IMG_EXTENSIONS = ('.jpg', '.png')

    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        transform: AugmentationSequential | None = None,
        special_transform: AugmentationSequential | None = None,
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.special_transform = special_transform
        self.samples = []

        all_images = [
            f for f in os.listdir(image_dir)
            if f.lower().endswith(self._IMG_EXTENSIONS)
        ]

        for img_filename in tqdm(all_images, desc="Procesando imágenes iniciales"):
            self.samples.append({"image_filename": img_filename, "apply_special_aug": False})

            if self.special_transform:
                mask_path = self._mask_path_from_image_name(img_filename)
                if os.path.exists(mask_path):
                    mask = np.array(Image.open(mask_path).convert("L"))
                    if mascara_contiene_solo_0_y_4(mask):
                        for _ in range(Config.NUM_SPECIAL_AUGMENTATIONS):
                            self.samples.append({"image_filename": img_filename, "apply_special_aug": True})

        print(f"Dataset inicializado. Tamaño original: {len(all_images)}, Tamaño con aumentación: {len(self.samples)}")

    def __len__(self) -> int:
        return len(self.samples)

    def _mask_path_from_image_name(self, image_filename: str) -> str:
        name_without_ext = image_filename.rsplit('.', 1)[0]
        mask_filename = f"{name_without_ext}_mask.png"
        return os.path.join(self.mask_dir, mask_filename)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        img_path = os.path.join(self.image_dir, sample["image_filename"])
        mask_path = self._mask_path_from_image_name(sample["image_filename"])

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))

        image_cropped, mask_cropped_3d = crop_around_classes(image, np.expand_dims(mask, -1))
        mask_cropped = mask_cropped_3d.squeeze()

        # Convertir a tensores
        img_t  = torch.from_numpy(image_cropped).permute(2,0,1).float() / 255.0
        mask_t = torch.from_numpy(mask_cropped).long()

        img_t, mask_t = img_t.to(Config.DEVICE), mask_t.to(Config.DEVICE)

        # Aplicar transformaciones
        if sample["apply_special_aug"] and self.special_transform:
            img_aug, mask_aug = self.special_transform(img_t, mask_t)
        elif self.transform:
            img_aug, mask_aug = self.transform(img_t, mask_t)
        else:
            img_aug, mask_aug = img_t, mask_t

        return img_aug, mask_aug


def train_fn(loader, model, optimizer, loss_fn, scaler, num_classes=6):
    loop = tqdm(loader, leave=True)
    model.train()

    tp = torch.zeros(num_classes, device=Config.DEVICE)
    fp = torch.zeros(num_classes, device=Config.DEVICE)
    fn = torch.zeros(num_classes, device=Config.DEVICE)

    for data, targets in loop:
        data = data.to(Config.DEVICE, non_blocking=True)
        targets = targets.to(Config.DEVICE, non_blocking=True)

        with autocast(device_type=Config.DEVICE, dtype=torch.float16):
            output = model(data)
            preds = output[0] if isinstance(output, tuple) else output
            loss = loss_fn(preds, targets)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        _, pred_classes = torch.max(preds, dim=1)
        for c in range(num_classes):
            tp[c] += ((pred_classes==c) & (targets==c)).sum()
            fp[c] += ((pred_classes==c) & (targets!=c)).sum()
            fn[c] += ((pred_classes!=c) & (targets==c)).sum()

        loop.set_postfix(loss=loss.item())

    eps = 1e-6
    iou_per_class  = tp / (tp + fp + fn + eps)
    dice_per_class = (2*tp) / (2*tp + fp + fn + eps)
    mean_iou       = torch.nanmean(iou_per_class)

    print(f"Dice por clase: {dice_per_class.cpu().numpy()}")
    print(f"IoU  por clase: {iou_per_class.cpu().numpy()}")
    print(f"mIoU: {mean_iou:.4f}")

    return mean_iou


def check_metrics(loader, model, n_classes=6, device=None):
    device = device or Config.DEVICE
    eps = 1e-8
    inter_sum = torch.zeros(n_classes, device=device, dtype=torch.float64)
    union_sum = torch.zeros_like(inter_sum)

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            output = model(x)
            logits = output[0] if isinstance(output, tuple) else output
            preds  = torch.argmax(logits, dim=1)
            for c in range(n_classes):
                pred_c = (preds==c)
                true_c = (y==c)
                inter = (pred_c & true_c).sum().double()
                union = pred_c.sum().double() + true_c.sum().double() - inter
                inter_sum[c] += inter
                union_sum[c] += union

    iou_per_class = (inter_sum + eps) / (union_sum + eps)
    miou = iou_per_class.mean()
    print(f"mIoU val: {miou:.4f}")
    model.train()
    return miou


torch.backends.cudnn.benchmark = True

def main():
    print(f"Using device: {Config.DEVICE}")

    train_transform = AugmentationSequential(
        K.RandomHorizontalFlip(p=0.5),
        K.RandomVerticalFlip(p=0.5),
        K.RandomRotation(90.0, p=0.5),
        K.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.15, p=1.0),
        K.Resize((Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH), resample='bilinear'),
        K.Normalize(mean=torch.tensor([0.5,0.5,0.5]), std=torch.tensor([0.5,0.5,0.5])),
        data_keys=["input", "mask"],
    ).to(Config.DEVICE)

    special_aug_transform = AugmentationSequential(
        RandAugment(n=Config.NUM_SPECIAL_AUGMENTATIONS, m=7),
        data_keys=["input", "mask"],
    ).to(Config.DEVICE)

    val_transform = AugmentationSequential(
        K.Resize((Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH), resample='nearest'),
        K.Normalize(mean=torch.tensor([0.5,0.5,0.5]), std=torch.tensor([0.5,0.5,0.5])),
        data_keys=["input", "mask"],
    ).to(Config.DEVICE)

    train_ds = CloudDataset(Config.TRAIN_IMG_DIR, Config.TRAIN_MASK_DIR, transform=train_transform, special_transform=special_aug_transform)
    val_ds   = CloudDataset(Config.VAL_IMG_DIR, Config.VAL_MASK_DIR, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY, shuffle=False)

    imprimir_distribucion_clases_post_augmentation(train_loader, 6, "Distribución de clases en ENTRENAMIENTO (post-aug)")

    model     = CloudDeepLabV3Plus(num_classes=6).to(Config.DEVICE)
    model     = torch.compile(model, mode="max-autotune")
    loss_fn   = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    scaler    = GradScaler()

    best_miou = -1.0
    train_hist, val_hist = [], []

    for epoch in range(Config.NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{Config.NUM_EPOCHS} ---")
        train_miou = train_fn(train_loader, model, optimizer, loss_fn, scaler)
        val_miou   = check_metrics(val_loader, model)
        train_hist.append(train_miou.item())
        val_hist.append(val_miou.item())

        if val_miou > best_miou:
            best_miou = val_miou
            print(f"🔹 Nuevo mejor mIoU: {best_miou:.4f} → guardando modelo…")
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_mIoU": best_miou,
            }, Config.MODEL_SAVE_PATH)

    save_performance_plot(train_history=train_hist, val_history=val_hist, save_path=Config.MODEL_PERF_PATH)

    print("\nEvaluando mejor modelo…")
    ckpt = torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    check_metrics(val_loader, model)


if __name__ == '__main__':
    main()
