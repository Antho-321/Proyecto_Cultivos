import os
import contextlib
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


# ================================================================================
# 1. DATASET
# ================================================================================
class CloudDataset(torch.utils.data.Dataset):
    _IMG_EXTENSIONS = (".jpg", ".png")

    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        transform: A.Compose | None = None,
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = [
            f for f in os.listdir(image_dir) if f.lower().endswith(self._IMG_EXTENSIONS)
        ]

    def __len__(self) -> int:
        return len(self.images)

    def _mask_path_from_image_name(self, image_filename: str) -> str:
        stem = image_filename.rsplit(".", 1)[0]
        return os.path.join(self.mask_dir, f"{stem}_mask.png")

    def __getitem__(self, idx: int):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = self._mask_path_from_image_name(img_name)

        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Máscara no encontrada: {mask_path}")

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))

        if self.transform:
            result = self.transform(image=image, mask=mask)
            image, mask = result["image"], result["mask"]

        return image, mask


# ================================================================================
# 2. ENTRENAMIENTO
# ================================================================================
def train_fn(loader, model, optimizer, loss_fn, scaler, num_classes: int = 6):
    model.train()
    loop = tqdm(loader, leave=True)

    tp = torch.zeros(num_classes, device=Config.DEVICE)
    fp = torch.zeros(num_classes, device=Config.DEVICE)
    fn = torch.zeros(num_classes, device=Config.DEVICE)

    for data, targets in loop:
        data = data.to(Config.DEVICE, non_blocking=True)
        targets = targets.to(Config.DEVICE, non_blocking=True).long()

        with autocast(device_type="cuda", dtype=torch.float16):
            output = model(data)
            logits = output[0] if isinstance(output, tuple) else output
            loss = loss_fn(logits, targets)

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
    iou = tp / (tp + fp + fn + eps)
    dice = (2 * tp) / (2 * tp + fp + fn + eps)
    print("\nTrain finished:")
    print("IoU :", iou)
    print("Dice:", dice)
    print("mIoU :", torch.nanmean(iou))
    print("mDice:", torch.nanmean(dice))


# ================================================================================
# 3. VALIDACIÓN
# ================================================================================
@torch.no_grad()
def check_metrics(
    loader,
    model,
    n_classes: int = 6,
    device: str = "cuda",
    use_amp: bool = True,
    compile_model: bool = False,
):
    if compile_model and hasattr(torch, "compile") and not isinstance(
        model, torch._dynamo.OptimizedModule
    ):
        model = torch.compile(model, mode="reduce-overhead", dynamic=True)

    model = model.to(device).eval()

    conf_mat = torch.zeros((n_classes, n_classes), device=device, dtype=torch.int32)
    amp_ctx = autocast(device_type="cuda") if use_amp else contextlib.nullcontext()

    for x, y in loader:
        x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        y = y.to(device, non_blocking=True).int()

        with amp_ctx:
            logits = model(x)
            logits = logits[0] if isinstance(logits, tuple) else logits

        preds = logits.argmax(1)
        hist = torch.bincount(
            (preds * n_classes + y).view(-1), minlength=n_classes * n_classes
        ).view(n_classes, n_classes)
        conf_mat += hist.to(conf_mat.dtype)

    inter = conf_mat.diag().float()
    sum_pred = conf_mat.sum(1).float()
    sum_truth = conf_mat.sum(0).float()
    union = sum_pred + sum_truth - inter + 1e-6

    miou = (inter / union).mean()
    mdice = ((2 * inter) / (sum_pred + sum_truth + 1e-6)).mean()
    print("\nValidation finished:")
    print("mIoU :", miou)
    print("mDice:", mdice)
    return miou, mdice


# ================================================================================
# 4. MAIN
# ================================================================================
def main():
    torch.backends.cudnn.benchmark = True
    print(f"Device: {Config.DEVICE}")

    train_tf = A.Compose(
        [
            A.Resize(Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH),
            A.Rotate(limit=35, p=0.7),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Normalize(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255.0),
            ToTensorV2(),
        ]
    )
    val_tf = A.Compose(
        [
            A.Resize(Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH),
            A.Normalize(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255.0),
            ToTensorV2(),
        ]
    )

    train_ds = CloudDataset(Config.TRAIN_IMG_DIR, Config.TRAIN_MASK_DIR, train_tf)
    val_ds = CloudDataset(Config.VAL_IMG_DIR, Config.VAL_MASK_DIR, val_tf)
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
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

    imprimir_distribucion_clases_post_augmentation(
        train_loader, 6, "Distribución de clases en ENTRENAMIENTO (post-aug)"
    )

    model = CloudDeepLabV3Plus(num_classes=6).to(Config.DEVICE)
    torch._inductor.config.triton.cudagraphs = True
    model = torch.compile(
        model,
        backend="inductor",            # the default ML-compiler backend
        mode="max-autotune",           # profiles multiple kernels for best speed
        fullgraph=True,                # fuse as much of the model into one big graph
        dynamic=True,                  # optionally generate dynamic-shape kernels
        options={
            "epilogue_fusion": True,   # fuse pointwise ops into templates (requires max-autotune)
            "shape_padding": True,     # pad tensor shapes for better Tensor-Core alignment
            # you can also tweak other flags, e.g. "fallback_random" or "triton.cudagraphs"
        }
    )

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scaler = GradScaler()
    best_miou = -1.0

    for epoch in range(Config.NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{Config.NUM_EPOCHS} ---")
        train_fn(train_loader, model, optimizer, loss_fn, scaler)
        miou, mdice = check_metrics(val_loader, model, n_classes=6, device=Config.DEVICE)

        if miou > best_miou:
            best_miou = miou
            print(f"🔹 New best mIoU {miou:.4f} | Dice {mdice:.4f}  →  saving …")
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_mIoU": best_miou,
                },
                Config.MODEL_SAVE_PATH,
            )

    print("\nEvaluating best checkpoint …")
    ckpt = torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    miou, mdice = check_metrics(val_loader, model, n_classes=6, device=Config.DEVICE)
    print(f"Best checkpoint ⇒ mIoU {miou:.4f} | Dice {mdice:.4f}")


if __name__ == "__main__":
    main()