# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
from PIL import Image
import numpy as np

from model2 import DeepLabV3Plus_EfficientNetV2S
from utils import imprimir_distribucion_clases_post_augmentation, crop_around_classes, save_performance_plot
from config2 import Config

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark      = True
torch.set_float32_matmul_precision("high")

# ================================
# 1. DATASET PERSONALIZADO
# ================================
class CloudDataset(torch.utils.data.Dataset):
    _IMG_EXTENSIONS = ('.jpg', '.png')

    def __init__(self, image_dir: str, mask_dir: str, transform: A.Compose | None = None):
        self.image_dir = image_dir
        self.mask_dir  = mask_dir
        self.transform = transform
        self.images    = [f for f in os.listdir(image_dir)
                          if f.lower().endswith(self._IMG_EXTENSIONS)]

    def __len__(self) -> int:
        return len(self.images)

    def _mask_path_from_image_name(self, image_filename: str) -> str:
        name = image_filename.rsplit('.', 1)[0]
        return os.path.join(self.mask_dir, f"{name}_mask.png")

    def __getitem__(self, idx: int):
        img_name  = self.images[idx]
        img_path  = os.path.join(self.image_dir, img_name)
        mask_path = self._mask_path_from_image_name(img_name)
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Máscara no encontrada para {img_name}")

        image = np.array(Image.open(img_path).convert("RGB"))
        mask  = np.array(Image.open(mask_path).convert("L"))
        mask_3d = np.expand_dims(mask, -1)
        image_c, mask_c_3d = crop_around_classes(image, mask_3d)
        mask_c = mask_c_3d.squeeze()

        if self.transform:
            aug = self.transform(image=image_c, mask=mask_c)
            image, mask = aug["image"], aug["mask"]

        return image, mask

# ================================
# 2. TRAIN & VALID FUNCTIONS
# ================================
def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader, leave=True)
    model.train()
    epoch_loss = 0.0

    for data, targets in loop:
        data, targets = data.to(Config.DEVICE), targets.to(Config.DEVICE)
        with autocast(device_type=Config.DEVICE, dtype=torch.float16):
            outputs = model(data)
            preds   = outputs[0] if isinstance(outputs, tuple) else outputs
            loss    = loss_fn(preds, targets)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item() * data.size(0)
        loop.set_postfix(train_loss=loss.item())

    return epoch_loss / len(loader.dataset)

def mean_iou(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    preds: Tensor de forma (B,H,W) con etiquetas predichas (0…num_classes-1)
    targets: Tensor de forma (B,H,W) con etiquetas reales
    """
    ious = []
    for cls in range(num_classes):
        pred_inds   = (preds == cls)
        target_inds = (targets == cls)
        intersection = (pred_inds & target_inds).sum().float()
        union        = (pred_inds | target_inds).sum().float()
        if union == 0:
            # Si no hay pixeles de esta clase en GT ni predicción, lo ignoramos
            continue
        ious.append(intersection / union)
    if not ious:
        return torch.tensor(0.0, device=preds.device)
    return torch.stack(ious).mean()

# ---------------------------------------------------
# 2) Modifica validate_fn para devolver también el mIoU
# ---------------------------------------------------
def validate_fn(loader, model, loss_fn, num_classes):
    model.eval()
    val_loss = 0.0
    iou_scores = []
    with torch.no_grad():
        for data, targets in loader:
            data, targets = data.to(Config.DEVICE), targets.to(Config.DEVICE)
            outputs = model(data)
            preds   = outputs[0] if isinstance(outputs, tuple) else outputs

            # 2.1 Calculamos pérdida
            loss = loss_fn(preds, targets)
            val_loss += loss.item() * data.size(0)

            # 2.2 Convertimos predicción a etiquetas
            # Si tu salida es multi‐clase con logits por canal:
            probs  = torch.softmax(preds, dim=1)              # (B, C, H, W)
            labels = probs.argmax(dim=1)                       # (B, H, W)

            # Asegúrate de que targets venga como (B,H,W) con etiquetas 0…C-1
            iou_batch = mean_iou(labels, targets, num_classes)
            iou_scores.append(iou_batch)

    avg_loss = val_loss / len(loader.dataset)
    mean_iou_epoch = torch.stack(iou_scores).mean().item()
    return avg_loss, mean_iou_epoch

# ================================
# 3. MAIN
# ================================
def main():
    print(f"Device: {Config.DEVICE}")

    train_tf = A.Compose([
        A.Resize(Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH),
        A.Rotate(limit=35, p=0.7),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Normalize(mean=[0,0,0], std=[1,1,1], max_pixel_value=255.0),
        ToTensorV2(),
    ])
    val_tf = A.Compose([
        A.Resize(Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH),
        A.Normalize(mean=[0,0,0], std=[1,1,1], max_pixel_value=255.0),
        ToTensorV2(),
    ])

    train_ds = CloudDataset(Config.TRAIN_IMG_DIR, Config.TRAIN_MASK_DIR, transform=train_tf)
    val_ds   = CloudDataset(Config.VAL_IMG_DIR,   Config.VAL_MASK_DIR,   transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE,
                              num_workers=Config.NUM_WORKERS,
                              pin_memory=Config.PIN_MEMORY, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=Config.BATCH_SIZE,
                              num_workers=Config.NUM_WORKERS,
                              pin_memory=Config.PIN_MEMORY, shuffle=False)

    imprimir_distribucion_clases_post_augmentation(train_loader, 6,
        "Distribución de clases en ENTRENAMIENTO (post-aug)")

    model = DeepLabV3Plus_EfficientNetV2S(num_classes=6).to(Config.DEVICE)
    print("Compiling the model... (this may take a minute)")
    torch._inductor.config.triton.unique_kernel_names = True
    torch._inductor.config.epilogue_fusion           = "max"
    model = torch.compile(model, mode="max-autotune")
    loss_fn   = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='min',
                                  factor=0.1, patience=5,
                                  min_lr=1e-6, verbose=True)
    scaler    = GradScaler()
    best_loss = float('inf')

    train_losses, val_losses = [], []

    for epoch in range(Config.NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{Config.NUM_EPOCHS} ---")
        train_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler)
        val_loss, val_miou = validate_fn(val_loader, model, loss_fn, num_classes=6)
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val mIoU: {val_miou:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            print(f"🔹 Nuevo mejor val_loss: {best_loss:.4f}, guardando modelo…")
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'opt_state': optimizer.state_dict(),
                'best_loss': best_loss
            }, Config.MODEL_SAVE_PATH)

    save_performance_plot(train_losses, val_losses, save_path=Config.PERFORMANCE_PATH)

if __name__ == "__main__":
    main()
