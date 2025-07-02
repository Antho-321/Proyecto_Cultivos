import os
import random
from glob import glob

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from model2 import ParallelFusionNet
from utils import crop_around_classes
from config import Config

# ────────────────────────────────────────────────────────────────────────────────
# 1. Configuración general
# ────────────────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(42)
torch.manual_seed(42)

INIT_LR      = 1e-3
POWER        = 0.9
BATCH_SIZE   = 16
NUM_EPOCHS   = 100
WEIGHT_DECAY = 1e-4
BETAS        = (0.9, 0.999)

# ────────────────────────────────────────────────────────────────────────────────
# 2. Dataset personalizado
# ────────────────────────────────────────────────────────────────────────────────
class CitrusDataset(Dataset):
    def __init__(self, img_paths, mask_paths, transform=None):
        self.img_paths  = img_paths
        self.mask_paths = mask_paths
        self.transform  = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # 1) cargar como numpy
        img_np  = np.array(Image.open(self.img_paths[idx]).convert("RGB"))
        mask_np = np.array(Image.open(self.mask_paths[idx]).convert("L"))

        # 2) recorte alrededor de clases
        mask_3d           = np.expand_dims(mask_np, axis=-1)
        img_crop, msk_3d  = crop_around_classes(img_np, mask_3d)
        mask_crop         = msk_3d.squeeze()

        # 3) pasar a PIL para Albumentations
        img_pil  = Image.fromarray(img_crop)
        msk_pil  = Image.fromarray(mask_crop.astype(np.uint8))

        # 4) aplicar transformaciones
        if self.transform:
            aug = self.transform(image=img_crop, mask=mask_crop)
            img_tensor = aug["image"]
            msk_tensor = aug["mask"]
        else:
            img_tensor = transforms.ToTensor()(img_pil)
            msk_tensor = transforms.ToTensor()(msk_pil)

        # 5) binarizar máscara
        msk_tensor = (msk_tensor > 0).float()
        return img_tensor, msk_tensor

# ────────────────────────────────────────────────────────────────────────────────
# 3. Albumentations para imagen y máscara
# ────────────────────────────────────────────────────────────────────────────────

train_transform = A.Compose([
    A.Resize(Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH),
    A.RandomHorizontalFlip(p=0.5),
    A.RandomVerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Normalize(mean=(0,0,0), std=(1,1,1), max_pixel_value=255.0),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH),
    A.Normalize(mean=(0,0,0), std=(1,1,1), max_pixel_value=255.0),
    ToTensorV2(),
])

# ────────────────────────────────────────────────────────────────────────────────
# 4. Rutas y DataLoaders
# ────────────────────────────────────────────────────────────────────────────────
train_imgs  = sorted(glob(os.path.join(Config.TRAIN_IMG_DIR, "*.jpg")))
train_masks = sorted(glob(os.path.join(Config.TRAIN_MASK_DIR, "*.png")))

val_imgs    = sorted(glob(os.path.join(Config.VAL_IMG_DIR,   "*.jpg")))
val_masks   = sorted(glob(os.path.join(Config.VAL_MASK_DIR,  "*.png")))

train_ds = CitrusDataset(train_imgs, train_masks, transform=train_transform)
val_ds   = CitrusDataset(val_imgs,  val_masks,   transform=val_transform)

train_loader = DataLoader(train_ds,
                          batch_size=Config.BATCH_SIZE,
                          shuffle=True,
                          num_workers=Config.NUM_WORKERS,
                          pin_memory=Config.PIN_MEMORY)
val_loader   = DataLoader(val_ds,
                          batch_size=Config.BATCH_SIZE,
                          shuffle=False,
                          num_workers=Config.NUM_WORKERS,
                          pin_memory=Config.PIN_MEMORY)

# ────────────────────────────────────────────────────────────────────────────────
# 5. Modelo, optimizador y scheduler
# ────────────────────────────────────────────────────────────────────────────────
model = ParallelFusionNet(num_classes=1).to(DEVICE)
optimizer = optim.AdamW(model.parameters(),
                        lr=INIT_LR,
                        betas=BETAS,
                        weight_decay=WEIGHT_DECAY)

max_iter = NUM_EPOCHS * (len(train_ds) // BATCH_SIZE)
def poly_lr(opt, init_lr, curr_iter, max_iter, power=POWER):
    lr = init_lr * (1 - curr_iter/max_iter)**power
    for pg in opt.param_groups:
        pg['lr'] = lr

# ────────────────────────────────────────────────────────────────────────────────
# 6. Entrenamiento
# ────────────────────────────────────────────────────────────────────────────────
global_iter, best_val = 0, float('inf')
for epoch in range(1, NUM_EPOCHS+1):
    model.train()
    pbar, train_loss = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}"), 0
    for imgs, masks in pbar:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad()
        preds = model(imgs)
        loss  = F.binary_cross_entropy_with_logits(preds, masks)
        loss.backward()
        optimizer.step()

        global_iter += 1
        poly_lr(optimizer, INIT_LR, global_iter, max_iter)

        train_loss += loss.item() * imgs.size(0)
        pbar.set_postfix(loss=loss.item())
    train_loss /= len(train_ds)

    # ――― validación ―――
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            preds = model(imgs)
            val_loss += F.binary_cross_entropy_with_logits(preds, masks).item() * imgs.size(0)
    val_loss /= len(val_ds)

    # ───> AQUÍ agregamos el cálculo e impresión de mIoU en validación
    def compute_iou(pred, target, eps=1e-6):
        pred  = (pred > 0).float()
        inter = (pred * target).sum()
        union = pred.sum() + target.sum() - inter
        return (inter + eps) / (union + eps)

    miou_val = 0.0
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs   = imgs.to(DEVICE)
            logits = model(imgs)
            miou_val += compute_iou(torch.sigmoid(logits), masks.to(DEVICE)).item() * imgs.size(0)
    miou_val /= len(val_ds)
    print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val mIoU: {miou_val:.4f}")

    # guardar si mejora
    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), "best_model.pth")

# ────────────────────────────────────────────────────────────────────────────────
# 7. Test mIoU
# ────────────────────────────────────────────────────────────────────────────────
def compute_iou(pred, target, eps=1e-6):
    pred  = (pred > 0).float()
    inter = (pred*target).sum()
    union = pred.sum()+target.sum()-inter
    return (inter+eps)/(union+eps)

model.load_state_dict(torch.load("best_model.pth"))
model.eval()
miou = sum(
    compute_iou(torch.sigmoid(model(imgs.to(DEVICE))), masks.to(DEVICE)).item() * imgs.size(0)
    for imgs, masks in val_loader
) / len(val_ds)
print(f"Validation mIoU: {miou:.4f}")
