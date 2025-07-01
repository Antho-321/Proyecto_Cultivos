import os
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from config import Config
from model3 import ImprovedDeepLabV3Plus  # Asegúrate de que este apunta al módulo correcto
from utils import crop_around_classes, check_metrics  # Agregamos check_metrics

# ==========================
# 1) Hiperparámetros
# ==========================
batch_size    = 8         # tamaño de lote por iteración
num_epochs    = 200       # número de épocas
learning_rate = 0.001     # tasa de aprendizaje fija

# =================================================================================
# 2. TRANSFORMACIONES (PDF: brillo, color, contraste, flips, rotaciones aleatorias)
# =================================================================================
train_transform = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=360, p=0.5),
    A.Resize(height=Config.IMAGE_HEIGHT, width=Config.IMAGE_WIDTH),
    A.Normalize(mean=[0.0,0.0,0.0], std=[1.0,1.0,1.0], max_pixel_value=255.0),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(height=Config.IMAGE_HEIGHT, width=Config.IMAGE_WIDTH),
    A.Normalize(mean=[0.0,0.0,0.0], std=[1.0,1.0,1.0], max_pixel_value=255.0),
    ToTensorV2(),
])

# =================================================================================
# 3. DATASET PERSONALIZADO (con recorte antes de transformaciones)
# =================================================================================
class CloudDataset(torch.utils.data.Dataset):
    _IMG_EXTENSIONS = ('.jpg', '.png')

    def __init__(self, image_dir: str, mask_dir: str, transform: A.Compose | None = None):
        self.image_dir = image_dir
        self.mask_dir  = mask_dir
        self.transform = transform
        self.images    = [
            f for f in os.listdir(image_dir)
            if f.lower().endswith(self._IMG_EXTENSIONS)
        ]

    def __len__(self) -> int:
        return len(self.images)

    def _mask_path_from_image_name(self, image_filename: str) -> str:
        name_without_ext = image_filename.rsplit('.', 1)[0]
        mask_filename    = f"{name_without_ext}_mask.png"
        return os.path.join(self.mask_dir, mask_filename)

    def __getitem__(self, idx: int):
        img_filename = self.images[idx]
        img_path     = os.path.join(self.image_dir, img_filename)
        mask_path    = self._mask_path_from_image_name(img_filename)

        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Máscara no encontrada para {img_filename}")

        image = np.array(Image.open(img_path).convert("RGB"))
        mask  = np.array(Image.open(mask_path).convert("L"))

        # --- Recorte alrededor de clases antes de augmentación ---
        mask_3d        = np.expand_dims(mask, axis=-1)
        image_cropped, mask_cropped_3d = crop_around_classes(image, mask_3d)
        mask_cropped   = mask_cropped_3d.squeeze()

        if self.transform:
            augmented = self.transform(image=image_cropped, mask=mask_cropped)
            image = augmented["image"]
            mask  = augmented["mask"]

        return image, mask

# =================================================================================
# 4. DATALOADERS
# =================================================================================
train_dataset = CloudDataset(
    image_dir=Config.TRAIN_IMG_DIR,
    mask_dir=Config.TRAIN_MASK_DIR,
    transform=train_transform
)
val_dataset = CloudDataset(
    image_dir=Config.VAL_IMG_DIR,
    mask_dir=Config.VAL_MASK_DIR,
    transform=val_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# =================================================================================
# 5. MODELO, SCALER, PÉRDIDA Y OPTIMIZADOR
# =================================================================================
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model     = ImprovedDeepLabV3Plus(n_classes=6).to(device)
scaler    = GradScaler()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# =================================================================================
# 6. BUCLE DE ENTRENAMIENTO
# =================================================================================
for epoch in range(1, num_epochs + 1):
    # --- Entrenamiento ---
    model.train()
    running_loss = 0.0
    loop = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", leave=False)
    for images, masks in loop:
        images, masks = images.to(device), masks.to(device)
        masks = masks.long()           # <<< add this
        optimizer.zero_grad()
        with autocast(device_type=device.type):
            outputs = model(images)
            loss    = criterion(outputs, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * images.size(0)
        loop.set_postfix(loss=loss.item())

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch}/{num_epochs} — Loss: {epoch_loss:.4f}")

    # --- Validación ---
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss    = criterion(outputs, masks)
            val_loss += loss.item() * images.size(0)
    val_loss /= len(val_loader.dataset)
    print(f"             Val Loss: {val_loss:.4f}")

    # --- Cálculo e impresión de mIoU y Dice --
    miou, dice = check_metrics(val_loader, model, n_classes=6, device=device)
    print(f"             mIoU: {miou:.4f} | Dice: {dice:.4f}\n")
