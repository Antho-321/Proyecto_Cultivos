# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
from PIL import Image
import numpy as np

# Importa la arquitectura del otro archivo
from model import CloudDeepLabV3Plus 

# =================================================================================
# 1. CONFIGURACIÓN
# Centraliza todos los hiperparámetros y rutas aquí.
# =================================================================================
class Config:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # --- MUY IMPORTANTE: Modifica estas rutas a tus directorios de datos ---
    TRAIN_IMG_DIR = "Balanced/train/images"
    TRAIN_MASK_DIR = "Balanced/train/masks"
    VAL_IMG_DIR = "Balanced/val/images"
    VAL_MASK_DIR = "Balanced/val/masks"
    
    # --- Hiperparámetros de entrenamiento ---
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 8
    NUM_EPOCHS = 25
    NUM_WORKERS = 2
    
    # --- Dimensiones de la imagen ---
    IMAGE_HEIGHT = 256
    IMAGE_WIDTH = 256
    
    # --- Configuraciones adicionales ---
    PIN_MEMORY = True
    LOAD_MODEL = False # Poner a True si quieres continuar un entrenamiento
    MODEL_SAVE_PATH = "best_model.pth.tar"

# =================================================================================
# 2. DATASET PERSONALIZADO
# Clase para cargar las imágenes y sus máscaras de segmentación.
# =================================================================================
class CloudDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        # Asumimos que las máscaras tienen el mismo nombre de archivo
        mask_path = os.path.join(self.mask_dir, self.images[index])
        
        # Las máscaras pueden tener un formato diferente (ej: .gif, .png), ajusta si es necesario
        # mask_path = mask_path.replace(".jpg", "_mask.gif") 

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32) # L = Grayscale
        
        # Las máscaras de nubes suelen tener valores 0 (fondo) y 255 (nube). Normalizamos a 0 y 1.
        mask[mask == 255.0] = 1.0

        if self.transform:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
            # Añadir una dimensión de canal a la máscara
            mask = mask.unsqueeze(0)

        return image, mask

# =================================================================================
# 3. FUNCIONES DE ENTRENAMIENTO Y VALIDACIÓN
# =================================================================================
def train_fn(loader, model, optimizer, loss_fn, scaler):
    """Procesa una época de entrenamiento."""
    loop = tqdm(loader, leave=True)
    model.train()

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=Config.DEVICE)
        targets = targets.to(device=Config.DEVICE)

        # Forward
        with torch.cuda.amp.autocast(): # Para entrenamiento de precisión mixta
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # Backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Actualizar la barra de progreso
        loop.set_postfix(loss=loss.item())

def check_metrics(loader, model, device="cuda"):
    """Calcula métricas de validación (mIoU, accuracy, dice score)."""
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    intersection_sum = 0
    union_sum = 0
    
    model.eval()

    with torch.no_grad():
        for x, y in tqdm(loader, leave=True, desc="Validating"):
            x = x.to(device)
            y = y.to(device)
            
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float() # Convertir a máscara binaria
            
            # Accuracy
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            
            # Dice Score
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)
            
            # IoU (Intersection over Union)
            intersection = (preds * y).sum()
            union = (preds + y).sum() - intersection # Union = A + B - Intersection
            intersection_sum += intersection
            union_sum += union

    accuracy = num_correct / num_pixels * 100
    avg_dice_score = dice_score / len(loader)
    mIoU = intersection_sum / (union_sum + 1e-8) # mIoU para la clase "nube"

    print(f"Validation Accuracy: {accuracy:.2f}%")
    print(f"Average Dice Score: {avg_dice_score:.4f}")
    print(f"Mean IoU (mIoU): {mIoU:.4f}")
    
    model.train()
    return mIoU

# =================================================================================
# 4. FUNCIÓN PRINCIPAL DE EJECUCIÓN
# =================================================================================
def main():
    print(f"Using device: {Config.DEVICE}")
    
    # --- Transformaciones y Aumento de Datos ---
    train_transform = A.Compose([
        A.Resize(height=Config.IMAGE_HEIGHT, width=Config.IMAGE_WIDTH),
        A.Rotate(limit=35, p=0.7),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(height=Config.IMAGE_HEIGHT, width=Config.IMAGE_WIDTH),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])

    # --- Creación de Datasets y DataLoaders ---
    train_dataset = CloudDataset(
        image_dir=Config.TRAIN_IMG_DIR,
        mask_dir=Config.TRAIN_MASK_DIR,
        transform=train_transform
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        shuffle=True
    )

    val_dataset = CloudDataset(
        image_dir=Config.VAL_IMG_DIR,
        mask_dir=Config.VAL_MASK_DIR,
        transform=val_transform
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        shuffle=False
    )
    
    # --- Instanciación del Modelo, Loss y Optimizador ---
    # Asumimos que la segmentación es binaria (1 clase de salida: nube)
    model = CloudDeepLabV3Plus(num_classes=1).to(Config.DEVICE)
    
    # La función de pérdida BCEWithLogitsLoss es ideal para segmentación binaria.
    # Es numéricamente más estable que usar Sigmoid + BCELoss por separado.
    loss_fn = nn.BCEWithLogitsLoss() 
    
    # AdamW es una buena elección de optimizador por defecto.
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    # El scaler es para el entrenamiento de precisión mixta (acelera el entrenamiento en GPUs compatibles)
    scaler = torch.cuda.amp.GradScaler()

    best_mIoU = -1.0

    # --- Bucle Principal de Entrenamiento ---
    for epoch in range(Config.NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{Config.NUM_EPOCHS} ---")
        train_fn(train_loader, model, optimizer, loss_fn, scaler)
        
        # Comprobar métricas en el conjunto de validación
        current_mIoU = check_metrics(val_loader, model, device=Config.DEVICE)

        # Guardar el mejor modelo basado en mIoU
        if current_mIoU > best_mIoU:
            best_mIoU = current_mIoU
            print(f"New best mIoU: {best_mIoU:.4f}! Saving model...")
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            torch.save(checkpoint, Config.MODEL_SAVE_PATH)

if __name__ == "__main__":
    main()