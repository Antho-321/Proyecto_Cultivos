# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
from PIL import Image
import numpy as np

# Importa la arquitectura del otro archivo
from model import CloudDeepLabV3Plus
from distribucion_por_clase import imprimir_distribucion_clases_post_augmentation 

# ---------------------------------------------------------------------------------
# 1. CONFIGURACIÓN  (añadimos 1 hiperparámetros nuevos)
# ---------------------------------------------------------------------------------
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
    NUM_EPOCHS = 50
    NUM_WORKERS = 2
    
    # --- Dimensiones de la imagen ---
    IMAGE_HEIGHT = 256
    IMAGE_WIDTH = 256

    CROP_P_ALWAYS     = 1.0    # fuerza el CropAroundClass4 cuando haya píxeles de clase 4
    
    # --- Configuraciones adicionales ---
    PIN_MEMORY = True
    LOAD_MODEL = False # Poner a True si quieres continuar un entrenamiento
    MODEL_SAVE_PATH = "best_model.pth.tar"

# =================================================================================
# 2. DATASET PERSONALIZADO
# Clase para cargar las imágenes y sus máscaras de segmentación.
# =================================================================================
class CloudDataset(torch.utils.data.Dataset):
    _IMG_EXTENSIONS = ('.jpg', '.png')

    def __init__(self, image_dir, mask_dir,
                 tf_with_c4: A.Compose | None = None,
                 tf_no_c4:   A.Compose | None = None,
                 transform:  A.Compose | None = None):
        if transform is not None:
            # Un solo pipeline para todo el dataset
            tf_with_c4 = tf_no_c4 = transform
        self.image_dir   = image_dir
        self.mask_dir    = mask_dir
        self.tf_with_c4  = tf_with_c4
        self.tf_no_c4    = tf_no_c4

        self.images = [f for f in os.listdir(image_dir)
                       if f.lower().endswith(self._IMG_EXTENSIONS)]

        # Pre-escaneo de máscaras
        self.contains_class4 = []
        for img_file in self.images:
            mask_path = self._mask_path_from_image_name(img_file)
            mask = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
            self.contains_class4.append((mask == 4).any())

    def __len__(self):  return len(self.images)

    def _mask_path_from_image_name(self, image_filename: str) -> str:
        name = image_filename.rsplit('.', 1)[0]
        return os.path.join(self.mask_dir, f"{name}_mask.png")

    def __getitem__(self, idx: int):
        img_path  = os.path.join(self.image_dir, self.images[idx])
        mask_path = self._mask_path_from_image_name(self.images[idx])

        image = np.array(Image.open(img_path).convert("RGB"))
        mask  = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)

        # ---------- DECISIÓN CONDICIONAL -----------------------------
        if self.contains_class4[idx]:
            aug = self.tf_with_c4(image=image, mask=mask)
        else:
            aug = self.tf_no_c4(image=image, mask=mask)
        # --------------------------------------------------------------
        image, mask = aug["image"], aug["mask"]
        return image, mask

class CropAroundClass4(A.DualTransform):
    def __init__(self, crop_size=(96,96), p=0.5):
        super().__init__(always_apply=False, p=p)
        self.ch, self.cw = crop_size

    def get_params_dependent_on_targets(self, params):
        image = params["image"]
        mask  = params["mask"]
        ys, xs = np.where(mask == 4)
        if ys.size:
            # pick one random pixel of class 4
            i = np.random.randint(ys.shape[0])
            cy, cx = ys[i], xs[i]
            y1 = int(np.clip(cy - self.ch//2, 0, image.shape[0] - self.ch))
            x1 = int(np.clip(cx - self.cw//2, 0, image.shape[1] - self.cw))
        else:
            # no class-4 pixels → random crop
            y1 = np.random.randint(0, image.shape[0] - self.ch + 1)
            x1 = np.random.randint(0, image.shape[1] - self.cw + 1)
        return {"y1": y1, "x1": x1}

    def apply(self, img, y1=0, x1=0, **params):
        return img[y1 : y1 + self.ch,
                   x1 : x1 + self.cw]

    def apply_to_mask(self, mask, y1=0, x1=0, **params):
        return mask[y1 : y1 + self.ch,
                    x1 : x1 + self.cw]

# =================================================================================
# 3. FUNCIONES DE ENTRENAMIENTO Y VALIDACIÓN
# =================================================================================
def train_fn(loader, model, optimizer, loss_fn, scaler):
    """Procesa una época de entrenamiento."""
    loop = tqdm(loader, leave=True)
    model.train()

    for batch_idx, (data, targets) in enumerate(loop):
        data     = data.to(Config.DEVICE, non_blocking=True)
        targets  = targets.to(Config.DEVICE, non_blocking=True).long()  # <- importante

        # Forward
        with autocast(device_type='cuda', dtype=torch.float16):
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # Backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Actualizar la barra de progreso
        loop.set_postfix(loss=loss.item())

def check_metrics(loader, model, n_classes=6, device="cuda"):
    """
    Devuelve mIoU macro y Dice macro.
    Imprime IoU y Dice por clase.
    """
    eps = 1e-8                          # para evitar divisiones por 0
    intersection_sum = torch.zeros(n_classes, dtype=torch.float64, device=device)
    union_sum        = torch.zeros_like(intersection_sum)
    dice_num_sum     = torch.zeros_like(intersection_sum)   # 2*intersección
    dice_den_sum     = torch.zeros_like(intersection_sum)   # pred + gt

    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).long()      # (N, H, W)

            logits = model(x)                               # (N, 6, H, W)
            preds  = torch.argmax(logits, dim=1)            # (N, H, W)

            for cls in range(n_classes):
                pred_c   = (preds == cls)
                true_c   = (y == cls)
                inter    = (pred_c & true_c).sum().double()
                pred_sum = pred_c.sum().double()
                true_sum = true_c.sum().double()
                union    = pred_sum + true_sum - inter

                intersection_sum[cls] += inter
                union_sum[cls]        += union
                dice_num_sum[cls]     += 2 * inter
                dice_den_sum[cls]     += pred_sum + true_sum

    # IoU y Dice por clase
    iou_per_class   = (intersection_sum + eps) / (union_sum + eps)
    dice_per_class  = (dice_num_sum   + eps) / (dice_den_sum + eps)

    # Promedios macro
    miou_macro  = iou_per_class.mean()
    dice_macro  = dice_per_class.mean()

    print("IoU por clase :", iou_per_class.cpu().numpy())
    print("Dice por clase:", dice_per_class.cpu().numpy())
    print(f"mIoU macro = {miou_macro:.4f} | Dice macro = {dice_macro:.4f}")

    model.train()
    return miou_macro, dice_macro

# =================================================================================
# 4. FUNCIÓN PRINCIPAL DE EJECUCIÓN
# =================================================================================
def main():
    print(f"Using device: {Config.DEVICE}")
    
    # --- 1.a  Aumentos COMPLETOS (cuando SÍ hay clase 4) --------------------------
    train_tf_with_c4 = A.Compose([
        CropAroundClass4(crop_size=(96, 96), p=Config.CROP_P_ALWAYS),
        A.Rotate(limit=35, p=0.7),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255),
        ToTensorV2(),
    ])

    # --- 1.b  Aumentos LIGEROS (cuando NO hay clase 4) ----------------------------
    train_tf_no_c4 = A.Compose([
        CropAroundClass4(crop_size=(96, 96), p=Config.CROP_P_ALWAYS),  # hará crop aleatorio
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255),
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
        image_dir = Config.TRAIN_IMG_DIR,
        mask_dir  = Config.TRAIN_MASK_DIR,
        tf_with_c4 = train_tf_with_c4,
        tf_no_c4   = train_tf_no_c4,
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

    imprimir_distribucion_clases_post_augmentation(
        train_loader, 
        n_classes=6, 
        title="Distribución de clases en el set de ENTRENAMIENTO (post-augmentation)"
    )
    
    # --- Instanciación del Modelo, Loss y Optimizador ---
    model = CloudDeepLabV3Plus(num_classes=6).to(Config.DEVICE)
    
    loss_fn = nn.CrossEntropyLoss()
    
    # AdamW es una buena elección de optimizador por defecto.
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    # El scaler es para el entrenamiento de precisión mixta (acelera el entrenamiento en GPUs compatibles)
    scaler = GradScaler()

    best_mIoU = -1.0

    # --- Bucle Principal de Entrenamiento ---
    for epoch in range(Config.NUM_EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{Config.NUM_EPOCHS} ---")

        # 1) Entrenamiento de una época
        train_fn(
            train_loader,
            model,
            optimizer,
            loss_fn,
            scaler
        )

        # 2) Evaluación en el conjunto de validación
        current_mIoU, current_dice = check_metrics(
            val_loader,
            model,
            n_classes=6,
            device=Config.DEVICE
        )

        # 3) Guardar checkpoint si hubo mejora en mIoU
        if current_mIoU > best_mIoU:
            best_mIoU = current_mIoU
            print(f"🔹 Nuevo mejor mIoU: {best_mIoU:.4f} | Dice: {current_dice:.4f}  →  guardando modelo…")

            checkpoint = {
                "epoch":     epoch,
                "state_dict": model.state_dict(),
                "optimizer":  optimizer.state_dict(),
                "best_mIoU":  best_mIoU,
            }
            torch.save(checkpoint, Config.MODEL_SAVE_PATH)

    print("\nEvaluando el modelo con mejor mIoU guardado…")
    best_mIoU, best_dice = check_metrics(
        val_loader,
        model,
        n_classes=6,
        device=Config.DEVICE
    )
    print(f"mIoU del modelo guardado: {best_mIoU:.4f} | Dice: {best_dice:.4f}")

if __name__ == "__main__":
    main()