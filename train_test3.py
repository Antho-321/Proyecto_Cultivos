# train.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
from PIL import Image
import numpy as np
# Importa la arquitectura del otro archivo
from model import CloudDeepLabV3Plus
from utils import imprimir_distribucion_clases_post_augmentation, save_performance_plot, crop_around_classes
from config import Config
import cv2

# =================================================================================
# 2. DATASET PERSONALIZADO (MODIFICADO)
# =================================================================================
def mascara_contiene_solo_0_y_4(mask: np.ndarray) -> bool: # <-- Ponemos la función aquí para que sea accesible
    """
    Verifica si una máscara contiene únicamente píxeles con valor 0 y 4.
    """
    return set(np.unique(mask)) == {0, 4}

class CloudDataset(torch.utils.data.Dataset):
    _IMG_EXTENSIONS = ('.jpg', '.png')

    # <-- MODIFICADO: __init__ ahora acepta un pipeline de transformación especial
    def __init__(self, image_dir: str, mask_dir: str, transform: A.Compose | None = None, special_transform: A.Compose | None = None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.special_transform = special_transform  # <-- NUEVO: Guardar el pipeline de aumentación especial
        
        self.samples = []  # <-- NUEVO: Esta será nuestra lista principal de muestras

        print("Inicializando dataset y buscando imágenes para aumentación especial...")
        all_images = [
            f for f in os.listdir(image_dir)
            if f.lower().endswith(self._IMG_EXTENSIONS)
        ]

        # <-- NUEVO: Lógica para construir la lista de muestras expandida
        for img_filename in tqdm(all_images, desc="Procesando imágenes iniciales"):
            # Siempre añadimos la muestra original (usará la transformación normal)
            self.samples.append({"image_filename": img_filename, "apply_special_aug": False})

            # Si se proporciona una transformación especial, verificamos la condición
            if self.special_transform:
                mask_path = self._mask_path_from_image_name(img_filename)
                if os.path.exists(mask_path):
                    mask = np.array(Image.open(mask_path).convert("L"))
                    
                    # Si la máscara cumple la condición, añadimos N copias "virtuales"
                    if mascara_contiene_solo_0_y_4(mask):
                        for _ in range(Config.NUM_SPECIAL_AUGMENTATIONS):
                            self.samples.append({"image_filename": img_filename, "apply_special_aug": True})
        
        print(f"Dataset inicializado. Tamaño original: {len(all_images)}, Tamaño con aumentación: {len(self.samples)}")

    def __len__(self) -> int:
        # <-- MODIFICADO: La longitud ahora es el tamaño de nuestra lista de muestras expandida
        return len(self.samples)

    def _mask_path_from_image_name(self, image_filename: str) -> str:
        name_without_ext = image_filename.rsplit('.', 1)[0]
        mask_filename = f"{name_without_ext}_mask.png"
        return os.path.join(self.mask_dir, mask_filename)

    def __getitem__(self, idx: int):
        # <-- MODIFICADO: Obtenemos la información de la muestra de nuestra nueva lista
        sample_info = self.samples[idx]
        img_filename = sample_info["image_filename"]
        apply_special_aug = sample_info["apply_special_aug"]

        img_path = os.path.join(self.image_dir, img_filename)
        mask_path = self._mask_path_from_image_name(img_filename)
        
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Máscara no encontrada para {img_filename} en {mask_path}")

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))

        # Tu lógica de recorte permanece igual
        mask_3d = np.expand_dims(mask, axis=-1)
        image_cropped, mask_cropped_3d = crop_around_classes(image, mask_3d)
        mask_cropped = mask_cropped_3d.squeeze()
        
        # <-- MODIFICADO: Aplicamos la transformación condicionalmente
        if apply_special_aug and self.special_transform:
            # Esta es una muestra que cumplió la condición y debe ser aumentada especialmente
            augmented = self.special_transform(image=image_cropped, mask=mask_cropped)
        elif self.transform:
            # Esta es una muestra normal
            augmented = self.transform(image=image_cropped, mask=mask_cropped)
        else:
            # Si no hay transformaciones
            augmented = {"image": image_cropped, "mask": mask_cropped}

        image = augmented["image"]
        mask = augmented["mask"]

        return image, mask

# =================================================================================
# 3. FUNCIONES DE ENTRENAMIENTO Y VALIDACIÓN (Sin cambios)
# ... (El resto de tu código: train_fn, check_metrics)
# =================================================================================
def train_fn(loader, model, optimizer, loss_fn, scaler):
    """Procesa una época de entrenamiento."""
    loop = tqdm(loader, leave=True)
    model.train()

    for batch_idx, (data, targets) in enumerate(loop):
        data     = data.to(Config.DEVICE, non_blocking=True)
        targets  = targets.to(Config.DEVICE, non_blocking=True).long()

        with torch.cuda.amp.autocast():
            # Desempaquetamos o seleccionamos la salida principal
            output = model(data)
            predictions = output[0] if isinstance(output, tuple) else output
            loss = loss_fn(predictions, targets)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss=loss.item())

def check_metrics(loader, model, n_classes=6, device="cuda"):
    """
    Calcula mIoU macro y Dice macro usando operaciones vectorizadas en la GPU.
    Esta versión es significativamente más rápida que usar un bucle for sobre las clases.
    """
    eps = 1e-8
    # Tensores para acumular los valores a lo largo de todos los lotes
    total_intersection = torch.zeros(n_classes, dtype=torch.float64, device=device)
    total_union = torch.zeros(n_classes, dtype=torch.float64, device=device)
    total_dice_num = torch.zeros(n_classes, dtype=torch.float64, device=device)
    total_dice_den = torch.zeros(n_classes, dtype=torch.float64, device=device)

    model.eval()  # Poner el modelo en modo de evaluación

    with torch.no_grad():  # Desactivar el cálculo de gradientes para la inferencia
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).long()  # Shape: (B, H, W)

            # Inferencia del modelo
            output = model(x)
            # Asegurarse de obtener los logits, incluso si el modelo devuelve una tupla
            logits = output[0] if isinstance(output, tuple) else output
            preds = torch.argmax(logits, dim=1)  # Shape: (B, H, W)

            # --- INICIO DEL CÁLCULO VECTORIZADO ---

            # 1. Convertir a formato One-Hot
            # Convierte las máscaras (y, preds) de (B, H, W) a (B, H, W, C)
            y_one_hot = F.one_hot(y, num_classes=n_classes)
            preds_one_hot = F.one_hot(preds, num_classes=n_classes)
            
            # 2. Reordenar dimensiones para el cálculo
            # Cambia el formato a (B, C, H, W) para que las operaciones por canal (clase) sean fáciles
            y_one_hot = y_one_hot.permute(0, 3, 1, 2)
            preds_one_hot = preds_one_hot.permute(0, 3, 1, 2)

            # 3. Calcular intersección y unión para todas las clases a la vez
            # Multiplicar los tensores one-hot nos da la intersección
            intersection = (preds_one_hot * y_one_hot).sum(dim=(0, 2, 3)).double()
            
            # La suma de los píxeles de cada clase por separado
            sum_preds = preds_one_hot.sum(dim=(0, 2, 3)).double()
            sum_true = y_one_hot.sum(dim=(0, 2, 3)).double()
            
            # Fórmula de la unión: A + B - Intersección
            union = sum_preds + sum_true - intersection

            # 4. Acumular los resultados del lote actual
            total_intersection += intersection
            total_union += union
            total_dice_num += 2 * intersection
            total_dice_den += sum_preds + sum_true

            # --- FIN DEL CÁLCULO VECTORIZADO ---

    # Calcular métricas finales después de procesar todos los lotes
    iou_per_class = (total_intersection + eps) / (total_union + eps)
    dice_per_class = (total_dice_num + eps) / (total_dice_den + eps)

    miou_macro = iou_per_class.mean()
    dice_macro = dice_per_class.mean()

    # Imprimir resultados
    print("IoU por clase :", iou_per_class.cpu().numpy())
    print("Dice por clase:", dice_per_class.cpu().numpy())
    print(f"mIoU macro = {miou_macro:.4f} | Dice macro = {dice_macro:.4f}")

    model.train()  # Restaurar el modelo a modo de entrenamiento
    return miou_macro, dice_macro

# =================================================================================
# 4. FUNCIÓN PRINCIPAL DE EJECUCIÓN (Sin cambios)
# =================================================================================
def main():
    print(f"Using device: {Config.DEVICE}")
    
    # Transformaciones normales para todas las imágenes
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

    # <-- NUEVO: Pipeline de transformaciones especiales para las clases {0, 4}
    # Incluye las transformaciones que pediste: rotaciones, traslaciones, etc.
    special_aug_transform = A.Compose([
        # **1. Redimensionamiento Inicial (Generalmente el primer paso)**
        A.Resize(height=Config.IMAGE_HEIGHT, width=Config.IMAGE_WIDTH, interpolation=cv2.INTER_LINEAR),

        # **2. Transformaciones Geométricas**
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Rotate(limit=45, p=0.7, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=0, p=0.7, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.Affine(scale=(0.9, 1.1), translate_percent=(-0.1, 0.1), rotate=(-10, 10), shear=(-10, 10), p=0.5),
        A.Perspective(scale=(0.05, 0.1), keep_size=True, p=0.3),
        A.ElasticTransform(p=0.3, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
        A.GridDistortion(p=0.3),
        A.OpticalDistortion(p=0.3, distort_limit=2, shift_limit=0.5),

        # **3. Transformaciones de Color y Brillo**
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        A.ToGray(p=0.1),
        A.ToSepia(p=0.1),
        A.InvertImg(p=0.1),
        A.Solarize(p=0.1),
        A.Equalize(p=0.2),
        A.ChannelShuffle(p=0.2),

        # **4. Transformaciones de Desenfoque (Blur)**
        A.OneOf([
            A.Blur(blur_limit=7, p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), p=0.5),
            A.MedianBlur(blur_limit=5, p=0.5),
            A.MotionBlur(blur_limit=(3, 7), p=0.5),
        ], p=0.4),

        # **5. Adición de Ruido**
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.5),
            A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=0.5),
        ], p=0.4),

        # **6. Transformaciones de Clima (Simulación)**
        A.RandomFog(fog_coef_lower=0.3, fog_coef_upper=0.5, alpha_coef=0.1, p=0.1),
        A.RandomRain(p=0.1),
        A.RandomSnow(p=0.1),
        A.RandomSunFlare(p=0.1),
        A.RandomShadow(p=0.1),

        # **7. Recorte y Eliminación (Cutout/Dropout)**
        A.CoarseDropout(max_holes=8, max_height=8, max_width=8, min_holes=1, min_height=1, min_width=1, fill_value=0, p=0.3),
        A.RandomResizedCrop(size=(Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH), scale=(0.8, 1.0), p=0.3),
        
        # **8. Otras Transformaciones de Píxeles y Estilo**
        A.Downscale(scale_min=0.25, scale_max=0.25, p=0.1),
        A.Emboss(p=0.2),
        A.Sharpen(p=0.2),
        A.Posterize(p=0.1),
        A.FancyPCA(alpha=0.1, p=0.2),

        # **9. Normalización y Conversión a Tensor (Generalmente al final)**
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

    # <-- MODIFICADO: Pasamos ambos pipelines de transformación al dataset de entrenamiento
    train_dataset = CloudDataset(
        image_dir=Config.TRAIN_IMG_DIR,
        mask_dir=Config.TRAIN_MASK_DIR,
        transform=train_transform,
        special_transform=special_aug_transform  # <-- Pasamos el nuevo pipeline
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        shuffle=True  # Shuffle es clave para mezclar las muestras originales y las aumentadas
    )

    # El dataset de validación no se expande para tener una métrica consistente
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

    imprimir_distribucion_clases_post_augmentation(train_loader, 6,
        "Distribución de clases en ENTRENAMIENTO (post-aug)")

    model = CloudDeepLabV3Plus(num_classes=6).to(Config.DEVICE)
    model = torch.compile(model)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scaler = GradScaler() 
    best_mIoU = -1.0

    train_miou_history = []
    val_miou_history = []

    for epoch in range(Config.NUM_EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{Config.NUM_EPOCHS} ---")
        train_fn(train_loader, model, optimizer, loss_fn, scaler)
        
        print("Calculando métricas de entrenamiento...")
        train_mIoU, _ = check_metrics(train_loader, model, n_classes=6, device=Config.DEVICE)
        
        print("Calculando métricas de validación...")
        current_mIoU, current_dice = check_metrics(val_loader, model, n_classes=6, device=Config.DEVICE)

        train_miou_history.append(train_mIoU.item())
        val_miou_history.append(current_mIoU.item())

        if current_mIoU > best_mIoU:
            best_mIoU = current_mIoU
            print(f"🔹 Nuevo mejor mIoU: {best_mIoU:.4f} | Dice: {current_dice:.4f}  →  guardando modelo…")
            checkpoint = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_mIoU": best_mIoU,
            }
            torch.save(checkpoint, Config.MODEL_SAVE_PATH)

    save_performance_plot(
        train_history=train_miou_history,
        val_history=val_miou_history,
        save_path=Config.PERFORMANCE_PATH
    )

    print("\nEvaluando el modelo con mejor mIoU guardado…")
    best_model_checkpoint = torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE)
    model.load_state_dict(best_model_checkpoint['state_dict'])
    best_mIoU, best_dice = check_metrics(val_loader, model, n_classes=6, device=Config.DEVICE)
    print(f"mIoU del modelo guardado: {best_mIoU:.4f} | Dice: {best_dice:.4f}")

if __name__ == "__main__":
    main()