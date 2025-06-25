
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
from utils import imprimir_distribucion_clases_post_augmentation, crop_around_classes, save_performance_plot
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

    def __init__(self, image_dir: str, mask_dir: str, transform: A.Compose | None = None, special_transform: A.Compose | None = None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.special_transform = special_transform  # <-- Pasar como argumento
        
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
        return len(self.images)

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

        # --- MODIFICACIÓN CLAVE: Aplicar recorte ANTES de las transformaciones ---
        # 1. Añadir una dimensión de canal a la máscara para que sea (H, W, 1)
        mask_3d = np.expand_dims(mask, axis=-1)
        
        # 2. Aplicar la función de recorte
        image_cropped, mask_cropped_3d = crop_around_classes(image, mask_3d)

        # 3. Quitar la dimensión del canal de la máscara para Albumentations
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
def train_fn(loader, model, optimizer, loss_fn, scaler, num_classes=6):
    """Procesa una época de entrenamiento con cálculo de IoU por clase."""
    loop = tqdm(loader, leave=True)
    model.train()

    # Inicializamos los contadores para cada clase
    tp = torch.zeros(num_classes).to(Config.DEVICE)
    fp = torch.zeros(num_classes).to(Config.DEVICE)
    fn = torch.zeros(num_classes).to(Config.DEVICE)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(Config.DEVICE, non_blocking=True)
        targets = targets.to(Config.DEVICE, non_blocking=True).long()

        with autocast(device_type=Config.DEVICE, dtype=torch.float16):
            # Desempaquetamos o seleccionamos la salida principal
            output = model(data)
            predictions = output[0] if isinstance(output, tuple) else output
            loss = loss_fn(predictions, targets)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Convertir las predicciones a etiquetas de clase
        _, predicted_classes = torch.max(predictions, dim=1)

        # Para cada clase, contar TP, FP y FN
        for c in range(num_classes):
            true_positives = (predicted_classes == c) & (targets == c)
            false_positives = (predicted_classes == c) & (targets != c)
            false_negatives = (predicted_classes != c) & (targets == c)

            tp[c] += true_positives.sum().item()
            fp[c] += false_positives.sum().item()
            fn[c] += false_negatives.sum().item()

        # Actualizar el loop con la pérdida
        loop.set_postfix(loss=loss.item())

    # Calcular el IoU para cada clase
    iou = tp / (tp + fp + fn)
    
    return iou

def check_metrics(loader, model, n_classes=6, device="cuda"):
    """Devuelve mIoU macro y Dice macro."""
    eps = 1e-8
    intersection_sum = torch.zeros(n_classes, dtype=torch.float64, device=device)
    union_sum        = torch.zeros_like(intersection_sum)
    dice_num_sum     = torch.zeros_like(intersection_sum)
    dice_den_sum     = torch.zeros_like(intersection_sum)

    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).long()

            output = model(x)
            logits = output[0] if isinstance(output, tuple) else output # <-- ¡ESTA ES LA CORRECCIÓN CLAVE!
            preds  = torch.argmax(logits, dim=1)

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

    iou_per_class  = (intersection_sum + eps) / (union_sum + eps)
    dice_per_class = (dice_num_sum   + eps) / (dice_den_sum + eps)

    miou_macro = iou_per_class.mean()
    dice_macro = dice_per_class.mean()

    print("IoU por clase :", iou_per_class.cpu().numpy())
    print("Dice por clase:", dice_per_class.cpu().numpy())
    print(f"mIoU macro = {miou_macro:.4f} | Dice macro = {dice_macro:.4f}")

    model.train()
    return miou_macro, dice_macro

# =================================================================================
# 4. FUNCIÓN PRINCIPAL DE EJECUCIÓN (Sin cambios)
# =================================================================================
def main():
    torch.backends.cudnn.benchmark = True

    print(f"Using device: {Config.DEVICE}")
    
    train_transform = A.Compose([
        A.Resize(height=Config.IMAGE_HEIGHT, width=Config.IMAGE_WIDTH), # <-- MUY IMPORTANTE
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
        A.Resize(height=Config.IMAGE_HEIGHT, width=Config.IMAGE_WIDTH,
                interpolation=cv2.INTER_LINEAR),

        A.SomeOf([
            # Geometric
            A.HorizontalFlip(p=1),
            A.VerticalFlip(p=1),
            A.RandomRotate90(p=1),
            A.Rotate(limit=45, border_mode=cv2.BORDER_CONSTANT, fill=0, p=1),
            A.Affine(scale=(0.9,1.1), translate_percent=(-.1,.1),
                    rotate=(-10,10), shear=(-10,10), fill=0, p=1),
            A.Perspective(scale=(.05,.1), keep_size=True, fill=0, p=1),
            A.ElasticTransform(alpha=120, sigma=120*0.05, affine_alpha=120*0.03,
                            border_mode=cv2.BORDER_CONSTANT, fill=0, p=1),
            A.GridDistortion(border_mode=cv2.BORDER_CONSTANT, fill=0, p=1),
            A.OpticalDistortion(distort_limit=.5, shift_limit_x=.5, shift_limit_y=.5,
                                border_mode=cv2.BORDER_CONSTANT, fill=0, p=1),
            A.RandomResizedCrop(size=(Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH),
                                scale=(0.8,1.0), p=1),

            # Color / brightness
            A.RandomBrightnessContrast(.2, .2, p=1),
            A.ColorJitter(.2, .2, .2, .2, p=1),
            A.HueSaturationValue(20, 30, 20, p=1),
            A.RGBShift(20, 20, 20, p=1),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8,8), p=1),
            A.RandomGamma((80,120), p=1),
            A.ToGray(p=1), A.ToSepia(p=1), A.InvertImg(p=1),
            A.Solarize(p=1), A.Equalize(p=1), A.ChannelShuffle(p=1),

            # Blur
            A.Blur(blur_limit=7, p=1),
            A.GaussianBlur(blur_limit=(3,7), p=1),
            A.MedianBlur(blur_limit=5, p=1),
            A.MotionBlur(blur_limit=(3,7), p=1),

            # Noise  (⇣ aquí el cambio)
            A.GaussNoise(std_range=(10/255.0, 50/255.0), p=1),
            A.ISONoise(color_shift=(0.01,0.05), intensity=(0.1,0.5), p=1),
            A.MultiplicativeNoise(multiplier=(0.9,1.1), p=1),

            # Dropout
            A.CoarseDropout(
                num_holes_range=(1, 8),        # 1-8 agujeros por imagen
                hole_height_range=(1, 8),      # altura de cada hueco en píxeles
                hole_width_range=(1, 8),       # anchura de cada hueco en píxeles
                fill=0,                        # píxeles negros en los huecos
                p=1
            ),

            # Weather
            A.RandomFog(fog_coef_range=(0.3,0.5), alpha_coef=0.1, p=1),
            A.RandomRain(p=1), A.RandomSnow(p=1),
            A.RandomSunFlare(p=1), A.RandomShadow(p=1),

            # Other
            A.Downscale(scale_range=(0.25,0.25), p=1),
            A.Emboss(p=1), A.Sharpen(p=1), A.Posterize(p=1), A.FancyPCA(alpha=.1, p=1),
        ], n=Config.NUM_SPECIAL_AUGMENTATIONS, p=1.0),

        A.Normalize(mean=[0,0,0], std=[1,1,1], max_pixel_value=255.0),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(height=Config.IMAGE_HEIGHT, width=Config.IMAGE_WIDTH), # <-- MUY IMPORTANTE
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])

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

    imprimir_distribucion_clases_post_augmentation(train_loader, 6,
        "Distribución de clases en ENTRENAMIENTO (post-aug)")

    model = CloudDeepLabV3Plus(num_classes=6).to(Config.DEVICE, memory_format=torch.channels_last)
    model = torch.compile(model, mode="reduce-overhead")   # or "max-autotune" when you’re done debugging
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(),
                        lr=Config.LEARNING_RATE,
                        fused=True)        # single-kernel update on GPU
    scaler = GradScaler(enabled=Config.DEVICE == "cuda")
    best_mIoU = -1.0

    # --- 2. INICIALIZAR LISTAS PARA EL HISTORIAL ---
    train_miou_history = []
    val_miou_history = []

    for epoch in range(Config.NUM_EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{Config.NUM_EPOCHS} ---")
        
        # --- 3. CALCULAR MÉTRICAS PARA ENTRENAMIENTO Y VALIDACIÓN ---
        print("Calculando métricas de entrenamiento...")
        train_mIoU = train_fn(train_loader, model, optimizer, loss_fn, scaler)
        print(f"mIoU por clase en entrenamiento: {train_mIoU.cpu().numpy()}")
        #train_mIoU, _ = check_metrics(train_loader, model, n_classes=6, device=Config.DEVICE)
        
        print("Calculando métricas de validación...")
        current_mIoU, current_dice = check_metrics(val_loader, model, n_classes=6, device=Config.DEVICE)

        # --- 4. GUARDAR LAS MÉTRICAS EN EL HISTORIAL ---
        train_miou_history.append(train_mIoU.item()) # .item() para obtener el valor escalar
        val_miou_history.append(current_mIoU.item())

        if current_mIoU > best_mIoU:
            best_mIoU = current_mIoU
            print(f"🔹 Nuevo mejor mIoU: {best_mIoU:.4f} | Dice: {current_dice:.4f}  →  guardando modelo…")
            checkpoint = {
                "epoch":      epoch,
                "state_dict": model.state_dict(),
                "optimizer":  optimizer.state_dict(),
                "best_mIoU":  best_mIoU,
            }
            torch.save(checkpoint, Config.MODEL_SAVE_PATH)

    # --- 5. LLAMAR A LA FUNCIÓN DE GRAFICADO AL FINALIZAR ---
    save_performance_plot(
        train_history=train_miou_history,
        val_history=val_miou_history,
        save_path=Config.PERFORMANCE_PATH
    )

    print("\nEvaluando el modelo con mejor mIoU guardado…")

    # --- Cargar el checkpoint del mejor modelo ---
    # Añadir map_location para asegurar compatibilidad entre CPU/GPU
    best_model_checkpoint = torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE)
    model.load_state_dict(best_model_checkpoint['state_dict'])

    # Ahora que el mejor modelo está cargado, se ejecuta la evaluación
    best_mIoU, best_dice = check_metrics(val_loader, model, n_classes=6, device=Config.DEVICE)
    print(f"mIoU del modelo guardado: {best_mIoU:.4f} | Dice: {best_dice:.4f}")
if __name__ == "__main__":
    main()
