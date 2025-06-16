
# train.py

import torch
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
from PIL import Image
import numpy as np

# Importa la arquitectura del otro archivo
from model import CloudDeepLabV3Plus
from distribucion_por_clase import imprimir_distribucion_clases_post_augmentation
from loss_function import FocalTverskyLoss
import math

# ---------------------------------------------------------------------------------
# 1. CONFIGURACIÓN  (añadimos dos hiperparámetros nuevos)
# ---------------------------------------------------------------------------------
class Config:
    DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"

    TRAIN_IMG_DIR     = "Balanced/train/images"
    TRAIN_MASK_DIR    = "Balanced/train/masks"
    VAL_IMG_DIR       = "Balanced/val/images"
    VAL_MASK_DIR      = "Balanced/val/masks"

    LEARNING_RATE     = 1e-4
    BATCH_SIZE        = 8
    NUM_EPOCHS        = 25
    NUM_WORKERS       = 2

    IMAGE_HEIGHT      = 256
    IMAGE_WIDTH       = 256

    # —— NUEVO ——
    CLASS4_WEIGHT     = 6      # cuántas veces “vale” una imagen con clase 4
    CROP_P_ALWAYS     = 1.0    # fuerza el CropAroundClass4 cuando haya píxeles de clase 4

    PIN_MEMORY        = True
    LOAD_MODEL        = False
    MODEL_SAVE_PATH   = "best_model.pth.tar"

# ---------------------------------------------------------------------------------
# 2. DATASET PERSONALIZADO con flag ‘contains_class4’
# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Dataset que genera TODOS los patches 128×128 de cada imagen y
# guarda qué clases contiene cada patch.  Esto permite muestrear después
# con conocimiento de clase.
# ---------------------------------------------------------------------------

class CloudPatchDataset(torch.utils.data.Dataset):
    """
    Devuelve tuplas (patch_RGB, patch_mask) de tamaño fijo 128×128.
    Durante la construcción pre-indexa *todos* los patches para saber
    si contienen la clase 4 (y/o cualquier otra clase que te interese).
    """
    _IMG_EXTENSIONS = ('.jpg', '.png')

    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        patch_size: int = 128,
        stride: int = 128,
        min_class4_pixels: int = 32,
        transform: A.Compose | None = None,
    ):
        self.image_dir   = image_dir
        self.mask_dir    = mask_dir
        self.patch_size  = patch_size
        self.stride      = stride
        self.transform   = transform

        # Lista de nombres de archivo
        self.images = [
            f for f in os.listdir(image_dir)
            if f.lower().endswith(self._IMG_EXTENSIONS)
        ]

        # ------------------------------------------------------------------
        # Construimos un índice: cada entrada es un dict con:
        #   • img_idx ........ índice en self.images
        #   • y, x  .......... coordenadas de esquina superior izquierda
        #   • has_c4 ......... bool  → contiene ≥ min_class4_pixels de clase 4
        # ------------------------------------------------------------------
        self.index = []
        for img_idx, img_name in enumerate(self.images):
            # Cargamos solo la máscara (más rápido)
            mask_path = self._mask_path_from_image_name(img_name)
            mask_full = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)

            H, W = mask_full.shape
            for y in range(0, H - patch_size + 1, stride):
                for x in range(0, W - patch_size + 1, stride):
                    patch_mask = mask_full[y:y + patch_size, x:x + patch_size]

                    # Ignoramos patches 100 % fondo
                    if (patch_mask != 0).sum() == 0:
                        continue

                    has_c4 = (patch_mask == 4).sum() >= min_class4_pixels
                    self.index.append(
                        dict(img_idx=img_idx, y=y, x=x, has_c4=has_c4)
                    )

        # Pre-cálculo para el sampler
        self.idxs_c4     = [i for i, d in enumerate(self.index) if d["has_c4"]]
        self.idxs_others = [i for i, d in enumerate(self.index) if not d["has_c4"]]

    # -------------------------------- utility ------------------------------
    def _mask_path_from_image_name(self, image_filename: str) -> str:
        name = image_filename.rsplit('.', 1)[0]
        return os.path.join(self.mask_dir, f"{name}_mask.png")

    def __len__(self) -> int:
        return len(self.index)

    # -------------------------------- main ---------------------------------
    def __getitem__(self, idx: int):
        rec       = self.index[idx]
        img_name  = self.images[rec["img_idx"]]

        # Cargamos recorte
        img_path  = os.path.join(self.image_dir, img_name)
        mask_path = self._mask_path_from_image_name(img_name)

        image_full = np.array(Image.open(img_path).convert("RGB"))
        mask_full  = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)

        y, x = rec["y"], rec["x"]
        image = image_full[y:y + self.patch_size, x:x + self.patch_size]
        mask  = mask_full[y:y + self.patch_size, x:x + self.patch_size]

        if self.transform:
            aug   = self.transform(image=image, mask=mask)
            image = aug["image"]
            mask  = aug["mask"]

        return image, mask

# ---------------------------------------------------------------------------
# Sampler que, al formar cada batch, toma exactamente
#   • n_c4   = batch_size // 2   patches con clase 4
#   • n_rest = batch_size - n_c4  patches sin clase 4
# ---------------------------------------------------------------------------

class BalancedPatchSampler(torch.utils.data.Sampler):
    def __init__(self, dataset: CloudPatchDataset, batch_size: int, drop_last: bool = True):
        self.dataset     = dataset
        self.batch_size  = batch_size
        self.drop_last   = drop_last

        self.n_c4   = batch_size // 2
        self.n_rest = batch_size - self.n_c4

    def __iter__(self):
        # Barajamos por separado cada época
        idxs_c4     = np.random.permutation(self.dataset.idxs_c4).tolist()
        idxs_others = np.random.permutation(self.dataset.idxs_others).tolist()

        # Cuántos batches se pueden formar
        n_batches = min(len(idxs_c4) // self.n_c4,
                        len(idxs_others) // self.n_rest)

        for b in range(n_batches):
            start_c4   = b * self.n_c4
            start_rest = b * self.n_rest

            batch = (
                idxs_c4[start_c4 : start_c4 + self.n_c4] +
                idxs_others[start_rest : start_rest + self.n_rest]
            )
            np.random.shuffle(batch)      # mezcla interna
            yield batch

        # Si no quieres perder los últimos ejemplos, quita el if
        if not self.drop_last:
            remaining = idxs_c4[n_batches * self.n_c4 :] + \
                        idxs_others[n_batches * self.n_rest :]
            while len(remaining) >= self.batch_size:
                batch  = remaining[:self.batch_size]
                del remaining[:self.batch_size]
                np.random.shuffle(batch)
                yield batch

    def __len__(self):
        if self.drop_last:
            return min(
                len(self.dataset.idxs_c4)     // self.n_c4,
                len(self.dataset.idxs_others) // self.n_rest
            )
        else:
            total = len(self.dataset.idxs_c4) + len(self.dataset.idxs_others)
            return math.ceil(total / self.batch_size)

class CropAroundClass4(A.DualTransform):
    def __init__(self, crop_size=(256,256), p=0.5):
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

class CropWithoutBackground(A.DualTransform):
    """
    Recorta un patch que NO contenga píxeles de clase 0 (background).
    • Se activa con probabilidad p (por defecto 0.7).  
    • Si la máscara ya carece de fondo, simplemente pasa de largo.  
    • Intentará hasta 10 veces encontrar un recorte válido; si no lo
      consigue, hace un recorte aleatorio normal.
    """
    def __init__(self, crop_size=(96, 96), p: float = 0.7):
        super().__init__(always_apply=False, p=p)
        self.ch, self.cw = crop_size

    # ------------------------------------------------------------------
    def get_params_dependent_on_targets(self, params):
        image, mask = params["image"], params["mask"]
        h, w        = mask.shape[:2]

        # -- Si la imagen NO contiene fondo ⇒ no hacemos nada especial
        if not (mask == 0).any():
            return {"y1": 0, "x1": 0, "skip": True}

        # -- Hasta 10 intentos para encontrar un recorte sin fondo
        ys_fg, xs_fg = np.where(mask != 0)
        for _ in range(10):
            i          = np.random.randint(len(ys_fg))
            cy, cx     = int(ys_fg[i]), int(xs_fg[i])
            y1         = int(np.clip(cy - self.ch // 2, 0, h - self.ch))
            x1         = int(np.clip(cx - self.cw // 2, 0, w - self.cw))
            crop_mask  = mask[y1:y1 + self.ch, x1:x1 + self.cw]
            if (crop_mask == 0).sum() == 0:          # ¡Sin fondo!
                return {"y1": y1, "x1": x1, "skip": False}

        # Fall-back: recorte aleatorio
        y1 = np.random.randint(0, h - self.ch + 1)
        x1 = np.random.randint(0, w - self.cw + 1)
        return {"y1": y1, "x1": x1, "skip": False}

    # ------------------------------------------------------------------
    def apply(self, img, y1=0, x1=0, skip=False, **params):
        return img if skip else img[y1:y1 + self.ch, x1:x1 + self.cw]

    def apply_to_mask(self, mask, y1=0, x1=0, skip=False, **params):
        return mask if skip else mask[y1:y1 + self.ch, x1:x1 + self.cw]

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

# ------------------------------------------------------------
# Cálculo de pesos por clase  ❚  inverse-frequency  /  MFB
# ------------------------------------------------------------
def compute_class_weights(
    dataset,
    n_classes: int = 6,
    method: str = "median"   # "inverse"  ▸ frecuencia inversa sin normalizar
):
    """
    Recorre el dataset (patches o imágenes) una sola vez y devuelve
    un array NumPy con el peso de cada clase.

    • method == "inverse"  →  w_c =  total_pixels / pixels_c
    • method == "median"   →  “Median-Frequency Balancing” (MFB):
        w_c = median(freq) / freq_c,
        donde  freq_c = pixels_c / pixels_totales_en_imágenes_que_contienen_c
    """
    pixel_counts        = np.zeros(n_classes, dtype=np.int64)
    image_pixel_totals  = np.zeros(n_classes, dtype=np.int64)

    for _, mask in dataset:                      # mask: Tensor o np.ndarray (H×W)
        m = np.array(mask)                       # aseguramos NumPy
        total_px = m.size

        for c in range(n_classes):
            px_c            = (m == c).sum()
            pixel_counts[c] += px_c
            if px_c > 0:                         # la imagen/patch contiene la clase
                image_pixel_totals[c] += total_px

    if method == "inverse":
        weights = pixel_counts.sum() / (pixel_counts + 1e-8)

    elif method == "median":
        freq_per_class = pixel_counts / (image_pixel_totals + 1e-8)
        median_freq    = np.median(freq_per_class[freq_per_class > 0])
        weights        = median_freq / (freq_per_class + 1e-8)

    else:
        raise ValueError('method debe ser "inverse" o "median"')

    # (Opcional) Ignorar por completo el fondo en CE/Focal
    weights[0] = 0.0

    return weights

# =================================================================================
# 4. FUNCIÓN PRINCIPAL DE EJECUCIÓN
# =================================================================================
def main():
    print(f"Using device: {Config.DEVICE}")
    
    # --- Transformaciones y Aumento de Datos ---
    train_transform = A.Compose([
        CropAroundClass4(crop_size=(96, 96), p=Config.CROP_P_ALWAYS),
        CropWithoutBackground(
            crop_size=(Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH),  # igual al tamaño final
            p=0.7                                                 # ← 70 %
        ),
        A.Rotate(limit=35, p=0.7),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255),
        ToTensorV2()
    ])

    val_transform = A.Compose([
        A.Resize(height=Config.IMAGE_HEIGHT, width=Config.IMAGE_WIDTH),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255),
        ToTensorV2()
    ])

    # --- Creación de Datasets y DataLoaders ---
    train_dataset = CloudPatchDataset(
        Config.TRAIN_IMG_DIR,
        Config.TRAIN_MASK_DIR,
        patch_size = 128,
        stride     = 128,     # 100 % cobertura, sin solaparse
        transform  = train_transform
    )

    # --- Ponderaciones: alto peso si la imagen tiene clase 4, 1 en caso contrario ---
    weights = [
        Config.CLASS4_WEIGHT if has_c4 else 1
        for has_c4 in train_dataset.contains_class4
    ]

    # Sampler que garantiza 50 % de clase 4 por batch
    train_sampler = BalancedPatchSampler(
        dataset     = train_dataset,
        batch_size  = Config.BATCH_SIZE,
        drop_last   = True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler = train_sampler,   # ¡ojo! → reemplaza batch_size & shuffle
        num_workers   = Config.NUM_WORKERS,
        pin_memory    = Config.PIN_MEMORY
    )

    val_dataset = CloudPatchDataset(
        Config.VAL_IMG_DIR,
        Config.VAL_MASK_DIR,
        patch_size = 128,
        stride     = 128,
        transform  = val_transform
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size   = Config.BATCH_SIZE,
        shuffle      = False,
        num_workers  = Config.NUM_WORKERS,
        pin_memory   = Config.PIN_MEMORY
    )

    imprimir_distribucion_clases_post_augmentation(
        train_loader, 
        n_classes=6, 
        title="Distribución de clases en el set de ENTRENAMIENTO (post-augmentation)"
    )
    
    # --- Instanciación del Modelo, Loss y Optimizador ---
    model = CloudDeepLabV3Plus(num_classes=6).to(Config.DEVICE)
    
    alpha_np = compute_class_weights(
        dataset = train_dataset,      # puede ser el de patches
        n_classes = 6,
        method = "median"             # o "inverse"
    )
    print("Pesos alpha =", alpha_np)  # para que veas el resultado

    alpha = torch.tensor(alpha_np, dtype=torch.float32, device=Config.DEVICE)

    loss_fn = FocalTverskyLoss(
        n_classes      = 6,
        alpha          = 0.3,        # pondera FP
        beta           = 0.7,        # pondera FN  → ↑ sensibilidad
        gamma          = 2.0,        # componente “focal”
        class_weights  = alpha       # tensor de 6 pesos que ya calculaste
    ).to(Config.DEVICE)
    
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

    # ---------------------------------------------------------------
    # Cargar el modelo con mejor mIoU ANTES de evaluarlo definitivamente
    # ---------------------------------------------------------------
    if os.path.exists(Config.MODEL_SAVE_PATH):
        checkpoint = torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE)
        model.load_state_dict(checkpoint["state_dict"])          # ←  ❗️
        print(f"\nModelo recargado del epoch {checkpoint['epoch']} "
            f"con mIoU guardado = {checkpoint['best_mIoU']:.4f}")
    else:
        print("\n⚠️  No se encontró el checkpoint; "
            "se evaluará el modelo tal cual está en memoria.")

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
