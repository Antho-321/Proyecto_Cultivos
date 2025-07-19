# postprocesamiento.py
import os
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import (
    unary_from_softmax,
    create_pairwise_gaussian,
    create_pairwise_bilateral,
)

from model2 import CloudDeepLabV3Plus
from config import Config
from continuar_entrenamiento import CloudDataset   # dataset ya definido

USE_CRF   = False           # ← pon True si quieres volver a activarlo
NUM_CLASSES = 6
CKPT_PATH   = Config.MODEL_SAVE_PATH
SAVE_DIR    = "./pred_masks_pp"

# --------------------------------------------------------------------
# 1. Funciones de pos-procesado
# --------------------------------------------------------------------
def dense_crf_refine(image_rgb, probs, n_iters: int = 50):
    """Refina softmax (C×H×W) con Dense CRF → máscara H×W."""
    h, w = image_rgb.shape[:2]
    d = dcrf.DenseCRF2D(w, h, NUM_CLASSES)

    unary = unary_from_softmax(probs)          # C × (H*W)
    d.setUnaryEnergy(unary)

    gauss = create_pairwise_gaussian(sdims=(3, 3), shape=(h, w))
    d.addPairwiseEnergy(gauss, compat=3)

    bilateral = create_pairwise_bilateral(
        sdims=(80, 80), schan=(13, 13, 13),
        img=image_rgb, chdim=2
    )
    d.addPairwiseEnergy(bilateral, compat=10)

    Q = d.inference(n_iters)
    return np.argmax(Q, axis=0).reshape(h, w).astype(np.uint8)


def morph_refine(mask, k: int = 3, min_area: int = 50):
    """Opening + Closing + eliminación de componentes pequeñas."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    refined = np.zeros_like(mask, dtype=np.uint8)

    for c in range(NUM_CLASSES):
        bin_mask = (mask == c).astype(np.uint8)

        bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_OPEN,  kernel)
        bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kernel)

        n_lbl, lbls, stats, _ = cv2.connectedComponentsWithStats(bin_mask, 8)
        for lab in range(1, n_lbl):
            if stats[lab, cv2.CC_STAT_AREA] < min_area:
                bin_mask[lbls == lab] = 0

        refined[bin_mask == 1] = c
    return refined


def postprocess(logits, rgb_img):
    """
    logits: torch.Tensor C×H×W (en CPU)
    rgb_img: np.ndarray H×W×3  (solo se usa si USE_CRF=True)
    """
    if USE_CRF:
        probs = torch.softmax(logits, dim=0).cpu().numpy()
        crf_mask = dense_crf_refine(rgb_img, probs)      # CRF
        return morph_refine(crf_mask)                    # morfología
    else:
        # 1️⃣ argmax directo  2️⃣ morfología ligera
        raw_mask = logits.argmax(0).cpu().numpy().astype(np.uint8)
        return morph_refine(raw_mask)

# --------------------------------------------------------------------
# 2. Inferencia + métricas
# --------------------------------------------------------------------
@torch.no_grad()
def run_inference():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Transform de validación (sin augmentations)
    val_tf = Compose([
        Resize(Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH),
        Normalize(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255.0),
        ToTensorV2(),
    ])
    val_ds = CloudDataset(Config.VAL_IMG_DIR, Config.VAL_MASK_DIR, val_tf)
    loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    # --------  Modelo  --------
    model = CloudDeepLabV3Plus(num_classes=NUM_CLASSES).to(Config.DEVICE)
    torch._inductor.config.triton.cudagraphs = True
    model = torch.compile(model)          # necesario para coincidir con el checkpoint
    ckpt  = torch.load(CKPT_PATH, map_location=Config.DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # --------  Métricas  --------
    conf_mat = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64)

    for i, (img_t, gt_mask) in enumerate(loader):
        img_t   = img_t.to(Config.DEVICE, non_blocking=True)
        gt_mask = gt_mask.squeeze(0).cpu().numpy()            # H×W

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(img_t)[0].squeeze(0).to("cpu")     # C×H×W

        # Imagen RGB original y **resize** al tamaño de logits
        rgb = np.array(Image.open(
            os.path.join(Config.VAL_IMG_DIR, val_ds.images[i])
        ))
        h, w = logits.shape[1:]
        if (rgb.shape[0], rgb.shape[1]) != (h, w):
            rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)

        refined = postprocess(logits, rgb)                    # H×W uint8

        # ---- guardar máscara refinada
        out_pth = os.path.join(
            SAVE_DIR, f"{os.path.splitext(val_ds.images[i])[0]}_pp.png"
        )
        Image.fromarray(refined).save(out_pth)
        print(f"[{i+1}/{len(loader)}] → {out_pth}")

        # ---- actualizar matriz de confusión
        flat = refined.flatten() * NUM_CLASSES + gt_mask.flatten()
        hist = torch.bincount(torch.from_numpy(flat),
                              minlength=NUM_CLASSES**2).view(NUM_CLASSES, NUM_CLASSES)
        conf_mat += hist.to(conf_mat.dtype)

    # --------  Cálculo final de IoU / Dice  --------
    inter      = conf_mat.diag().float()
    sum_pred   = conf_mat.sum(1).float()
    sum_truth  = conf_mat.sum(0).float()
    union      = sum_pred + sum_truth - inter + 1e-6

    iou   = inter / union
    dice  = (2 * inter) / (sum_pred + sum_truth + 1e-6)

    print("\nPost-process metrics")
    for c, (iou_c, dice_c) in enumerate(zip(iou, dice)):
        print(f"Clase {c:<2}: IoU {iou_c:.3f} | Dice {dice_c:.3f}")
    print(f"mIoU  : {iou.mean():.4f}")
    print(f"mDice : {dice.mean():.4f}")

# --------------------------------------------------------------------
if __name__ == "__main__":
    run_inference()
