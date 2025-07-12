import os
import torch
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import confusion_matrix
import numpy as np

from config import Config
from train_test4 import CloudDataset, CloudDeepLabV3Plus

def get_val_loader():
    val_transform = A.Compose([
        A.Resize(Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH),
        A.Normalize(mean=[0,0,0], std=[1,1,1], max_pixel_value=255.0),
        ToTensorV2(),
    ])

    val_ds = CloudDataset(
        image_dir=Config.VAL_IMG_DIR,
        mask_dir=Config.VAL_MASK_DIR,
        transform=val_transform
    )
    return DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        shuffle=False
    )

def load_model(checkpoint_path: str, device: torch.device):
    model = CloudDeepLabV3Plus(num_classes=6).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['state_dict']
    new_sd = {}
    for k, v in state_dict.items():
        if k.startswith('_orig_mod.'):
            new_sd[k[10:]] = v
        else:
            new_sd[k] = v
    model.load_state_dict(new_sd)
    model.eval()
    return model

def calculate_mean_iou(cm: np.ndarray):
    """
    Calcula el IoU por clase y el mean IoU a partir de la matriz de confusión absoluta.
    Devuelve una lista de IoU por clase y el mean IoU.
    """
    ious = []
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        denom = tp + fp + fn
        iou = tp / denom if denom > 0 else float('nan')
        ious.append(iou)
    return ious, np.nanmean(ious)

def compute_miou(model, loader, device):
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(device, non_blocking=True)
            logits = model(imgs)
            logits = logits[0] if isinstance(logits, tuple) else logits
            preds = torch.argmax(logits, dim=1).cpu().numpy().ravel()
            all_preds.append(preds)

            labels = masks.numpy().ravel()
            all_labels.append(labels)

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)

    # 1) Matriz de confusión absoluta
    cm = confusion_matrix(y_true, y_pred, labels=list(range(6)))
    print("Matriz de confusión (absoluta):\n", cm)

    class_names = [
        "Fondo",           # índice 0  → (255,255,255)
        "Lengua de vaca",  # índice 1  → (128,0,0)
        "Diente de león",  # índice 2  → (0,128,0)
        "Kikuyo",          # índice 3  → (255,255,0)
        "Otro",            # índice 4  → (0,0,0)
        "Papa",            # índice 5  → (128,0,128)
    ]

    # Cálculo de IoU
    ious, mean_iou = calculate_mean_iou(cm)
    for cls_name, iou in zip(class_names, ious):
        print(f"IoU {cls_name}: {iou:.4f}")
    print(f"Mean IoU: {mean_iou:.4f}")

def main():
    device = torch.device(Config.DEVICE)
    val_loader = get_val_loader()

    checkpoint_path = "/content/drive/MyDrive/colab/0.8410miou.pth.tar"
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"No se encontró el archivo: {checkpoint_path}")

    model = load_model(checkpoint_path, device)
    compute_miou(model, val_loader, device)

if __name__ == "__main__":
    main()
