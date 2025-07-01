import os
import torch
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
    # corregir prefijo de torch.compile()
    new_sd = {}
    for k,v in state_dict.items():
        if k.startswith('_orig_mod.'):
            new_sd[k[10:]] = v
        else:
            new_sd[k] = v
    model.load_state_dict(new_sd)
    model.eval()
    return model

def compute_and_plot_confusion_matrix(model, loader, device):
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

    # 1) Confusion matrix absoluta
    cm = confusion_matrix(y_true, y_pred, labels=list(range(6)))
    print("Matriz de confusión (absoluta):\n", cm)

    class_names = [
        "Fondo",   # índice 0  → (255,255,255)
        "Lengua de vaca",                # índice 1  → (128,0,0)
        "Diente de león",          # índice 2  → (0,128,0)
        "Kikuyo",        # índice 3  → (255,255,0)
        "Otro",       # índice 4  → (0,0,0)
        "Papa",    # índice 5  → (128,0,128)
    ]
    plt.figure(figsize=(7,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predichos")
    plt.ylabel("Reales")
    plt.title("Matriz de Confusión (absoluta)")
    plt.tight_layout()
    plt.savefig("matriz_confusion_absoluta.png", dpi=300)
    plt.close()

    # 2) Confusion matrix normalizada por fila
    cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(7,6))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predichos")
    plt.ylabel("Reales")
    plt.title("Matriz de Confusión (normalizada por fila)")
    plt.tight_layout()
    plt.savefig("matriz_confusion_normalizada.png", dpi=300)
    plt.show()

def main():
    device = torch.device(Config.DEVICE)
    val_loader = get_val_loader()

    checkpoint_path = r"C:\Users\Administrador\Documents\INGENIERIA_EN_SOFTWARE\BIG_DATA\CÓDIGO\Proyecto_Cultivos\datos_expo\0.8410.pth.tar"
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"No se encontró el archivo: {checkpoint_path}")

    model = load_model(checkpoint_path, device)
    compute_and_plot_confusion_matrix(model, val_loader, device)

if __name__ == "__main__":
    main()