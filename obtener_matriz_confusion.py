import torch
from config import Config
from train_test4 import CloudDataset, CloudDeepLabV3Plus
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import confusion_matrix
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

def get_val_loader():
    val_transform = A.Compose([
        A.Resize(Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH),
        A.Normalize(mean=[0,0,0], std=[1,1,1], max_pixel_value=255.0),
        ToTensorV2(),
    ])

    val_ds = CloudDataset(
        image_dir=Config.VAL_IMG_DIR,
        mask_dir= Config.VAL_MASK_DIR,
        transform=val_transform
    )
    return DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        shuffle=False
    )

# ================================
# 2. CARGA DE MODELO
# ================================
def load_model(checkpoint_path: str, device: torch.device):
    """
    Carga el modelo desde un checkpoint, manejando el prefijo '_orig_mod.' 
    añadido por torch.compile().
    """
    model = CloudDeepLabV3Plus(num_classes=6).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Obtener el state_dict del checkpoint
    state_dict = checkpoint['state_dict']
    
    # Crear un nuevo state_dict sin el prefijo '_orig_mod.'
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('_orig_mod.'):
            # Eliminar el prefijo: '_orig_mod.' tiene 10 caracteres
            name = k[10:] 
            new_state_dict[name] = v
        else:
            new_state_dict[k] = v
            
    # Cargar el state_dict corregido
    model.load_state_dict(new_state_dict)
    model.eval()
    return model

# ================================
# 3. CÁLCULO DE MATRIZ DE CONFUSIÓN
# ================================
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

    # generar matriz de confusión
    cm = confusion_matrix(y_true, y_pred, labels=list(range(6)))
    print("Matriz de confusión:\n", cm)

    # plot
    class_names = [f"Clase {i}" for i in range(6)]
    plt.figure(figsize=(7,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predichos")
    plt.ylabel("Reales")
    plt.title("Matriz de Confusión")
    plt.tight_layout()

    # =================================================================
    # AÑADE ESTA LÍNEA PARA GUARDAR EL GRÁFICO
    # =================================================================
    # Puedes cambiar el nombre y el formato del archivo (png, jpg, pdf, etc.)
    plt.savefig('matriz_de_confusion.png', dpi=300) 
    # =================================================================

    plt.show() # Esta línea ahora mostrará el gráfico después de guardarlo.

def main():
    device = torch.device(Config.DEVICE)
    val_loader = get_val_loader()

    # ruta a tu checkpoint
    checkpoint_path = r"/content/drive/MyDrive/colab/0.8410.pth.tar"
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"No se encontró el archivo: {checkpoint_path}")

    model = load_model(checkpoint_path, device)
    compute_and_plot_confusion_matrix(model, val_loader, device)

if __name__ == "__main__":
    main()