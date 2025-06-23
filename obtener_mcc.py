import os
import torch
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import confusion_matrix, matthews_corrcoef # confusion_matrix se mantiene para el cálculo por clase
import numpy as np
import matplotlib.pyplot as plt
# Se eliminó "import seaborn as sns"

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

def plot_mcc_per_class(cm, class_names):
    """
    Calcula y grafica el Coeficiente de Correlación de Matthews (MCC) para cada clase.
    """
    num_classes = cm.shape[0]
    mcc_scores = []

    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - (tp + fp + fn)

        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        if denominator == 0:
            mcc = 0.0
        else:
            mcc = (tp * tn - fp * fn) / denominator
        
        mcc_scores.append(mcc)

    plt.figure(figsize=(10, 6))
    bars = plt.bar(class_names, mcc_scores, color=plt.cm.viridis(np.linspace(0.1, 0.9, len(class_names))))
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.3f}', va='bottom' if yval >= 0 else 'top', ha='center')

    plt.ylabel("Puntuación MCC")
    plt.title("Coeficiente de Correlación de Matthews (MCC) por Clase")
    plt.ylim(min(mcc_scores) - 0.1, 1.1)
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("mcc_por_clase.png", dpi=300)
    print("\nGráfico de MCC por clase guardado como 'mcc_por_clase.png'")

def compute_and_plot_mcc(model, loader, device): # <--- FUNCIÓN RENOMBRADA
    all_preds = []
    all_labels = []

    print("Calculando predicciones en el conjunto de validación...")
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

    # 1) Calcular MCC General
    mcc_general = matthews_corrcoef(y_true, y_pred)
    print(f"\nMatthews Correlation Coefficient (MCC) General: {mcc_general:.4f}")

    # 2) Calcular MCC por clase (requiere la matriz de confusión como paso intermedio)
    class_names = [f"Clase {i}" for i in range(6)]
    cm = confusion_matrix(y_true, y_pred, labels=list(range(6)))
    
    # 3) Graficar MCC por clase
    plot_mcc_per_class(cm, class_names)
    
    # Mostrar el gráfico de MCC al final
    plt.show()

def main():
    device = torch.device(Config.DEVICE)
    val_loader = get_val_loader()

    checkpoint_path = r"/content/drive/MyDrive/colab/0.8410.pth.tar"
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"No se encontró el archivo: {checkpoint_path}")

    print("Cargando modelo...")
    model = load_model(checkpoint_path, device)
    
    compute_and_plot_mcc(model, val_loader, device) # <--- LLAMADA A LA FUNCIÓN RENOMBRADA

if __name__ == "__main__":
    main()