import torch
from tqdm import tqdm
def imprimir_distribucion_clases_post_augmentation(loader, n_classes=6, title="Distribución de clases post-augmentation"):
    """
    Calcula y muestra la distribución porcentual de píxeles por cada clase
    en un DataLoader después de que las transformaciones han sido aplicadas.

    Args:
        loader (DataLoader): El DataLoader a analizar (ej. train_loader).
        n_classes (int): El número total de clases en la segmentación.
        title (str): Título para la impresión de resultados.
    """
    print(f"\n--- {title} ---")
    class_counts = torch.zeros(n_classes, dtype=torch.int64)
    total_pixels = 0

    # Itera sobre el loader para acceder a las máscaras aumentadas
    loop = tqdm(loader, desc="Analizando distribución")
    for _, masks in loop:
        # masks tiene shape (N, H, W)
        total_pixels += masks.numel() # N * H * W

        # Cuenta los píxeles para cada clase en el batch actual
        # torch.bincount es muy eficiente para esta tarea
        # Aplanamos todas las máscaras del batch a un solo vector
        counts = torch.bincount(masks.flatten(), minlength=n_classes)
        class_counts += counts

    # Calcula el porcentaje para cada clase
    class_distribution = (class_counts.float() / total_pixels) * 100

    # Imprime los resultados
    print(f"Número total de píxeles analizados: {total_pixels}")
    print("Distribución final de píxeles por clase:")
    for i, dist in enumerate(class_distribution):
        print(f"  Clase {i}: {dist:.4f}%  ({class_counts[i]} píxeles)")
    print("-" * (len(title) + 6))