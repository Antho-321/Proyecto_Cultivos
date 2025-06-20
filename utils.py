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
        counts = torch.bincount(masks.flatten().long(), minlength=n_classes)
        class_counts += counts

    # Calcula el porcentaje para cada clase
    class_distribution = (class_counts.float() / total_pixels) * 100

    # ------------------------------------------------------------------
    # ### NUEVO: pesos de importancia inversamente proporcionales
    # Fórmula clásica:   w_i  =  total_pix / (n_clases * count_i)
    # ⇒  clases raras  →  peso grande,   clases frecuentes → pequeño.
    # Normalizamos para que el promedio sea 1 (opcional pero práctico).
    eps = 1e-8                                         # evita división por 0
    raw_weights = total_pixels / (n_classes * (class_counts.float() + eps))
    class_weights = raw_weights / raw_weights.mean()   # media = 1
    # ------------------------------------------------------------------

    # Imprime los resultados
    print(f"Número total de píxeles analizados: {total_pixels}")
    print("Distribución final de píxeles por clase:")
    for i, dist in enumerate(class_distribution):
        print(f"  Clase {i}: {dist:.4f}%  ({class_counts[i]} píxeles)")
    
    # ------------------------------------------------------------------
    # ### NUEVO: muestra los pesos calculados
    print("\nPesos de importancia por clase (para CrossEntropy/Focal):")
    for i, w in enumerate(class_weights):
        print(f"  Clase {i}: {w:.4f}")
    # ------------------------------------------------------------------

    print("-" * (len(title) + 6))

# =================================================================================
# FUNCIÓN DE RECORTE (AÑADIDA)
# =================================================================================
def crop_around_classes(
    image: np.ndarray,
    mask: np.ndarray,
    classes_to_find: list[int] = [1, 2, 3, 4, 5],
    margin: int = 10  # <--- AÑADIDO: Un pequeño margen puede ser útil
) -> tuple[np.ndarray, np.ndarray]:
    """
    Recorta un rectángulo alrededor de todos los píxeles que pertenecen a las
    clases especificadas en la máscara.
    """
    # is_class_present será 2D (H,W) ya que la máscara de entrada es (H,W,1)
    is_class_present = np.isin(mask.squeeze(), classes_to_find)

    ys, xs = np.where(is_class_present)

    if ys.size == 0:
        return image, mask # Devuelve la máscara original (H,W,1)

    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()

    y0 = max(0, y_min - margin)
    y1 = min(mask.shape[0], y_max + margin + 1)
    x0 = max(0, x_min - margin)
    x1 = min(mask.shape[1], x_max + margin + 1)

    cropped_image = image[y0:y1, x0:x1]
    # Se recorta la máscara (H,W,1) y mantiene sus 3 dimensiones
    cropped_mask = mask[y0:y1, x0:x1, :]

    return cropped_image, cropped_mask

def save_performance_plot(train_history, val_history, save_path):
    """
    Guarda un gráfico comparando el mIoU de entrenamiento y validación por época.

    Args:
        train_history (list): Lista con los valores de mIoU de entrenamiento por época.
        val_history (list): Lista con los valores de mIoU de validación por época.
        save_path (str): Ruta donde se guardará el gráfico en formato PNG.
    """
    epochs = range(1, len(train_history) + 1)
    
    plt.style.use('seaborn-v0_8-darkgrid') # Estilo visual atractivo
    fig, ax = plt.subplots(figsize=(12, 7))

    # Graficar ambas curvas
    ax.plot(epochs, train_history, 'o-', color="xkcd:sky blue", label='Entrenamiento (mIoU)', markersize=4)
    ax.plot(epochs, val_history, 'o-', color="xkcd:amber", label='Validación (mIoU)', markersize=4)

    # Títulos y etiquetas
    ax.set_title('Rendimiento del Modelo: mIoU por Época', fontsize=16, weight='bold')
    ax.set_xlabel('Época', fontsize=12)
    ax.set_ylabel('mIoU (Mean Intersection over Union)', fontsize=12)
    
    # Leyenda, cuadrícula y límites
    ax.legend(fontsize=11, frameon=True, shadow=True)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_ylim(0, max(1.0, max(val_history)*1.1)) # Límite Y hasta 1.0 o un poco más del máximo
    ax.set_xticks(epochs) # Asegura que se muestren todas las épocas si no son demasiadas

    plt.tight_layout()
    plt.savefig(save_path, dpi=150) # Guardar con buena resolución
    plt.close(fig) # Liberar memoria
    print(f"📈 Gráfico de rendimiento guardado en '{save_path}'")