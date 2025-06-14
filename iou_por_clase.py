import numpy as np
def print_class_iou(model, X, y_true, class_names=None):
    """
    Calcula e imprime el Intersection over Union (IoU) por clase.

    Parámetros
    ----------
    model : tf.keras.Model
        Modelo de segmentación entrenado.
    X : np.ndarray, shape (B, H, W, C)
        Imágenes de entrada.
    y_true : np.ndarray, shape (B, H, W)
        Máscaras verdaderas con etiquetas de clase (0..num_classes-1).
    class_names : list de str, opcional
        Nombres para cada clase; si es None, se usan los índices 0..num_classes-1.

    Retorna
    -------
    ious : list de float
        IoU calculado para cada clase.
    """
    # 1. Predecir máscaras
    preds = model.predict(X)
    pred_labels = np.argmax(preds, axis=-1)  # shape (B, H, W)

    # 2. Número de clases
    num_classes = int(np.max(y_true) + 1)
    ious = []

    # 3. Calcular IoU por clase
    for cls in range(num_classes):
        # máscaras binarias para la clase cls
        y_cls_true = (y_true == cls)
        y_cls_pred = (pred_labels == cls)

        # intersección y unión
        intersection = np.logical_and(y_cls_true, y_cls_pred).sum()
        union = np.logical_or(y_cls_true, y_cls_pred).sum()

        # evitar división por cero
        iou = intersection / union if union > 0 else np.nan
        ious.append(iou)

        # nombre de la clase para imprimir
        label = class_names[cls] if class_names is not None else cls
        print(f"IoU Clase {label}: {iou:.4f}")

    # 4. Imprimir mean IoU
    mean_iou = np.nanmean(ious)
    print(f"\nMean IoU (todas las clases): {mean_iou:.4f}")

    return ious