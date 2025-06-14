import numpy as np

def compute_image_sampling_weights(masks, class_weights):
    """
    Calcula un peso para cada imagen en función de las clases que contiene su máscara.
    
    Parámetros
    ----------
    masks : np.ndarray, shape (N, H, W)
        Array de máscaras con valores de clase enteros.
    class_weights : dict
        Diccionario {clase: peso} (por ejemplo, del median-frequency balancing).
        
    Devuelve
    -------
    image_weights : np.ndarray, shape (N,)
        Pesos normalizados (suman 1) para muestreo.
    """
    N = masks.shape[0]
    weights = np.zeros(N, dtype=np.float32)
    for i in range(N):
        clases_presentes = np.unique(masks[i])
        # sumar pesos de cada clase presente
        w = sum(class_weights.get(int(c), 0.0) for c in clases_presentes)
        weights[i] = w
    # evito ceros totales
    if weights.sum() == 0:
        weights[:] = 1.0
    image_weights = weights / weights.sum()
    return image_weights


def oversampled_data_generator(images, masks, class_weights, batch_size, shuffle=True):
    """
    Generador infinito de minibatches con oversampling de imágenes según class_weights.
    
    Parámetros
    ----------
    images : np.ndarray, shape (N, H, W, C)
    masks  : np.ndarray, shape (N, H, W)
    class_weights : dict
        Diccionario {clase: peso}.
    batch_size : int
    shuffle : bool
        Si es True, mezcla los índices de base antes de cada época.
        
    Yields
    ------
    batch_images : np.ndarray, shape (batch_size, H, W, C)
    batch_masks  : np.ndarray, shape (batch_size, H, W)
    """
    N = images.shape[0]
    # pesos de muestreo por imagen
    image_weights = compute_image_sampling_weights(masks, class_weights)
    # índices base
    base_idx = np.arange(N)
    
    while True:
        if shuffle:
            # opcional: mezclo el orden de los índices base
            np.random.shuffle(base_idx)
        # muestreo ponderado
        sampled_idx = np.random.choice(base_idx, size=batch_size, replace=True, p=image_weights)
        yield images[sampled_idx], masks[sampled_idx]

"""
# Supongamos que ya has calculado:
# class_weights = {0:0.12, 1:3.70, 2:7.18, 3:4.21, 4:16.33, 5:1.00}

batch_size = 16
gen = oversampled_data_generator(images, masks, class_weights, batch_size)

# En tu fit de Keras:
model.fit(
    gen,
    steps_per_epoch = len(images) // batch_size,
    epochs = 50,
    # validation_data=...
)
"""