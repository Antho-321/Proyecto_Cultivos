import cv2
import numpy as np

def canny_edge_detector(img, low_threshold=100, high_threshold=200, aperture_size=3, L2gradient=False):
    """
    Aplica el detector de bordes de Canny a una imagen.
    
    Parámetros
    ----------
    img : np.ndarray
        Imagen de entrada en formato H×W×C (uint8). Si tiene 3 canales, se convertirá a gris.
    low_threshold : int
        Umbral inferior para el histéresis.
    high_threshold : int
        Umbral superior para el histéresis.
    aperture_size : int
        Tamaño del kernel Sobel (3, 5 o 7).
    L2gradient : bool
        Si True, usa la norma L2 para el gradiente (más preciso, más lento).

    Retorna
    -------
    edges : np.ndarray
        Mapa de bordes binario (H×W, uint8, 0 o 255).
    """
    # 1) Convertir a escala de grises si viene en color
    if img.ndim == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # 2) Aplicar suavizado Gaussiano para reducir ruido (opcional pero recomendado)
    #    Ajusta ksize y sigma según tu dataset
    gray = cv2.GaussianBlur(gray, (5, 5), sigmaX=1.4, sigmaY=1.4)
    
    # 3) Canny
    edges = cv2.Canny(
        gray,
        threshold1=low_threshold,
        threshold2=high_threshold,
        apertureSize=aperture_size,
        L2gradient=L2gradient
    )
    return edges


"""
*********************************
 PARA IMPLEMENTACION EN DEVELOP
*********************************

1. Modificar tu loader para concatenar el canal de bordes
En tu función load_augmented_dataset, justo después de aplicar la augmentación y antes de normalizar a [0,1], haz algo así:


from your_canny_module import canny_edge_detector  # importa tu función

def load_augmented_dataset(img_dir, mask_dir, target_size=(256,256), augment=False):
    images, masks = [], []
    files = [...]
    aug = get_training_augmentation() if augment else get_validation_augmentation()

    with ThreadPoolExecutor() as exe:
        futures = {…}
        for future in tqdm(as_completed(futures), total=len(futures)):
            img_arr, mask_arr = future.result()
            
            # 1) si usas albumentations, tu img_arr está en uint8 [0–255]
            if augment:
                augm = aug(image=img_arr, mask=mask_arr)
                img_arr, mask_arr = augm['image'], augm['mask']

            # 2) calcula bordes sobre la imagen en uint8
            edges = canny_edge_detector(img_arr,
                                        low_threshold=50,
                                        high_threshold=150)
            # 3) normaliza todo a float32 [0,1]
            img_f   = img_arr.astype('float32')  / 255.0
            edges_f = edges.astype('float32')    / 255.0

            # 4) concatena como canal extra
            img_with_edges = np.concatenate([img_f, edges_f[...,None]], axis=-1)
            
            images.append(img_with_edges)
            masks.append(mask_arr)

    X = np.stack(images, dtype='float32')  # shape = (B, H, W, 4)
    y = np.stack(masks,  dtype='int32')
    …
    return X, y

2. Ajustar el modelo para 4 canales de entrada
Como tu loader ya entrega X.shape[-1]==4, simplemente pásalo al construir el modelo:

# antes: shape=(256,256,3)
iou_aware_model = IoUOptimizedModel(
    shape=train_X.shape[1:],   # ahora (256,256,4)
    num_classes=num_classes
)

Y en tu función build_model no necesitas tocar nada más, pues usas:
inp = Input(shape=shape)  # shape=(256,256,4) 

Modelo de dos ramas (más avanzado):

Rama A procesa los 3 canales RGB con EfficientNetV2S pretrained.

Rama B procesa el canal de bordes con un pequeño CNN.

Luego fusionas sus features antes del decodificador.
"""