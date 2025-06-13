# -*- coding: utf-8 -*-
"""
claude_efficientnetv2_deeplab_iou_optimized.py

Script completo: carga datos, construye y entrena el modelo DeepLabV3+ usando EfficientNetV2S,
con optimizaciones específicas para maximizar la métrica Mean IoU.

Implementa las siguientes recomendaciones:
1. Re-balanceo de pesos en la función de pérdida para priorizar términos de IoU.
2. Callbacks de EarlyStopping y LR Scheduler monitorean `val_enhanced_mean_iou`.
3. Composición de mini-lotes con sobremuestreo de clases minoritarias.
4. Inclusión de un término de pérdida de bordes (Boundary-aware).
5. Filtros más profundos (320 canales) en el módulo ASPP.
6. Aumentación en validación (TTA) durante el entrenamiento.
7. Eliminación de la métrica de 'accuracy' durante la compilación del modelo.
8. Carga explícita del mejor checkpoint entre fases de entrenamiento (skip mismatches).
"""

import os
os.environ['MPLBACKEND'] = 'agg'  # matplotlib backend

import tensorflow as tf
import keras
print("Versión de TensorFlow:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"GPUs disponibles: {[gpu.name for gpu in gpus]}")
else:
    print("Sin GPU, se usará CPU.")

import numpy as np
import cv2
from tqdm import tqdm
import albumentations as A
from concurrent.futures import ThreadPoolExecutor, as_completed
from tensorflow.keras import layers, backend as K, Model
from tensorflow.keras.layers import (
    Input, Conv2D, SeparableConv2D, Conv2DTranspose,
    BatchNormalization, Activation, Dropout,
    Lambda, Concatenate, UpSampling2D
)
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, TensorBoard,
    ReduceLROnPlateau, LearningRateScheduler
)
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from lovasz_losses_tf import lovasz_softmax


# -------------------------------------------------------------------------------
# Monkey-patch SE module in keras_efficientnet_v2
# -------------------------------------------------------------------------------
import keras_efficientnet_v2.efficientnet_v2 as _efn2

def patched_se_module(inputs, se_ratio=0.25, name=""):
    data_format = K.image_data_format()
    h_axis, w_axis = (1,2) if data_format=='channels_last' else (2,3)
    se = Lambda(lambda x: tf.reduce_mean(x, axis=[h_axis,w_axis], keepdims=True),
                name=name+"se_reduce_pool")(inputs)
    channels = inputs.shape[-1]
    reduced = max(1, int(channels/se_ratio))
    se = Conv2D(reduced,1,padding='same', name=name+"se_reduce_conv")(se)
    se = Activation('swish', name=name+"se_reduce_act")(se)
    se = Conv2D(channels,1,padding='same', name=name+"se_expand_conv")(se)
    se = Activation('sigmoid', name=name+"se_excite")(se)
    return layers.Multiply(name=name+"se_excite_mul")([inputs, se])

_efn2.se_module = patched_se_module
from keras_efficientnet_v2 import EfficientNetV2S

# ===============================================================================
# IMPLEMENTACIONES DEL CHEAT-SHEET (Traducción a TF/Keras)
# ===============================================================================

def compute_uncertainty_entropy_tf(logits):
    """
    Calcula la incertidumbre aleatoria (aleatoric) por píxel como la entropía.
    - logits: Tensor (N, H, W, C)
    - devuelve: Tensor (N, H, W) con entropía por píxel
    """
    probs = tf.nn.softmax(logits, axis=-1)
    entropy = -tf.reduce_sum(probs * tf.math.log(probs + 1e-8), axis=-1)
    return entropy

def uncertainty_weighted_loss_tf(y_true, y_pred, uncertainty_map, criterion):
    """
    Aplica ponderación inversa a la incertidumbre en la función de pérdida.
    - y_true, y_pred: Tensores (N, H, W, C) y (N, H, W)
    - uncertainty_map: Tensor (N, H, W)
    - criterion: una función de pérdida con reduction='none'
    """
    per_pixel_loss = criterion(y_true, y_pred) # (N, H, W)
    weights = 1.0 / (1.0 + uncertainty_map)
    return tf.reduce_mean(per_pixel_loss * weights)

def boundary_aware_dice_loss_tf(y_true, y_pred, n=3, smooth=1e-6):
    """
    Dice consciente de bordes en una banda de n píxeles (versión TF).
    - y_true: (N, H, W), valores enteros de clase.
    - y_pred: (N, H, W, C), logits.
    """
    y_true_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), depth=num_classes, axis=-1) # (N,H,W,C)
    y_pred_probs = tf.nn.softmax(y_pred, axis=-1)
    
    # 1) Detectar bordes con Sobel
    sobel_x = tf.constant([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=tf.float32)
    sobel_y = tf.transpose(sobel_x)
    sobel_filter = tf.stack([sobel_x, sobel_y], axis=-1)
    sobel_filter = tf.expand_dims(tf.expand_dims(sobel_filter, -2), -1) # (3,3,2,1)
    
    # Target debe ser (N, H, W, 1) para conv2d
    target_f = tf.cast(tf.expand_dims(y_true, -1), tf.float32)
    
    # Conv para cada clase
    edges_per_class = []
    for i in range(num_classes):
        class_mask = tf.expand_dims(y_true_one_hot[..., i], axis=-1)
        edge_x = tf.nn.conv2d(class_mask, tf.expand_dims(tf.expand_dims(sobel_x,-1),-1), strides=1, padding='SAME')
        edge_y = tf.nn.conv2d(class_mask, tf.expand_dims(tf.expand_dims(sobel_y,-1),-1), strides=1, padding='SAME')
        edges_per_class.append(tf.abs(edge_x) + tf.abs(edge_y))

    edge_mask_combined = tf.reduce_sum(tf.stack(edges_per_class, axis=-1), axis=-1)
    edge_mask = tf.cast(edge_mask_combined > 0, tf.float32) # (N, H, W, 1)

    band = distance_band(edge_mask, n)

    # 3) Ponderar y calcular Dice por clase
    weights = 1 + band
    p = y_pred_probs * weights
    t = y_true_one_hot * weights
    
    intersection = tf.reduce_sum(p * t, axis=[1, 2])
    denominator = tf.reduce_sum(p, axis=[1, 2]) + tf.reduce_sum(t, axis=[1, 2])
    dice_per_class = (2 * intersection + smooth) / (denominator + smooth)
    
    return 1 - tf.reduce_mean(dice_per_class)

def adaptive_boundary_enhanced_dice_loss_tf(y_true, y_pred, gamma=2.0, smooth=1e-6):
    """
    ABeDice: re-pesado exponencial de píxeles de borde (versión TF).
    """
    y_true_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), depth=num_classes, axis=-1)
    y_pred_probs = tf.nn.softmax(y_pred, axis=-1)

    sobel_x = tf.constant([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=tf.float32)
    sobel_filter = tf.reshape(sobel_x, [3, 3, 1, 1])
    
    edges_per_class = []
    for i in range(num_classes):
        class_mask = tf.expand_dims(y_true_one_hot[..., i], -1)
        edge_x = tf.nn.conv2d(class_mask, sobel_filter, strides=1, padding='SAME')
        edge_y = tf.nn.conv2d(class_mask, tf.transpose(sobel_filter, [1,0,2,3]), strides=1, padding='SAME')
        edges_per_class.append(tf.abs(edge_x) + tf.abs(edge_y))

    edge_mask = tf.reduce_sum(tf.concat(edges_per_class, axis=-1), axis=-1, keepdims=True)
    edge_mask = tf.cast(edge_mask > 0, tf.float32)
    
    weights = tf.exp(gamma * edge_mask)
    
    p = y_pred_probs * weights
    t = y_true_one_hot * weights
    
    inter = tf.reduce_sum(p * t, axis=[1,2])
    denominator = tf.reduce_sum(p, axis=[1,2]) + tf.reduce_sum(t, axis=[1,2])
    dice = (2 * inter + smooth) / (denominator + smooth)
    
    return 1 - tf.reduce_mean(dice)

def cutpaste_augment_np(image, mask, background):
    """
    Cut-Paste (NumPy): recorta el objeto usando la máscara y pega sobre otro fondo.
    - image, background: np.array HxWx3 (float32 en [0,1])
    - mask: np.array HxW (0/1)
    - devuelve: composite (HxWx3), mask (HxW)
    """
    mask_expanded = np.expand_dims(mask, axis=-1)
    fg = image * mask_expanded
    bg = background * (1 - mask_expanded)
    return fg + bg, mask

class LoRAAdapterLayer(layers.Layer):
    """
    Low-Rank Adapter para fine-tuning eficiente (LoRA) en Keras.
    Se inserta en capas convolucionales.
    """
    def __init__(self, original_layer, r=4, alpha=1.0, **kwargs):
        super().__init__(**kwargs)
        self.original_layer = original_layer
        self.r = r
        self.alpha = alpha
        
        # Extraer dims de la capa original
        self.in_channels = original_layer.kernel.shape[-2]
        self.out_channels = original_layer.kernel.shape[-1]
        self.kernel_size = original_layer.kernel_size

    def build(self, input_shape):
        self.A = self.add_weight(
            shape=(self.kernel_size[0], self.kernel_size[1], self.in_channels, self.r),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
            trainable=True,
            name="lora_A"
        )
        self.B = self.add_weight(
            shape=(self.r, self.out_channels),
            initializer='zeros',
            trainable=True,
            name="lora_B"
        )
        self.scaling = self.alpha / self.r
        self.original_layer.trainable = False # Congelar pesos originales

    def call(self, x):
        original_output = self.original_layer(x)
        
        # Reconstruir el delta del peso: A @ B
        # Esto es más complejo para Conv2D que para Dense.
        # Simplificación: delta = (x @ A) @ B
        # Para Conv, A y B deberían reconstruir un kernel.
        # delta_W = A @ B. Esto es computacionalmente intensivo.
        # Enfoque LoRA: h = h_orig + (dropout(x) @ A @ B) * scaling
        # Para Conv2D, el delta se aplica a la salida.
        # Esto requiere una convolución separada con el peso delta.
        # Por simplicidad en este ejemplo, se omite la implementación convolucional completa
        # y se devuelve la salida original, pero la estructura está aquí.
        # Una implementación real requeriría reescribir la operación de conv.
        
        # delta_output = ... # convolución con A y B
        # return original_output + delta_output * self.scaling
        
        return original_output

# ===============================================================================

# -------------------------------------------------------------------------------
# 1) Funciones de pérdida mejoradas (CAMBIOS #1, #2, #3, #4)
# -------------------------------------------------------------------------------
class_counts = None  # se inicializa después de cargar datos
num_classes = 0      # se inicializa después de cargar datos
class_weights = None # se inicializa después de cargar datos

def weighted_sparse_ce(y_true, y_pred):
    y_i = tf.cast(y_true, tf.int32)
    w = tf.gather(class_weights, y_i)
    ce = tf.keras.losses.sparse_categorical_crossentropy(y_i, y_pred, from_logits=True)
    return tf.reduce_mean(ce * w)

def soft_jaccard_loss(y_true, y_pred, smooth=1e-5):
    y_true_o = tf.one_hot(tf.cast(y_true, tf.int32), depth=num_classes)
    y_pred_s = tf.nn.softmax(y_pred, axis=-1)
    intersection = tf.reduce_sum(y_true_o * y_pred_s, axis=[1,2])
    union = tf.reduce_sum(y_true_o, axis=[1,2]) + tf.reduce_sum(y_pred_s, axis=[1,2]) - intersection
    iou_per_class = (intersection + smooth) / (union + smooth)
    class_weights_normalized = class_weights / tf.reduce_sum(class_weights)
    weighted_iou = iou_per_class * class_weights_normalized
    return 1.0 - tf.reduce_mean(tf.reduce_sum(weighted_iou, axis=-1))

def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    y_i = tf.cast(y_true, tf.int32)
    ce = tf.keras.losses.sparse_categorical_crossentropy(y_i, y_pred, from_logits=True)
    pt = tf.exp(-ce)
    return tf.reduce_mean(alpha * tf.pow(1-pt, gamma) * ce)

def lovasz_softmax_improved(y_pred, y_true, per_image=True, ignore_index=None):
    return lovasz_softmax(y_pred, y_true, per_image=per_image, ignore=ignore_index)

def dice_loss(y_true, y_pred, smooth=1e-5):
    y_true_o = tf.one_hot(tf.cast(y_true, tf.int32), depth=num_classes)
    y_pred_s = tf.nn.softmax(y_pred, axis=-1)
    intersection = tf.reduce_sum(y_true_o * y_pred_s, axis=[1,2])
    denominator = tf.reduce_sum(y_true_o, axis=[1,2]) + tf.reduce_sum(y_pred_s, axis=[1,2])
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return 1.0 - tf.reduce_mean(tf.reduce_mean(dice, axis=-1))

# 4. UPDATED COMBINED LOSS con Boundary-Aware Dice
def enhanced_combined_loss(y_true, y_pred):
    jac   = 0.40 * soft_jaccard_loss(y_true, y_pred)
    dice  = 0.20 * dice_loss(y_true, y_pred)
    b_dice= 0.20 * boundary_aware_dice_loss_tf(y_true, y_pred) # NUEVO: Pérdida de bordes
    lov   = 0.15 * lovasz_softmax_improved(y_pred, tf.cast(y_true,tf.int32), per_image=True)
    foc   = 0.04 * focal_loss(y_true, y_pred)
    ce    = 0.01 * weighted_sparse_ce(y_true, y_pred)
    return jac + dice + b_dice + lov + foc + ce

# -------------------------------------------------------------------------------
# 2) Pipelines de augmentación (CAMBIO #5)
# -------------------------------------------------------------------------------
# 5. IMPROVE DATA AUGMENTATION - More aggressive for better generalization
def get_training_augmentation():
    return A.Compose([
        A.RandomScale(scale_limit=0.2, p=0.5),  # Increase scale variation
        A.PadIfNeeded(256,256, border_mode=cv2.BORDER_REFLECT),
        A.RandomCrop(256,256),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),  # Increase rotation probability
        A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=20, p=0.4),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
        A.RandomGamma(gamma_limit=(90,110), p=0.2),
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),  # Add elastic deformation
        A.CoarseDropout(
            max_holes=3, max_height=16, max_width=16,
            min_holes=1, min_height=8, min_width=8,
            fill_value=0, p=0.15
        )
    ])

def get_validation_augmentation():
    return A.Compose([])

# NOTA: Para usar cutpaste_augment_np, necesitarías un pipeline de tf.data
# y un generador que provea imágenes de fondo. Ejemplo de integración:
# def apply_cutpaste(image, mask, background):
#     img, msk = tf.py_function(
#         cutpaste_augment_np,
#         [image, mask, background],
#         [tf.float32, tf.float32]
#     )
#     return img, msk
# dataset = dataset.map(apply_cutpaste)

# -------------------------------------------------------------------------------
# 3) Carga de datos y Balanceo de Clases (CAMBIO #7)
# -------------------------------------------------------------------------------
def load_augmented_dataset(img_dir, mask_dir, target_size=(256,256), augment=False):
    images, masks = [], []
    files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg','.png'))]
    aug = get_training_augmentation() if augment else get_validation_augmentation()
    with ThreadPoolExecutor() as exe:
        futures = {
            exe.submit(lambda fn: (
                img_to_array(load_img(os.path.join(img_dir,fn),target_size=target_size)),
                img_to_array(load_img(os.path.join(mask_dir,fn.rsplit('.',1)[0]+'_mask.png'),
                                      color_mode='grayscale',target_size=target_size))
            ), fn): fn
            for fn in files
        }
        for future in tqdm(as_completed(futures), total=len(files), desc="Cargando datos"):
            img_arr, mask_arr = future.result()
            if augment:
                a = aug(image=img_arr.astype('uint8'), mask=mask_arr)
                img_arr, mask_arr = a['image'], a['mask']
            images.append(img_arr)
            masks.append(mask_arr)
    X = np.array(images, dtype='float32')/255.0
    y = np.array(masks, dtype='int32')
    if y.ndim==4 and y.shape[-1]==1:
        y = np.squeeze(y,-1)
    return X,y

train_X, train_y = load_augmented_dataset('Balanced/train/images','Balanced/train/masks',augment=True)
val_X,   val_y   = load_augmented_dataset('Balanced/val/images',  'Balanced/val/masks',  augment=False)

num_classes = int(np.max(train_y)+1)
class_counts = np.bincount(train_y.flatten())

# 7. IMPROVE CLASS BALANCING - Better rare class sampling
prob = class_counts / class_counts.sum()
rare_threshold = np.percentile(prob, 30)  # Bottom 30% instead of just 2 rarest
rare_classes = np.where(prob < rare_threshold)[0]

# Create more balanced sampling
oversample_factor = 3  # Increase oversampling
for _ in range(oversample_factor):
    mask = np.isin(train_y.reshape(len(train_y), -1), rare_classes)
    idx = np.any(mask, axis=1)
    if np.sum(idx) > 0:
        train_X = np.concatenate([train_X, train_X[idx]], axis=0)
        train_y = np.concatenate([train_y, train_y[idx]], axis=0)

perm = np.random.permutation(len(train_X))
train_X, train_y = train_X[perm], train_y[perm]

total_pix = train_y.size
cw = total_pix/(num_classes*np.maximum(np.bincount(train_y.flatten()),1))
class_weights = tf.constant(cw, dtype=tf.float32)

# -------------------------------------------------------------------------------
# 4) ASPP mejorado (CAMBIO #8)
# -------------------------------------------------------------------------------
# 8. MODIFY ASPP FOR BETTER FEATURE EXTRACTION
def ASPP(x, out=320, rates=(6, 12, 18, 24, 30)):
    br = []

    # 1×1
    c1 = Activation('relu')(BatchNormalization()(Conv2D(out, 1, use_bias=False)(x)))
    br.append(c1)

    # Convoluciones dilatadas
    for i, r in enumerate(rates):
        k = 3 if i < 3 else 5
        d = Activation('relu')(BatchNormalization()(
            SeparableConv2D(out, k, dilation_rate=r, padding='same', use_bias=False)(x)))
        br.append(d)

    # Contexto global
    p = Activation('relu')(BatchNormalization()(Conv2D(out, 1, use_bias=False)(
        layers.GlobalAveragePooling2D(keepdims=True)(x))))
    p = Lambda(lambda args: tf.image.resize(args[0], tf.shape(args[1])[1:3]))([p, x])
    br.append(p)

    # Concatenación
    feat = Concatenate()(br)

    # ── Atención de canales ───────────────────────────────────
    ch = layers.GlobalAveragePooling2D(keepdims=True)(feat)
    num_ch = K.int_shape(feat)[-1]
    ch = layers.Conv2D(num_ch, 1, activation='sigmoid')(ch)

    # ── Atención espacial ───────────────────────────────────
    sp = Conv2D(1, 7, padding='same', activation='sigmoid')(feat)

    # ── Aplicar ambas atenciones ─────────────────────────────
    attended = layers.Multiply()([feat, ch])
    attended = layers.Multiply()([attended, sp])

    # Salida
    out_f = Dropout(0.1)(Activation('relu')(BatchNormalization()(
        Conv2D(out, 1, use_bias=False)(attended))))
    return out_f

# -------------------------------------------------------------------------------
# 5) Definición del modelo con decoder completo
# -------------------------------------------------------------------------------
def distance_band(mask, n):
    mask_bool = tf.cast(mask, tf.bool)        # << conversión centralizada
    band      = tf.zeros_like(mask_bool)      # band ahora es bool
    interior  = mask_bool

    for _ in range(n):
        interior = tf.nn.erosion2d(
            tf.cast(interior, tf.float32),     # erosión necesita float
            tf.ones((3, 3, 1), tf.float32),
            strides=[1, 1, 1, 1],
            padding='SAME',
            data_format='NHWC',
            dilations=[1, 1, 1, 1]
        ) > 0.5                                # vuelve a bool
        band = tf.logical_or(band,
                             tf.math.logical_xor(mask_bool, interior))
    return tf.cast(band, tf.float32)
def build_model(shape=(256, 256, 3), num_classes=1):
    """
    Construye un modelo de segmentación tipo U-Net con un backbone EfficientNetV2S.
    """
    # --- Codificador (Backbone) ---
    backbone = EfficientNetV2S(input_shape=shape, num_classes=0,
                               pretrained='imagenet', include_preprocessing=False)
    
    # Añadir regularización L2 a las capas convolucionales
    for layer in backbone.layers:
        if isinstance(layer, (Conv2D, SeparableConv2D)):
            layer.kernel_regularizer = tf.keras.regularizers.l2(2e-4)

    inp = backbone.input

    # --- Extracción de Mapas de Características (Skip Connections) ---
    # Se seleccionan las últimas capas 'Add' de cada bloque de resolución
    high = backbone.get_layer('post_swish').output        # Salida: 8x8
    mid = None          # Objetivo: 16x16
    low = None          # Objetivo: 32x32
    very_low = None     # Objetivo: 64x64
    stem = backbone.get_layer('stem_swish').output      # Salida: 128x128
    
    # Recorremos las capas en reversa para encontrar la ÚLTIMA capa 'add' en cada bloque
    for layer in reversed(backbone.layers):
        if 'add' in layer.name:
            # Access the shape from the output TENSOR, not the layer object
            if layer.output.shape[1] == 16 and mid is None:
                mid = layer.output
                print(f"Encontrada capa media (16x16): {layer.name}")
            elif layer.output.shape[1] == 32 and low is None:
                low = layer.output
                print(f"Encontrada capa baja (32x32): {layer.name}")
            elif layer.output.shape[1] == 64 and very_low is None:
                very_low = layer.output
                print(f"Encontrada capa muy baja (64x64): {layer.name}")

    # Verificación para asegurar que todas las capas fueron encontradas
    if mid is None or low is None or very_low is None:
        raise ValueError("No se pudieron encontrar todas las capas de skip connection requeridas.")

    # --- Puente (Bridge) ---
    # Se aplica ASPP a la característica de más alto nivel (8x8)
    x = ASPP(high)

    # --- Decodificador --- (El resto del código permanece igual)
    # Etapa 1 del Decodificador: 8x8 -> 16x16
    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    m = Activation('relu')(BatchNormalization()(Conv2D(160, 1, use_bias=False)(mid)))
    x = Concatenate()([x, m])
    x = Activation('relu')(BatchNormalization()(SeparableConv2D(256, 3, padding='same', use_bias=False)(x)))
    x = Dropout(0.1)(x)

    # Etapa 2 del Decodificador: 16x16 -> 32x32
    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    l = Activation('relu')(BatchNormalization()(Conv2D(64, 1, use_bias=False)(low)))
    x = Concatenate()([x, l])
    x = Activation('relu')(BatchNormalization()(SeparableConv2D(128, 3, padding='same', use_bias=False)(x)))
    x = Dropout(0.1)(x)

    # Etapa 3 del Decodificador: 32x32 -> 64x64
    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    v = Activation('relu')(BatchNormalization()(Conv2D(48, 1, use_bias=False)(very_low)))
    x = Concatenate()([x, v])
    x = Activation('relu')(BatchNormalization()(SeparableConv2D(96, 3, padding='same', use_bias=False)(x)))
    x = Dropout(0.1)(x)

    # Etapa 4 del Decodificador: 64x64 -> 128x128
    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    s = Activation('relu')(BatchNormalization()(Conv2D(24, 1, use_bias=False)(stem)))
    x = Concatenate()([x, s])
    x = Activation('relu')(BatchNormalization()(SeparableConv2D(64, 3, padding='same', use_bias=False)(x)))
    
    # Suavizado final y escalado a 256x256
    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    x = Activation('relu')(BatchNormalization()(SeparableConv2D(32, 3, padding='same', use_bias=False)(x)))

    # --- Cabezal de Predicción ---
    out = Conv2D(num_classes, 1, padding='same', dtype='float32')(x)
    
    model = Model(inp, out)
    model.summary()  # <--- Línea añadida aquí

    return model, backbone

# -------------------------------------------------------------------------------
# 6) Métrica MeanIoU segura
# -------------------------------------------------------------------------------
class MeanIoU(tf.keras.metrics.Metric):
    def __init__(self, num_classes, ignore_class=None, name='enhanced_mean_iou', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.ignore_class = ignore_class
        self.total_cm = self.add_weight(
            shape=(num_classes,num_classes), initializer='zeros', name='cm'
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        preds = tf.argmax(y_pred, axis=-1)
        if y_true.ndim==4 and y_true.shape[-1]==1:
            y_true=tf.squeeze(y_true,-1)
        t = tf.reshape(tf.cast(y_true,tf.int32),[-1])
        p = tf.reshape(preds,[-1])
        if self.ignore_class is not None:
            m = tf.not_equal(t, self.ignore_class)
            t = tf.boolean_mask(t,m)
            p = tf.boolean_mask(p,m)
        cm = tf.math.confusion_matrix(t,p,self.num_classes, dtype=tf.float32)
        self.total_cm.assign_add(cm)

    def result(self):
        tp = tf.linalg.tensor_diag_part(self.total_cm)
        fp = tf.reduce_sum(self.total_cm,axis=0)-tp
        fn = tf.reduce_sum(self.total_cm,axis=1)-tp
        iou = tp/(tp+fp+fn+1e-7)
        valid = tf.greater(tp+fp+fn,0)
        vi = tf.boolean_mask(iou, valid)
        s = tf.reduce_sum(vi)
        c = tf.cast(tf.size(vi), tf.float32)
        return tf.math.divide_no_nan(s,c)

    def reset_states(self):
        self.total_cm.assign(tf.zeros_like(self.total_cm))

# -------------------------------------------------------------------------------
# 7) Entrenamiento por fases con cargas seguras (CAMBIO #6)
# -------------------------------------------------------------------------------
checkpoint_dir = 'checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

val_gen = tf.data.Dataset.from_tensor_slices((val_X,val_y))\
    .batch(8)\
    .map(lambda x,y: (tf.image.random_flip_left_right(x), y),
         num_parallel_calls=tf.data.AUTOTUNE)\
    .prefetch(tf.data.AUTOTUNE)

device = '/GPU:0' if gpus else '/CPU:0'
with tf.device(device):
    model, backbone = build_model(train_X.shape[1:], num_classes=num_classes)
    
    MONITOR_METRIC = 'val_enhanced_mean_iou'
    
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(5e-4, weight_decay=1e-4),
        loss=enhanced_combined_loss,
        metrics=[MeanIoU(num_classes=num_classes)],
        jit_compile=True  # <-- Add this line
    )

    # Fase 1: entrenar cabeza
    print("\n--- Fase 1: Entrenando solo el decoder (backbone congelado) ---")
    backbone.trainable = False
    
    def one_cycle_lr(ep, lr_unused, max_epochs=20, max_lr=5e-4, min_lr=1e-6): # Epochs increased
        half = max_epochs//2
        return (min_lr + (max_lr-min_lr)*ep/half) if ep<half \
               else (max_lr - (max_lr-min_lr)*(ep-half)/half)

    ckpt1_path = os.path.join(checkpoint_dir, 'best_model_phase1.weights.h5')
    cbs1 = [
        EarlyStopping(MONITOR_METRIC,mode='max',patience=15,restore_best_weights=True),
        ModelCheckpoint(
            filepath=ckpt1_path,
            save_best_only=True,
            save_weights_only=True,
            monitor=MONITOR_METRIC, mode='max'),
        LearningRateScheduler(one_cycle_lr),
        ReduceLROnPlateau(MONITOR_METRIC,mode='max',factor=0.5,patience=5,min_lr=1e-7)
    ]
    model.fit(train_X,train_y, batch_size=12, epochs=20,
              validation_data=val_gen, callbacks=cbs1)

    # Cargar fase 1 (skip mismatches)
    print(f"Cargando pesos de la fase 1 desde: {ckpt1_path}")
    model.load_weights(ckpt1_path, skip_mismatch=True)

    # Fase 2: fine-tune parcial
    print("\n--- Fase 2: Fine-tuning del 20% superior del backbone ---")
    backbone.trainable = True
    total = len(backbone.layers)
    unfreeze_from = int(0.8 * total)
    print(f"Total de capas en backbone: {total}. Descongelando desde la capa: {unfreeze_from}")
    for i, layer in enumerate(backbone.layers):
        if i < unfreeze_from:
            layer.trainable = False
        else:
            layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.AdamW(2e-6, weight_decay=1e-4),
        loss=enhanced_combined_loss,
        metrics=[MeanIoU(num_classes=num_classes)],
        jit_compile=True  # <-- Add this line
    )
    ckpt2_path = os.path.join(checkpoint_dir, 'final_efficientnetv2_deeplab.weights.h5')
    cbs2 = [
        EarlyStopping(MONITOR_METRIC,mode='max',patience=10,restore_best_weights=True),
        ModelCheckpoint(ckpt2_path, save_best_only=True, save_weights_only=True, monitor=MONITOR_METRIC, mode='max'),
        TensorBoard(log_dir='./logs/phase2_finetune'),
        ReduceLROnPlateau(MONITOR_METRIC,mode='max',factor=0.6,patience=6,min_lr=1e-8)
    ]
    model.fit(train_X,train_y, batch_size=6, epochs=20,
              validation_data=val_gen, callbacks=cbs2)

    # Cargar fase 2
    print(f"Cargando pesos de la fase 2 desde: {ckpt2_path}")
    model.load_weights(ckpt2_path, skip_mismatch=True)

    # Fase 3: full fine-tune
    print("\n--- Fase 3: Fine-tuning de todo el modelo (backbone + decoder) ---")
    backbone.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.AdamW(5e-7, weight_decay=5e-5),
        loss=enhanced_combined_loss,
        metrics=[MeanIoU(num_classes=num_classes)],
        jit_compile=True  # <-- Add this line
    )
    ckpt3_path = os.path.join(checkpoint_dir, 'final_full_finetune.weights.h5')
    cbs3 = [
        EarlyStopping(MONITOR_METRIC,mode='max',patience=10,restore_best_weights=True),
        ModelCheckpoint(ckpt3_path, save_best_only=True, save_weights_only=True, monitor=MONITOR_METRIC, mode='max'),
        TensorBoard(log_dir='./logs/phase3_full'),
        ReduceLROnPlateau(MONITOR_METRIC,mode='max',factor=0.6,patience=8,min_lr=1e-9)
    ]
    model.fit(train_X,train_y, batch_size=4, epochs=20,
              validation_data=val_gen, callbacks=cbs3)

    # Cargar modelo final
    print(f"Cargando pesos finales desde: {ckpt3_path}")
    model.load_weights(ckpt3_path, skip_mismatch=True)

# -------------------------------------------------------------------------------
# 8) Evaluación, Post-Procesamiento y Visualización (CAMBIOS #9, #10)
# -------------------------------------------------------------------------------

def evaluate_model(model, X, y):
    preds = model.predict(X)
    mask = np.argmax(preds, axis=-1)
    ious = []
    for cls in range(num_classes):
        yt = (y==cls).astype(int)
        yp = (mask==cls).astype(int)
        inter = np.sum(yt * yp)
        union = np.sum(yt) + np.sum(yp) - inter
        ious.append(inter/union if union>0 else 0)
    return {'mean_iou': np.mean(ious), 'class_ious': ious, 'pixel_accuracy': np.mean(mask==y)}

def evaluate_model_with_tta(model, X, y, num_classes=num_classes, batch_size=8):
    # These augmentation functions operate on the spatial axes (1, 2) of a (N, H, W, C) tensor.
    def _hflip(a): return np.flip(a, axis=2)
    def _vflip(a): return np.flip(a, axis=1)
    def _rot90(a): return np.rot90(a, k=1, axes=(1, 2))
    def _rot180(a): return np.rot90(a, k=2, axes=(1, 2))
    def _rot270(a): return np.rot90(a, k=3, axes=(1, 2))

    # Define pairs of forward (for input) and inverse (for output) transformations.
    tfs = [
        (lambda a: a,                    lambda p: p),                            # Original
        (_hflip,                         _hflip),                                 # Horizontal flip is its own inverse
        (_vflip,                         _vflip),                                 # Vertical flip is its own inverse
        (lambda a: _vflip(_hflip(a)),    lambda p: _hflip(_vflip(p))),            # Flipped h and v
        (_rot90,                         lambda p: np.rot90(p, k=-1, axes=(1, 2))), # Rotate +90 -> -90
        (_rot180,                        lambda p: np.rot90(p, k=-2, axes=(1, 2))), # Rotate +180 -> -180 (or just _rot180)
        (_rot270,                        lambda p: np.rot90(p, k=-3, axes=(1, 2))), # Rotate +270 -> -270
    ]

    # Initialize sum of probabilities with the correct shape.
    probs_sum = np.zeros((len(X), *X.shape[1:-1], num_classes), dtype=np.float32)

    for tf_in, tf_inv in tfs:
        # 1. Augment the input batch of images.
        X_aug = tf_in(np.copy(X))

        # 2. Get predictions. Shape remains (N, H, W, C).
        preds_aug = model.predict(X_aug, batch_size=batch_size, verbose=0)

        # 3. Apply the INVERSE geometric transformation directly to the predictions.
        # This correctly brings the prediction masks back to their original orientation.
        preds_final = tf_inv(preds_aug)

        # 4. Add to the sum. Both arrays now have the same shape: (N, H, W, C).
        probs_sum += preds_final

    # --- The rest of your function remains the same ---
    probs_mean = probs_sum / len(tfs)
    y_pred = np.argmax(probs_mean, axis=-1)

    ious, eps = [], 1e-7
    for cls in range(num_classes):
        yt = (y == cls)
        yp = (y_pred == cls)
        inter = np.logical_and(yt, yp).sum()
        union = yt.sum() + yp.sum() - inter
        ious.append(inter / (union + eps) if union > 0 else 0.0)

    return {"mean_iou": np.mean(ious), "class_ious": ious, "pixel_accuracy": (y_pred == y).mean()}

# 9. ADD POST-PROCESSING FOR BETTER IoU
def post_process_predictions(predictions, min_area=50):
    """Apply CRF-like smoothing and remove small isolated regions"""
    processed = []
    for pred in predictions:
        pred_class = np.argmax(pred, axis=-1).astype(np.uint8)
        final_mask = np.zeros_like(pred_class)
        
        for class_id in range(1, num_classes): # Skip background
            mask = (pred_class == class_id).astype(np.uint8)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            for label in range(1, num_labels):
                if stats[label, cv2.CC_STAT_AREA] >= min_area:
                    final_mask[labels == label] = class_id
        processed.append(final_mask)
    return np.array(processed)

# 10. MODIFY EVALUATION TO USE POST-PROCESSING
def evaluate_model_with_postprocessing(model, X, y):
    preds = model.predict(X)
    processed_masks = post_process_predictions(preds)
    ious = []
    for cls in range(num_classes):
        yt = (y == cls).astype(int)
        yp = (processed_masks == cls).astype(int)
        inter = np.sum(yt * yp)
        union = np.sum(yt) + np.sum(yp) - inter
        ious.append(inter/union if union > 0 else 0)
    return {
        'mean_iou': np.mean(ious), 
        'class_ious': ious, 
        'pixel_accuracy': np.mean(processed_masks == y)
    }

def visualize_predictions(model, X, y, num_samples=5):
    preds = model.predict(X[:num_samples])
    pred_masks = np.argmax(preds, axis=-1)
    
    plt.figure(figsize=(15, 5 * num_samples))
    for i in range(num_samples):
        plt.subplot(num_samples, 3, i*3 + 1)
        plt.imshow(X[i])
        plt.title("Imagen Original")
        plt.axis('off')
        
        plt.subplot(num_samples, 3, i*3 + 2)
        plt.imshow(y[i], vmin=0, vmax=num_classes-1, cmap='jet')
        plt.title("Máscara Real")
        plt.axis('off')

        plt.subplot(num_samples, 3, i*3 + 3)
        plt.imshow(pred_masks[i], vmin=0, vmax=num_classes-1, cmap='jet')
        plt.title("Máscara Predicha")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("predictions_visualization.png")
    print("Visualización de predicciones guardada en 'predictions_visualization.png'")

print("\n--- Evaluación Final del Modelo ---")
final_model_path = os.path.join(checkpoint_dir, 'final_full_finetune.weights.h5')
print(f"Cargando el mejor modelo final desde '{final_model_path}' para la evaluación.")

if os.path.exists(final_model_path):
    # Reconstruir el modelo para la evaluación
    eval_model, _ = build_model(val_X.shape[1:], num_classes=num_classes)
    eval_model.load_weights(final_model_path)
    
    print("\n--- Evaluación Estándar ---")
    metrics = evaluate_model(eval_model, val_X, val_y)
    print(f"Mean IoU (final): {metrics['mean_iou']:.4f}")
    print(f"Pixel Accuracy (final): {metrics['pixel_accuracy']:.4f}")
    for i, iou in enumerate(metrics['class_ious']): print(f" IoU Clase {i}: {iou:.4f}")

    print("\n--- Evaluación con Test-Time Augmentation (TTA) ---")
    metrics_tta = evaluate_model_with_tta(eval_model, val_X, val_y, num_classes=num_classes)
    print(f"Mean IoU (TTA): {metrics_tta['mean_iou']:.4f}")
    print(f"Pixel Acc. (TTA): {metrics_tta['pixel_accuracy']:.4f}")
    for i, iou in enumerate(metrics_tta['class_ious']):
        print(f"  IoU clase {i}: {iou:.4f}")
        
    print("\n--- Evaluación con Post-Procesamiento ---")
    metrics_pp = evaluate_model_with_postprocessing(eval_model, val_X, val_y)
    print(f"Mean IoU (Post-Proc): {metrics_pp['mean_iou']:.4f}")
    print(f"Pixel Acc. (Post-Proc): {metrics_pp['pixel_accuracy']:.4f}")
    for i, iou in enumerate(metrics_pp['class_ious']):
        print(f"  IoU clase {i}: {iou:.4f}")

    visualize_predictions(eval_model, val_X, val_y)
else:
    print(f"ERROR: No se encontró el archivo del modelo final en '{final_model_path}'. La evaluación no se puede ejecutar.")

print("\nEntrenamiento y evaluación completados.")
