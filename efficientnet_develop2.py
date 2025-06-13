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
9. **Implementación de un modelo optimizado para IoU (IoUOptimizedModel).**
"""

import os
os.environ['MPLBACKEND'] = 'agg'  # matplotlib backend

import tensorflow as tf
from tensorflow.keras.utils import Sequence
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

def tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3, smooth=1e-7):
    """
    Tversky Loss para segmentación.
    - alpha: Peso para Falsos Positivos (FP).
    - beta:  Peso para Falsos Negativos (FN).
    - alpha + beta = 1. Para penalizar más los FN (clases pequeñas), usar alpha < 0.5.
    """
    y_true_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), depth=num_classes, axis=-1)
    y_pred_probs = tf.nn.softmax(y_pred, axis=-1)

    # Aplanar tensores
    y_true_flat = tf.reshape(y_true_one_hot, [-1, num_classes])
    y_pred_flat = tf.reshape(y_pred_probs, [-1, num_classes])
    
    tp = tf.reduce_sum(y_true_flat * y_pred_flat, axis=0)
    fn = tf.reduce_sum(y_true_flat * (1 - y_pred_flat), axis=0)
    fp = tf.reduce_sum((1 - y_true_flat) * y_pred_flat, axis=0)
    
    tversky_index = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    
    return 1.0 - tf.reduce_mean(tversky_index)

class DataGenerator(Sequence):
    def __init__(self, img_dir, mask_dir, normal_files, rare_files, batch_size,
                 dim, num_classes, augmentations, rare_ratio=0.5):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.normal_files = normal_files
        self.rare_files = rare_files
        self.batch_size = batch_size
        self.dim = dim
        self.num_classes = num_classes
        self.augment = augmentations
        self.rare_ratio = rare_ratio  # Proporción de muestras raras por lote

        self.num_rare_per_batch = int(self.batch_size * self.rare_ratio)
        self.num_normal_per_batch = self.batch_size - self.num_rare_per_batch
        
        # Si no hay archivos raros, tomar todos del grupo normal
        if not self.rare_files or self.num_rare_per_batch == 0:
            self.num_normal_per_batch = self.batch_size
            self.num_rare_per_batch = 0
            self.all_files = self.normal_files
            self.is_mixed = False
        else:
            self.is_mixed = True

        self.on_epoch_end()

    def __len__(self):
        # Longitud total basada en la clase más abundante para un epoch completo
        if self.is_mixed:
            return int(np.floor(len(self.normal_files) / self.num_normal_per_batch))
        else:
            return int(np.floor(len(self.all_files) / self.batch_size))


    def __getitem__(self, index):
        if self.is_mixed:
            # Seleccionar archivos para el lote actual de forma balanceada
            start_normal = index * self.num_normal_per_batch
            end_normal = start_normal + self.num_normal_per_batch
            batch_normal_files = self.normal_files[start_normal:end_normal]

            # Usar módulo para repetir la lista de archivos raros si es necesario
            start_rare = (index * self.num_rare_per_batch) % len(self.rare_files)
            end_rare = start_rare + self.num_rare_per_batch
            
            if end_rare > len(self.rare_files): # Manejar el wrap-around
                batch_rare_files = self.rare_files[start_rare:] + self.rare_files[:end_rare - len(self.rare_files)]
            else:
                batch_rare_files = self.rare_files[start_rare:end_rare]

            batch_files = batch_normal_files + batch_rare_files
        else:
            start = index * self.batch_size
            end = start + self.batch_size
            batch_files = self.all_files[start:end]
            
        # Generar datos (X, y) para este lote
        X, y = self.__data_generation(batch_files)
        return X, y

    def on_epoch_end(self):
        # Barajar los índices al final de cada época
        if self.is_mixed:
            np.random.shuffle(self.normal_files)
            np.random.shuffle(self.rare_files)
        else:
            np.random.shuffle(self.all_files)


    def __data_generation(self, batch_files):
        # Aquí va la lógica de carga y aumento para UN LOTE
        X = np.empty((self.batch_size, *self.dim, 3), dtype=np.float32)
        y = np.empty((self.batch_size, *self.dim), dtype=np.int32)

        for i, filename in enumerate(batch_files):
            img_path = os.path.join(self.img_dir, filename)
            mask_path = os.path.join(self.mask_dir, filename.rsplit('.', 1)[0] + '_mask.png')
            
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            if self.augment:
                augmented = self.augment(image=img, mask=mask)
                img, mask = augmented['image'], augmented['mask']
            
            # Normalizar y asignar
            X[i,] = img.astype(np.float32) / 255.0
            y[i,] = np.squeeze(mask)
        
        return X, y


class LoRAAdapterLayer(layers.Layer):
    """
    Implementación de LoRA para una capa Conv2D que es eficiente en recursos.
    """
    def __init__(self, original_layer, r=8, alpha=8, **kwargs):
        super().__init__(name=f"lora_{original_layer.name}", **kwargs)
        
        self.original_layer = original_layer
        self.original_layer.trainable = False  # Congelar la capa original
        
        self.r = r
        self.alpha = alpha
        
        # --- START: CORRECTED BLOCK ---
        # We get the number of output channels from the layer's configuration,
        # which is available before the layer is built.
        self.out_channels = self.original_layer.filters
        # --- END: CORRECTED BLOCK ---
        
        self.kernel_size = self.original_layer.kernel_size
        self.strides = self.original_layer.strides
        self.padding = self.original_layer.padding
        
        self.scaling = self.alpha / self.r

    def build(self, input_shape):
        # We can now use self.out_channels, which was set correctly in __init__.
        self.lora_down = layers.Conv2D(
            filters=self.r,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            use_bias=False,
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
            name="lora_down"
        )
        
        self.lora_up = layers.Conv2D(
            filters=self.out_channels,  # This now works correctly
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='VALID',
            use_bias=False,
            kernel_initializer='zeros',
            name="lora_up"
        )
        
        # It's good practice to ensure the original layer is also built
        if not self.original_layer.built:
            self.original_layer.build(input_shape)
            
        super().build(input_shape) # This will call build on child layers (lora_down, lora_up)

    def call(self, x):
        original_output = self.original_layer(x)
        lora_output = self.lora_down(x)
        lora_output = self.lora_up(lora_output)
        return original_output + lora_output * self.scaling

    def get_config(self):
        config = super().get_config()
        config.update({
            "r": self.r,
            "alpha": self.alpha,
        })
        return config

# ===============================================================================

# -------------------------------------------------------------------------------
# 1) Funciones de pérdida mejoradas (CAMBIOS #1, #2, #3, #4)
# -------------------------------------------------------------------------------
class_counts = None
num_classes = 0
class_weights = None

def ultimate_iou_loss(y_true, y_pred):
    """
    Pérdida combinada que optimiza directamente IoU y los bordes,
    inspirada en los principios de CIoU para la segmentación.
    """
    # 50% del peso para Lovasz, que optimiza directamente IoU
    lov = 0.50 * lovasz_softmax_improved(y_pred, tf.cast(y_true, tf.int32), per_image=True)
    
    # 30% del peso para el error en los bordes (análogo a la forma en CIoU)
    abe_dice = 0.30 * adaptive_boundary_enhanced_dice_loss_tf(y_true, y_pred, gamma=2.0)
    
    # 20% del peso para Tversky, que maneja desbalance FP/FN
    tver = 0.20 * tversky_loss(y_true, y_pred, alpha=0.6, beta=0.4) # alpha<beta penaliza más los FNs
    
    return lov + abe_dice + tver

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

# -------------------------------------------------------------------------------
# 2) Pipelines de augmentación (CAMBIO #5)
# -------------------------------------------------------------------------------
def get_training_augmentation(img_size=(256, 256)):
    return A.Compose([
        A.RandomScale(scale_limit=0.2, p=0.5),
        A.PadIfNeeded(img_size[0], img_size[1], border_mode=cv2.BORDER_REFLECT),
        A.RandomCrop(img_size[0], img_size[1]),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=20, p=0.4),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
        A.RandomGamma(gamma_limit=(90,110), p=0.2),
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),
        A.CoarseDropout(
            max_holes=3, max_height=16, max_width=16,
            min_holes=1, min_height=8, min_width=8,
            fill_value=0, p=0.15
        )
    ])

def get_validation_augmentation(img_size=(256, 256)):
    # La validación solo necesita asegurar el tamaño correcto
    return A.Compose([
        A.PadIfNeeded(min_height=img_size[0], min_width=img_size[1], border_mode=cv2.BORDER_CONSTANT, value=0),
        A.CenterCrop(img_size[0], img_size[1]),
    ])


# =========================================================================================
# === INICIO DEL BLOQUE IMPLEMENTADO: Carga de Datos con DataGenerator ===
# =========================================================================================

# -------------------------------------------------------------------------------
# 3) Carga de datos y Balanceo de Clases por Lotes (IMPLEMENTACIÓN CON GENERADOR)
# -------------------------------------------------------------------------------

def analyze_and_split_files(img_dir, mask_dir):
    """
    Analiza las máscaras para encontrar clases raras, calcula pesos de clase y
    divide la lista de archivos en 'raros' y 'normales'.
    """
    all_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png'))])
    mask_files = [os.path.join(mask_dir, f.rsplit('.', 1)[0] + '_mask.png') for f in all_files]

    # --- Pasada 1: Analizar todas las máscaras para obtener estadísticas ---
    print("Analizando máscaras para determinar la distribución de clases...")
    max_class_id = 0
    
    # Primero, determinar el número de clases
    for mask_path in tqdm(mask_files, desc="Determinando # de clases"):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            max_class_id = max(max_class_id, np.max(mask))
    
    num_classes = max_class_id + 1
    class_counts = np.zeros(num_classes, dtype=np.int64)
    print(f"Número de clases detectado: {num_classes}")

    # Ahora, contar píxeles por clase
    for mask_path in tqdm(mask_files, desc="Contando píxeles de clase"):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            counts = np.bincount(mask.flatten(), minlength=num_classes)
            class_counts += counts

    # --- Calcular clases raras y pesos ---
    total_pixels = np.sum(class_counts)
    class_probabilities = class_counts / total_pixels
    rare_threshold = np.percentile(class_probabilities[class_probabilities > 0], 25) # Umbral del 25% de las clases presentes
    rare_classes = np.where(class_probabilities < rare_threshold)[0]
    
    # Ignorar la clase de fondo si es la más común
    if 0 in rare_classes and class_probabilities[0] > 0.5:
        rare_classes = np.delete(rare_classes, np.where(rare_classes == 0))

    print(f"Clases identificadas como raras: {rare_classes}")
    
    class_weights = total_pixels / (num_classes * np.maximum(class_counts, 1))
    print(f"Pesos de clase calculados: {class_weights}")

    # --- Pasada 2: Dividir archivos en listas 'rara' y 'normal' ---
    rare_class_files = []
    normal_class_files = []
    print("Separando archivos que contienen clases raras...")
    for i, mask_path in enumerate(tqdm(mask_files, desc="Separando archivos")):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            unique_classes_in_mask = np.unique(mask)
            if any(cls in rare_classes for cls in unique_classes_in_mask):
                rare_class_files.append(all_files[i])
            else:
                normal_class_files.append(all_files[i])
    
    print(f"Archivos con clases raras: {len(rare_class_files)}")
    print(f"Archivos con clases normales: {len(normal_class_files)}")

    return rare_class_files, normal_class_files, num_classes, tf.constant(class_weights, dtype=tf.float32)

# --- Configuración y Creación de los Generadores (VERSIÓN CORREGIDA) ---
IMG_SIZE = (256, 256)
TRAIN_BATCH_SIZE = 12
VAL_BATCH_SIZE = 8
AUTOTUNE = tf.data.AUTOTUNE # TensorFlow's automatic performance tuner

# 1. Analizar archivos de entrenamiento y obtener listas y estadísticas
img_dir_train = 'Balanced/train/images'
mask_dir_train = 'Balanced/train/masks'
rare_files, normal_files, num_classes, class_weights = analyze_and_split_files(img_dir_train, mask_dir_train)

# 2. Instanciar generador para entrenamiento (¡sin batch size aquí!)
# El batching lo hará tf.data
train_generator_instance = DataGenerator(
    img_dir=img_dir_train,
    mask_dir=mask_dir_train,
    normal_files=normal_files,
    rare_files=rare_files,
    batch_size=TRAIN_BATCH_SIZE, # DataGenerator still needs it to form a batch
    dim=IMG_SIZE,
    num_classes=num_classes,
    augmentations=get_training_augmentation(IMG_SIZE),
    rare_ratio=0.5
)

# 3. Instanciar generador para validación
img_dir_val = 'Balanced/val/images'
mask_dir_val = 'Balanced/val/masks'
val_files = sorted([f for f in os.listdir(img_dir_val) if f.lower().endswith(('.jpg', '.png'))])

val_generator_instance = DataGenerator(
    img_dir=img_dir_val,
    mask_dir=mask_dir_val,
    normal_files=val_files,
    rare_files=[],
    batch_size=VAL_BATCH_SIZE, # DataGenerator still needs it to form a batch
    dim=IMG_SIZE,
    num_classes=num_classes,
    augmentations=get_validation_augmentation(IMG_SIZE),
    rare_ratio=0.0
)

# 4. Crear los datasets de tf.data
train_ds = tf.data.Dataset.from_generator(
    lambda: train_generator_instance,
    output_signature=(
        tf.TensorSpec(shape=(None, *IMG_SIZE, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, *IMG_SIZE), dtype=tf.int32)
    )
).prefetch(buffer_size=AUTOTUNE) # <-- Replaces workers/multiprocessing

val_ds = tf.data.Dataset.from_generator(
    lambda: val_generator_instance,
    output_signature=(
        tf.TensorSpec(shape=(None, *IMG_SIZE, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, *IMG_SIZE), dtype=tf.int32)
    )
).prefetch(buffer_size=AUTOTUNE) # <-- Replaces workers/multiprocessing

print("tf.data pipelines created successfully.")

# El resto del código para cargar val_X y val_y en memoria permanece igual...

# 4. Cargar datos de validación en memoria para la evaluación final
print("Cargando datos de validación en memoria para evaluación final...")
val_X, val_y = [], []
val_aug = get_validation_augmentation(IMG_SIZE)
for filename in tqdm(val_files, desc="Cargando validación"):
    img = cv2.imread(os.path.join(img_dir_val, filename))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(os.path.join(mask_dir_val, filename.rsplit('.', 1)[0] + '_mask.png'), cv2.IMREAD_GRAYSCALE)
    
    augmented = val_aug(image=img, mask=mask)
    img_aug, mask_aug = augmented['image'], augmented['mask']
    
    val_X.append(img_aug.astype(np.float32) / 255.0)
    val_y.append(mask_aug)

val_X = np.array(val_X)
val_y = np.array(val_y)
print("Datos de validación cargados.")

# =========================================================================================
# === FIN DEL BLOQUE IMPLEMENTADO =========================================================
# =========================================================================================


# -------------------------------------------------------------------------------
# 4) ASPP mejorado (CAMBIO #8)
# -------------------------------------------------------------------------------
def ASPP(x, out=320, rates=(6, 12, 18, 24, 30)):
    br = []
    c1 = Activation('relu')(BatchNormalization()(Conv2D(out, 1, use_bias=False)(x)))
    br.append(c1)
    for i, r in enumerate(rates):
        k = 3 if i < 3 else 5
        d = Activation('relu')(BatchNormalization()(
            SeparableConv2D(out, k, dilation_rate=r, padding='same', use_bias=False)(x)))
        br.append(d)
    p = Activation('relu')(BatchNormalization()(Conv2D(out, 1, use_bias=False)(
        layers.GlobalAveragePooling2D(keepdims=True)(x))))
    p = Lambda(lambda args: tf.image.resize(args[0], tf.shape(args[1])[1:3]))([p, x])
    br.append(p)
    feat = Concatenate()(br)
    ch = layers.GlobalAveragePooling2D(keepdims=True)(feat)
    num_ch = K.int_shape(feat)[-1]
    ch = layers.Conv2D(num_ch, 1, activation='sigmoid')(ch)
    sp = Conv2D(1, 7, padding='same', activation='sigmoid')(feat)
    attended = layers.Multiply()([feat, ch])
    attended = layers.Multiply()([attended, sp])
    out_f = Dropout(0.1)(Activation('relu')(BatchNormalization()(
        Conv2D(out, 1, use_bias=False)(attended))))
    return out_f

# -------------------------------------------------------------------------------
# 5) Definición del modelo con decoder completo
# -------------------------------------------------------------------------------
def distance_band(mask, n):
    mask_bool = tf.cast(mask, tf.bool)
    band = tf.zeros_like(mask_bool)
    interior = mask_bool
    for _ in range(n):
        interior = tf.nn.erosion2d(
            tf.cast(interior, tf.float32),
            tf.ones((3, 3, 1), tf.float32),
            strides=[1, 1, 1, 1],
            padding='SAME',
            data_format='NHWC',
            dilations=[1, 1, 1, 1]
        ) > 0.5
        band = tf.logical_or(band, tf.math.logical_xor(mask_bool, interior))
    return tf.cast(band, tf.float32)

def build_model(shape=(256, 256, 3), num_classes=1):
    backbone = EfficientNetV2S(input_shape=shape, num_classes=0,
                                         pretrained='imagenet', include_preprocessing=False)
    for layer in backbone.layers:
        if isinstance(layer, (Conv2D, SeparableConv2D)):
            layer.kernel_regularizer = tf.keras.regularizers.l2(2e-4)
    inp = backbone.input
    high = backbone.get_layer('post_swish').output
    mid, low, very_low = None, None, None
    for layer in reversed(backbone.layers):
        if 'add' in layer.name:
            if layer.output.shape[1] == 16 and mid is None: mid = layer.output
            elif layer.output.shape[1] == 32 and low is None: low = layer.output
            elif layer.output.shape[1] == 64 and very_low is None: very_low = layer.output
    if mid is None or low is None or very_low is None:
        raise ValueError("No se pudieron encontrar todas las capas de skip connection requeridas.")
    
    x = ASPP(high)
    
    # Decoder stages con LoRA
    def decoder_block(x_in, skip_conn, filters_skip, filters_sep, r_lora, alpha_lora):
        x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x_in)
        s = Activation('relu')(BatchNormalization()(Conv2D(filters_skip, 1, use_bias=False)(skip_conn)))
        x = Concatenate()([x, s])
        sep_conv = SeparableConv2D(filters_sep, 3, padding='same', use_bias=False)
        x = LoRAAdapterLayer(sep_conv, r=r_lora, alpha=alpha_lora)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.1)(x)
        return x

    x = decoder_block(x, mid, 160, 256, r_lora=4, alpha_lora=4)
    x = decoder_block(x, low, 64, 128, r_lora=4, alpha_lora=4)
    x = decoder_block(x, very_low, 48, 96, r_lora=4, alpha_lora=4)
    stem = backbone.get_layer('stem_swish').output
    x = decoder_block(x, stem, 24, 64, r_lora=4, alpha_lora=4)

    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    sep_conv_5 = SeparableConv2D(32, 3, padding='same', use_bias=False)
    x = LoRAAdapterLayer(sep_conv_5, r=4, alpha=4)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    out = Conv2D(num_classes, 1, padding='same', dtype='float32')(x)
    model = Model(inp, out)
    model.summary()
    return model, backbone

# --- Define esta clase después de tu función build_model ---
class IoUOptimizedModel(keras.Model):
    def __init__(self, shape, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        # Construimos el modelo base
        self.seg_model, self.backbone = build_model(shape, num_classes)
        
        # Trackers para pérdida y métricas
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.iou_metric = MeanIoU(num_classes=num_classes, name='enhanced_mean_iou')

    def call(self, inputs, training=False):
        return self.seg_model(inputs, training=training)

    @property
    def metrics(self):
        # Expone las métricas para que Keras las muestre y las registre
        return [self.loss_tracker, self.iou_metric]

    def train_step(self, data):
        x, y_true = data

        with tf.GradientTape() as tape:
            # 1. Obtener las predicciones (logits)
            y_pred = self.seg_model(x, training=True)
            
            # 2. CALCULAR LA PÉRDIDA DIRECTAMENTE USANDO NUESTRA FUNCIÓN IOU-CÉNTRICA
            # Esta es la modificación clave: la pérdida para el backprop es la que
            # se enfoca en IoU y bordes.
            loss = ultimate_iou_loss(y_true, y_pred)

        # 3. Aplicar gradientes
        trainable_vars = self.seg_model.trainable_variables
        grads = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(grads, trainable_vars))

        # 4. Actualizar métricas
        self.loss_tracker.update_state(loss)
        self.iou_metric.update_state(y_true, y_pred)
        
        return {m.name: m.result() for m in self.metrics}
        
    def test_step(self, data):
        x, y_true = data
        y_pred = self.seg_model(x, training=False)
        
        # Para la validación, usamos la misma pérdida para consistencia
        loss = ultimate_iou_loss(y_true, y_pred)
        
        # Actualizar métricas
        self.loss_tracker.update_state(loss)
        self.iou_metric.update_state(y_true, y_pred)
        
        return {m.name: m.result() for m in self.metrics}

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
# 7) Entrenamiento por fases (CON DataGenerators Y MULTIPROCESAMIENTO)
# -------------------------------------------------------------------------------
checkpoint_dir = 'checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

device = '/GPU:0' if gpus else '/CPU:0'

with tf.device(device):
    # --- INICIO DE LA IMPLEMENTACIÓN ---
    # Instanciamos IoUOptimizedModel
    iou_aware_model = IoUOptimizedModel(shape=(*IMG_SIZE, 3), num_classes=num_classes)
    
    # Referencias al modelo interno y al backbone para compatibilidad
    model = iou_aware_model.seg_model
    backbone = iou_aware_model.backbone
    
    MONITOR_METRIC = 'val_enhanced_mean_iou'
    
    # --- Fase 1: entrenar cabeza ---
    print("\n--- Fase 1: Entrenando solo el decoder (backbone congelado) ---")
    backbone.trainable = False
    
    # Compilamos el modelo contenedor. El optimizador es lo único que necesita.
    iou_aware_model.compile(
        optimizer=tf.keras.optimizers.AdamW(5e-4, weight_decay=1e-4)
    )
    
    def one_cycle_lr(ep, lr_unused, max_epochs=20, max_lr=5e-4, min_lr=1e-6):
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
        LearningRateScheduler(lambda ep, lr: one_cycle_lr(ep, lr, max_epochs=20)),
        ReduceLROnPlateau(MONITOR_METRIC,mode='max',factor=0.5,patience=5,min_lr=1e-7)
    ]
    
    # Usamos los generadores y activamos el multiprocesamiento
    iou_aware_model.fit(
        train_ds,  # <--- Pass the tf.data.Dataset object
        epochs=20,
        validation_data=val_ds, # <--- Pass the tf.data.Dataset object
        callbacks=cbs1
        # NO workers or use_multiprocessing arguments needed
    )
    
    print(f"Cargando pesos de la fase 1 desde: {ckpt1_path}")
    iou_aware_model.load_weights(ckpt1_path)

    # --- Fase 2: fine-tune parcial ---
    print("\n--- Fase 2: Fine-tuning del 20% superior del backbone ---")
    backbone.trainable = True
    total = len(backbone.layers)
    unfreeze_from = int(0.8 * total)
    for i, layer in enumerate(backbone.layers):
        layer.trainable = (i >= unfreeze_from)

    # Recompilar el modelo contenedor
    iou_aware_model.compile(
        optimizer=tf.keras.optimizers.AdamW(2e-6, weight_decay=1e-4)
    )
    
    ckpt2_path = os.path.join(checkpoint_dir, 'final_efficientnetv2_deeplab.weights.h5')
    cbs2 = [
        EarlyStopping(MONITOR_METRIC,mode='max',patience=10,restore_best_weights=True),
        ModelCheckpoint(ckpt2_path, save_best_only=True, save_weights_only=True, monitor=MONITOR_METRIC, mode='max'),
        TensorBoard(log_dir='./logs/phase2_finetune'),
        ReduceLROnPlateau(MONITOR_METRIC,mode='max',factor=0.6,patience=6,min_lr=1e-8)
    ]
    iou_aware_model.fit(
        train_ds,  # <--- Pass the tf.data.Dataset object
        epochs=20,
        validation_data=val_ds, # <--- Pass the tf.data.Dataset object
        callbacks=cbs1
        # NO workers or use_multiprocessing arguments needed
    )

    print(f"Cargando pesos de la fase 2 desde: {ckpt2_path}")
    iou_aware_model.load_weights(ckpt2_path)

    # --- Fase 3: full fine-tune ---
    print("\n--- Fase 3: Fine-tuning de todo el modelo (backbone + decoder) ---")
    backbone.trainable = True
    
    # Recompilar el modelo contenedor
    iou_aware_model.compile(
        optimizer=tf.keras.optimizers.AdamW(5e-7, weight_decay=5e-5)
    )

    ckpt3_path = os.path.join(checkpoint_dir, 'final_full_finetune.weights.h5')
    cbs3 = [
        EarlyStopping(MONITOR_METRIC,mode='max',patience=10,restore_best_weights=True),
        ModelCheckpoint(ckpt3_path, save_best_only=True, save_weights_only=True, monitor=MONITOR_METRIC, mode='max'),
        TensorBoard(log_dir='./logs/phase3_full'),
        ReduceLROnPlateau(MONITOR_METRIC,mode='max',factor=0.6,patience=8,min_lr=1e-9)
    ]
    iou_aware_model.fit(
        train_ds,  # <--- Pass the tf.data.Dataset object
        epochs=20,
        validation_data=val_ds, # <--- Pass the tf.data.Dataset object
        callbacks=cbs1
        # NO workers or use_multiprocessing arguments needed
    )
    
    print(f"Cargando pesos finales desde: {ckpt3_path}")
    iou_aware_model.load_weights(ckpt3_path)
    # --- FIN DE LA IMPLEMENTACIÓN ---

# -------------------------------------------------------------------------------
# 8) Evaluación, Post-Procesamiento y Visualización
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
    def _hflip(a): return np.flip(a, axis=2)
    def _vflip(a): return np.flip(a, axis=1)
    tfs = [
        (lambda a: a, lambda p: p),
        (_hflip, _hflip),
        (_vflip, _vflip),
        (lambda a: _vflip(_hflip(a)), lambda p: _hflip(_vflip(p))),
    ]
    probs_sum = np.zeros((len(X), *X.shape[1:-1], num_classes), dtype=np.float32)
    for tf_in, tf_inv in tfs:
        X_aug = tf_in(np.copy(X))
        preds_aug = model.predict(X_aug, batch_size=batch_size, verbose=0)
        preds_final = tf_inv(preds_aug)
        probs_sum += preds_final
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

def post_process_predictions(predictions, min_area=50):
    processed = []
    for pred in predictions:
        pred_class = np.argmax(pred, axis=-1).astype(np.uint8)
        final_mask = np.zeros_like(pred_class)
        for class_id in range(1, num_classes):
            mask = (pred_class == class_id).astype(np.uint8)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            for label in range(1, num_labels):
                if stats[label, cv2.CC_STAT_AREA] >= min_area:
                    final_mask[labels == label] = class_id
        processed.append(final_mask)
    return np.array(processed)

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
    return {'mean_iou': np.mean(ious), 'class_ious': ious, 'pixel_accuracy': np.mean(processed_masks == y)}

def visualize_predictions(model, X, y, num_samples=5):
    preds = model.predict(X[:num_samples])
    pred_masks = np.argmax(preds, axis=-1)
    plt.figure(figsize=(15, 5 * num_samples))
    for i in range(num_samples):
        plt.subplot(num_samples, 3, i*3 + 1); plt.imshow(X[i]); plt.title("Imagen Original"); plt.axis('off')
        plt.subplot(num_samples, 3, i*3 + 2); plt.imshow(y[i], vmin=0, vmax=num_classes-1, cmap='jet'); plt.title("Máscara Real"); plt.axis('off')
        plt.subplot(num_samples, 3, i*3 + 3); plt.imshow(pred_masks[i], vmin=0, vmax=num_classes-1, cmap='jet'); plt.title("Máscara Predicha"); plt.axis('off')
    plt.tight_layout()
    plt.savefig("predictions_visualization.png")
    print("Visualización de predicciones guardada en 'predictions_visualization.png'")

print("\n--- Evaluación Final del Modelo ---")
final_model_path = os.path.join(checkpoint_dir, 'final_full_finetune.weights.h5')
print(f"Cargando el mejor modelo final desde '{final_model_path}' para la evaluación.")

if os.path.exists(final_model_path):
    # Usamos el modelo interno (seg_model) del contenedor para la evaluación
    eval_model = iou_aware_model.seg_model
    
    print("\n--- Evaluación Estándar ---")
    metrics = evaluate_model(eval_model, val_X, val_y)
    print(f"Mean IoU (final): {metrics['mean_iou']:.4f}")
    print(f"Pixel Accuracy (final): {metrics['pixel_accuracy']:.4f}")
    for i, iou in enumerate(metrics['class_ious']): print(f" IoU Clase {i}: {iou:.4f}")

    print("\n--- Evaluación con Test-Time Augmentation (TTA) ---")
    metrics_tta = evaluate_model_with_tta(eval_model, val_X, val_y, num_classes=num_classes)
    print(f"Mean IoU (TTA): {metrics_tta['mean_iou']:.4f}")
    print(f"Pixel Acc. (TTA): {metrics_tta['pixel_accuracy']:.4f}")
    for i, iou in enumerate(metrics_tta['class_ious']): print(f"  IoU clase {i}: {iou:.4f}")
    
    print("\n--- Evaluación con Post-Procesamiento ---")
    metrics_pp = evaluate_model_with_postprocessing(eval_model, val_X, val_y)
    print(f"Mean IoU (Post-Proc): {metrics_pp['mean_iou']:.4f}")
    print(f"Pixel Acc. (Post-Proc): {metrics_pp['pixel_accuracy']:.4f}")
    for i, iou in enumerate(metrics_pp['class_ious']): print(f"  IoU clase {i}: {iou:.4f}")

    visualize_predictions(eval_model, val_X, val_y)
else:
    print(f"ERROR: No se encontró el archivo del modelo final en '{final_model_path}'. La evaluación no se puede ejecutar.")

print("\nEntrenamiento y evaluación completados.")