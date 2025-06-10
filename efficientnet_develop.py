# -*- coding: utf-8 -*-
"""
claude_efficientnetv2_deeplab.py
Script completo: carga datos, construye y entrena el modelo DeepLabV3+ usando EfficientNetV2S como backbone,
con monkey-patch de SE module, pérdida ponderada píxel-wise, métricas y visualización.

MODIFICADO para implementar las 5 optimizaciones de mIoU.
"""

# 1) Imports ------------------------------------------------------------------
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial # <-- CAMBIO 1: Importado para la pérdida

import numpy as np
import cv2
from tqdm import tqdm
import albumentations as A

import tensorflow as tf
from tensorflow.keras import layers, backend as K
from tensorflow.keras.layers import (
    Input, Conv2D, Conv2DTranspose, BatchNormalization,
    Activation, Dropout, Lambda, Concatenate, UpSampling2D, SeparableConv2D
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
)
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import tensorflow_addons as tfa
from lovasz_loss_tf import lovasz_hinge, lovasz_softmax

tf.keras.backend.clear_session()
gpus = tf.config.experimental.list_physical_devices('GPU')
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)
# CAMBIO 5: Usar mixed precision para un batch más grande / entrenamiento más rápido
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# -------------------------------------------------------------------------------
# Monkey-patch para usar un Lambda layer en SE module de keras_efficientnet_v2
# -------------------------------------------------------------------------------
import keras_efficientnet_v2.efficientnet_v2 as _efn2

def patched_se_module(inputs, se_ratio=0.25, name=""):
    """Reimplementa SE usando Lambda + reducción **DIVIDIDA** para evitar OOM."""
    data_format = K.image_data_format()
    h_axis, w_axis = (1, 2) if data_format == 'channels_last' else (2, 3)

    # Pooling espaciado en un Lambda para mantener trazabilidad
    se = Lambda(lambda x: tf.reduce_mean(x, axis=[h_axis, w_axis], keepdims=True),
                name=name + "se_reduce_pool")(inputs)

    channels = inputs.shape[-1]
    reduced_filters = max(1, int(channels / se_ratio))  # <–– divide, no multiplica

    se = Conv2D(reduced_filters, 1, padding='same', name=name + "se_reduce_conv")(se)
    se = Activation('swish', name=name + "se_reduce_act")(se)
    se = Conv2D(channels, 1, padding='same', name=name + "se_expand_conv")(se)
    se = Activation('sigmoid', name=name + "se_excite")(se)
    return layers.Multiply(name=name + "se_excite_mul")([inputs, se])

_efn2.se_module = patched_se_module  # sobrescribe la original

from keras_efficientnet_v2 import EfficientNetV2S

# 2) Augmentation pipelines --------------------------------------------------
def get_training_augmentation():
    # CAMBIO 3: Aumentar el tamaño del recorte semántico
    return A.Compose([
        A.RandomScale(scale_limit=0.2, p=0.5),
        A.PadIfNeeded(min_height=256, min_width=256, border_mode=cv2.BORDER_REFLECT),
        A.RandomCrop(height=256, width=256),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.Affine(translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)}, scale=(0.9, 1.1), rotate=(-15, 15), p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.RandomGamma(gamma_limit=(80, 120), p=0.2),
        A.GaussianBlur(blur_limit=3, p=0.1),
        A.GaussNoise(p=0.1),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.2),
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, min_holes=1, min_height=16, min_width=16, p=0.2),
    ])

def get_validation_augmentation():
    return A.Compose([]) # La validación no debe tener aumentos aleatorios

# 3) Data loading -------------------------------------------------------------
def load_augmented_dataset(img_dir, mask_dir, target_size=(192,192), augment=False): # <-- CAMBIO 3
    images, masks = [], []
    files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg','.png'))]
    aug = get_training_augmentation() if augment else get_validation_augmentation()

    with ThreadPoolExecutor() as exe:
        futures = {
            exe.submit(lambda fn: (
                img_to_array(load_img(os.path.join(img_dir, fn), target_size=target_size)),
                img_to_array(load_img(os.path.join(mask_dir, fn.rsplit('.',1)[0] + '_mask.png'),
                                      color_mode='grayscale', target_size=target_size))
            ), fn): fn for fn in files
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Loading data"):
            img_arr, mask_arr = future.result()
            if augment:
                augm = aug(image=img_arr.astype('uint8'), mask=mask_arr) # albumentations a menudo prefiere uint8
                img_arr, mask_arr = augm['image'], augm['mask']
            images.append(img_arr)
            masks.append(mask_arr)

    X = np.array(images, dtype='float32') / 255.0
    y = np.squeeze(np.array(masks, dtype='int32'), axis=-1)
    return X, y

# 4) Load / split into train & val -------------------------------------------
# Nota: target_size se establece por defecto a 256x256 en la función
train_X, train_y = load_augmented_dataset('Balanced/train/images', 'Balanced/train/masks', augment=True, target_size=(192, 192))
val_X, val_y = load_augmented_dataset('Balanced/val/images', 'Balanced/val/masks', augment=False, target_size=(192, 192))

# 5) Número de clases & modelo -----------------------------------------------
num_classes = int(np.max(train_y) + 1)

def improved_ASPP(x, out_channels=128, rates=(6, 12, 18, 24)):
    # 1x1 convolution branch
    conv1x1 = layers.Conv2D(out_channels, 1, padding='same', use_bias=False)(x)
    conv1x1 = layers.BatchNormalization()(conv1x1)
    conv1x1 = layers.Activation('relu')(conv1x1)
    branches = [conv1x1]
    # Atrous separable convolutions branches
    for r in rates:
        d = layers.SeparableConv2D(out_channels, 3, dilation_rate=r,
                                      padding='same', use_bias=False)(x)
        d = layers.BatchNormalization()(d)
        d = layers.Activation('relu')(d)
        branches.append(d)
    # Global average pooling branch
    pool = layers.GlobalAveragePooling2D(keepdims=True)(x)
    pool = layers.Conv2D(out_channels, 1, padding='same', use_bias=False)(pool)
    pool = layers.BatchNormalization()(pool)
    pool = layers.Activation('relu')(pool)
    pool = layers.Lambda(lambda args: tf.image.resize(args[0], tf.shape(args[1])[1:3]))([pool, x])
    branches.append(pool)
    # Concatenate and final convolution
    y = layers.Concatenate()(branches)
    y = layers.Conv2D(out_channels, 1, padding='same', use_bias=False)(y)
    y = layers.BatchNormalization()(y)
    y = layers.Dropout(0.3)(y)
    return layers.Activation('relu')(y)

# CAMBIO 2: Hacer que el decodificador genere bordes más nítidos
def build_model(input_shape=(256,256,3)): # <-- CAMBIO 3
    backbone = EfficientNetV2S(
        input_shape=input_shape,
        num_classes=0,
        pretrained='imagenet',
        include_preprocessing=False
    )
    inputs = backbone.input
    for l in backbone.layers:
        if isinstance(l, (Conv2D, SeparableConv2D)):
            l.kernel_regularizer = tf.keras.regularizers.l2(1e-4)

    # Puntos de conexión en keras-efficientnet-v2
    very_low = backbone.get_layer('stem_swish').output       # 64×64 (en 256x256) -> ahora 128x128
    low      = backbone.get_layer('add_2').output            # 32×32 -> ahora 64x64
    mid      = backbone.get_layer('add_6').output            # 16×16 -> ahora 32x32
    high     = backbone.get_layer('post_swish').output       # 4×4   -> ahora 8x8

    # ASPP sobre características de alto nivel
    x = improved_ASPP(high, rates=(6, 12, 18)) # Tasas reducidas para feature map más grande
    
    # --- DECODIFICADOR MEJORADO ---
    # Stage 1: 8→32
    x = layers.Conv2DTranspose(256, 4, strides=4, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.Activation('relu')(x)
    
    # Refinar características 'mid'
    mid_p = layers.Conv2D(128, 1, padding='same', use_bias=False)(mid)
    mid_p = layers.BatchNormalization()(mid_p); mid_p = layers.Activation('relu')(mid_p)
    
    x = layers.Concatenate()([x, mid_p])
    x = layers.SeparableConv2D(256, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.Activation('relu')(x)
    
    # Stage 2: 32→64
    x = layers.Conv2DTranspose(128, 2, strides=2, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.Activation('relu')(x)
    
    # Refinar características 'low'
    low_p = layers.Conv2D(64, 1, padding='same', use_bias=False)(low)
    low_p = layers.BatchNormalization()(low_p); low_p = layers.Activation('relu')(low_p)
    
    x = layers.Concatenate()([x, low_p])
    x = layers.SeparableConv2D(128, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.Activation('relu')(x)
    
    # -------- NUEVA ETAPA: 64→128 con skip extra superficial --------
    x = layers.Conv2DTranspose(96, 2, strides=2, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.Activation('relu')(x)

    # Procesar skip 'very_low'
    very_low_p = layers.Conv2D(48, 1, padding='same', use_bias=False)(very_low)
    very_low_p = layers.BatchNormalization()(very_low_p); very_low_p = layers.Activation('relu')(very_low_p)

    x = layers.Concatenate()([x, very_low_p])
    x = layers.SeparableConv2D(96, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.Activation('relu')(x)

    # Refinamiento final (128×128 → 256×256)
    x = layers.Conv2DTranspose(64, 2, strides=2, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.Activation('relu')(x)
    
    x = layers.SeparableConv2D(64, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.Activation('relu')(x)
    
    # Cabeza de clasificación con salida float32 para la pérdida (importante en mixed precision)
    outputs = layers.Conv2D(num_classes, 1, padding='same', dtype='float32')(x)
    
    return Model(inputs, outputs)

model = build_model(input_shape=train_X.shape[1:])

# 6) Pérdida, métricas --------------------------------------------------------
# CAMBIO 1: Optimizar para IoU en lugar de CE + Dice
class_counts = np.bincount(train_y.flatten())
total_pixels = class_counts.sum()
class_weights_np = total_pixels / (num_classes * np.maximum(class_counts,1))
class_weights = tf.constant(class_weights_np, dtype=tf.float32)

def weighted_sparse_ce(y_true, y_pred):
    y_true_i = tf.cast(y_true, tf.int32)
    w = tf.gather(class_weights, y_true_i)
    ce = tf.keras.losses.sparse_categorical_crossentropy(y_true_i, y_pred, from_logits=True)
    return tf.reduce_mean(ce * w)

# Nueva función de pérdida: Soft Jaccard
def soft_jaccard_loss(y_true, y_pred, smooth=1e-5):
    y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=num_classes)
    y_pred = tf.nn.softmax(y_pred, axis=-1)
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1,2])
    union        = tf.reduce_sum(y_true + y_pred, axis=[1,2]) - intersection
    jaccard      = (intersection + smooth) / (union + smooth)
    return 1.0 - tf.reduce_mean(jaccard)

# Nueva función de pérdida combinada
def combined_loss(y_true, y_pred):
    # Asegura que todo se calcule en float32
    y_pred = tf.cast(y_pred, tf.float32)
    y_true = tf.cast(y_true, tf.float32)  # o int32 según la parte

    ce  = 0.5 * weighted_sparse_ce(y_true, y_pred)
    jac = 0.2 * soft_jaccard_loss(y_true, y_pred)
    lov = 0.3 * lovasz_softmax(y_pred, tf.cast(y_true, tf.int32), per_image=False)
    return ce + jac + lov

class MeanIoU(tf.keras.metrics.Metric):
    def __init__(self, num_classes, name='mean_iou', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.total_cm = self.add_weight('total_confusion_matrix', shape=(num_classes,num_classes), initializer='zeros')
    def update_state(self, y_true, y_pred, sample_weight=None):
        preds = tf.argmax(y_pred, axis=-1)
        y_t = tf.reshape(tf.cast(y_true,tf.int32), [-1])
        y_p = tf.reshape(preds, [-1])
        cm = tf.math.confusion_matrix(y_t, y_p, num_classes=self.num_classes, dtype=tf.float32)
        self.total_cm.assign_add(cm)
    def result(self):
        cm = self.total_cm
        tp = tf.linalg.tensor_diag_part(cm)
        sum_r = tf.reduce_sum(cm, axis=0)
        sum_c = tf.reduce_sum(cm, axis=1)
        denom = sum_r + sum_c - tp
        iou = tf.math.divide_no_nan(tp, denom)
        return tf.reduce_mean(iou)
    def reset_states(self):
        self.total_cm.assign(tf.zeros_like(self.total_cm))

# 7) Compilación y entrenamiento ---------------------------------------------
# El entrenamiento inicial se elimina según el flujo de fase 1 y 2
# model.compile(...)
# model.fit(...)

# --- FASE 1: ENTRENAR SOLO LA CABEZA ---
print("Iniciando Fase 1: Entrenando solo la cabeza del modelo...")
for layer in model.layers:
    if 'efficientnetv2' in layer.name.lower():
        layer.trainable = False

model.compile(
    optimizer=tfa.optimizers.AdamW(learning_rate=1e-5, weight_decay=1e-4),
    loss=combined_loss,
    metrics=['accuracy', MeanIoU(num_classes=num_classes)]
)

cbs_phase1 = [
    EarlyStopping(monitor='val_mean_iou', mode='max', patience=8, restore_best_weights=True),
    ModelCheckpoint('phase1_head_training.h5', save_best_only=True, monitor='val_mean_iou', mode='max'),
    TensorBoard(log_dir='./logs/phase1_head', update_freq='epoch')
]

# Con imágenes de 256x256, puede ser necesario reducir el batch_size si hay OOM.
# El texto sugiere aumentarlo, pero esto depende de la VRAM disponible.
model.fit(train_X, train_y,
          batch_size=1, # Reducido de 16 para acomodar el mayor tamaño de imagen
          epochs=20,
          validation_data=(val_X, val_y),
          callbacks=cbs_phase1)

# --- FASE 2: AJUSTE FINO DE TODO EL MODELO ---
print("\nIniciando Fase 2: Ajuste fino de todo el modelo...")
for layer in model.layers:
    layer.trainable = True

model.compile(
    optimizer=tfa.optimizers.AdamW(learning_rate=1e-5, weight_decay=1e-4),
    loss=combined_loss,
    metrics=['accuracy', MeanIoU(num_classes=num_classes)]
)

cbs2 = [
    EarlyStopping(monitor='val_mean_iou', mode='max', patience=12, restore_best_weights=True),
    ModelCheckpoint('final_efficientnetv2_deeplab_miou_opt.h5', save_best_only=True, monitor='val_mean_iou', mode='max'),
    TensorBoard(log_dir='./logs/phase2_finetune', update_freq='epoch'),
    # CAMBIO 4: Dejar que mIoU impulse directamente el planificador
    ReduceLROnPlateau(monitor='val_mean_iou', mode='max',
                      factor=0.5, patience=4, min_lr=1e-7, verbose=1)
]

model.fit(train_X, train_y,
          batch_size=1, # Un batch_size aún más pequeño para fine-tuning con el modelo completo
          epochs=50,
          validation_data=(val_X, val_y),
          callbacks=cbs2)

# 8) Evaluación & visualización ----------------------------------------------
def evaluate_model(model, X, y):
    preds = model.predict(X)
    mask = np.argmax(preds, axis=-1)
    ious = []
    for cls in range(num_classes):
        yt = (y==cls).astype(int)
        yp = (mask==cls).astype(int)
        inter = np.sum(yt*yp)
        union = np.sum(yt) + np.sum(yp) - inter
        ious.append(inter/union if union>0 else 0)
    return {'mean_iou': np.mean(ious), 'class_ious': ious, 'pixel_accuracy': np.mean(mask==y)}

metrics = evaluate_model(model, val_X, val_y)
print(f"Mean IoU: {metrics['mean_iou']:.4f}")
print(f"Pixel Accuracy: {metrics['pixel_accuracy']:.4f}")
for i,iou in enumerate(metrics['class_ious']): print(f" Class {i}: {iou:.4f}")

def visualize_predictions(model, X, y, num_samples=3):
    idxs = np.random.choice(len(X), num_samples, replace=False)
    preds = model.predict(X[idxs])
    masks = np.argmax(preds, axis=-1)
    cmap = plt.cm.get_cmap('tab10', num_classes)
    plt.figure(figsize=(12,4*num_samples))
    for i, ix in enumerate(idxs):
        plt.subplot(num_samples,3,3*i+1); plt.imshow(X[ix]); plt.axis('off'); plt.title('Image')
        plt.subplot(num_samples,3,3*i+2); plt.imshow(y[ix],cmap=cmap,vmin=0,vmax=num_classes-1); plt.axis('off'); plt.title('GT Mask')
        plt.subplot(num_samples,3,3*i+3); plt.imshow(masks[i],cmap=cmap,vmin=0,vmax=num_classes-1); plt.axis('off'); plt.title('Predicted Mask')
    plt.tight_layout(); plt.show()

visualize_predictions(model, val_X, val_y)
print("Entrenamiento y evaluación completados.")