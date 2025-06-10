# -*- coding: utf-8 -*-
"""
claude_efficientnetv2_deeplab.py
Script completo: carga datos, construye y entrena el modelo DeepLabV3+ usando EfficientNetV2S como backbone,
con monkey-patch de SE module, pérdida ponderada píxel-wise, métricas y visualización.
"""

# 1) Imports ------------------------------------------------------------------
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import cv2
from tqdm import tqdm
import albumentations as A

import tensorflow as tf
from tensorflow.keras import layers, backend as K
from tensorflow.keras.layers import (
    Input, Conv2D, Conv2DTranspose, BatchNormalization,
    Activation, Dropout, Lambda, Concatenate, UpSampling2D
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
)
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import tensorflow_addons as tfa

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
    return A.Compose([
        A.RandomScale(scale_limit=0.2, p=0.5),
        A.PadIfNeeded(min_height=128, min_width=128, border_mode=cv2.BORDER_REFLECT),
        A.RandomCrop(height=128, width=128),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        # Use Affine instead of ShiftScaleRotate
        A.Affine(translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)}, scale=(0.9, 1.1), rotate=(-15, 15), p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.RandomGamma(gamma_limit=(80, 120), p=0.2),
        A.GaussianBlur(blur_limit=3, p=0.1),
        # The 'var_limit' argument is deprecated for GaussNoise
        A.GaussNoise(p=0.1),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.2),
        # The arguments for CoarseDropout have changed
        A.CoarseDropout(max_holes=8, max_height=16, max_width=16, min_holes=1, min_height=8, min_width=8, p=0.2),
    ])

def get_validation_augmentation():
    return A.Compose([])

# 3) Data loading -------------------------------------------------------------
def load_augmented_dataset(img_dir, mask_dir, target_size=(128,128), augment=False):
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
                augm = aug(image=img_arr, mask=mask_arr)
                img_arr, mask_arr = augm['image'], augm['mask']
            images.append(img_arr)
            masks.append(mask_arr)

    X = np.array(images, dtype='float32') / 255.0
    y = np.squeeze(np.array(masks, dtype='int32'), axis=-1)
    return X, y

# 4) Load / split into train & val -------------------------------------------
train_X, train_y = load_augmented_dataset('Balanced/train/images', 'Balanced/train/masks', augment=True)
val_X, val_y     = load_augmented_dataset('Balanced/val/images',   'Balanced/val/masks',   augment=False)

# 5) Número de clases & modelo -----------------------------------------------
num_classes = int(np.max(train_y) + 1)

def improved_ASPP(x, out_channels=256, rates=(6, 12, 18, 24)):
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
    # Correctly resize the pooled features to match the input feature map size
    pool = layers.Lambda(lambda args: tf.image.resize(args[0], tf.shape(args[1])[1:3]))([pool, x])
    branches.append(pool)

    # Concatenate and final convolution
    y = layers.Concatenate()(branches)
    y = layers.Conv2D(out_channels, 1, padding='same', use_bias=False)(y)
    y = layers.BatchNormalization()(y)
    y = layers.Dropout(0.3)(y)
    
    return layers.Activation('relu')(y)


def build_model(input_shape=(128,128,3)):
    backbone = EfficientNetV2S(
        input_shape=input_shape,
        num_classes=0,
        pretrained='imagenet',
        include_preprocessing=False
    )
    inputs = backbone.input
    # Regularización L2
    for l in backbone.layers:
        if isinstance(l, Conv2D):
            l.kernel_regularizer = tf.keras.regularizers.l2(1e-4)

    # Puntos de conexión correctos en keras-efficientnet-v2
    very_low = backbone.get_layer('stem_swish').output       # ~64×64
    low  = backbone.get_layer('add_2').output                # ~32×32  
    mid  = backbone.get_layer('add_6').output                # ~16×16
    high = backbone.get_layer('post_swish').output           # ~4×4
    
    # ASPP on high-level features
    x = improved_ASPP(high)
    
    # Decoder with multiple skip connections
    # Stage 1: 4→16
    x = layers.Conv2DTranspose(256, 4, strides=4, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Refine mid features
    mid_p = layers.Conv2D(128, 1, padding='same', use_bias=False)(mid)
    mid_p = layers.BatchNormalization()(mid_p)
    mid_p = layers.Activation('relu')(mid_p)
    
    x = layers.Concatenate()([x, mid_p])
    x = layers.SeparableConv2D(256, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.2)(x)
    
    # Stage 2: 16→32
    x = layers.Conv2DTranspose(128, 2, strides=2, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Refine low features
    low_p = layers.Conv2D(64, 1, padding='same', use_bias=False)(low)
    low_p = layers.BatchNormalization()(low_p)
    low_p = layers.Activation('relu')(low_p)
    
    x = layers.Concatenate()([x, low_p])
    x = layers.SeparableConv2D(128, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.1)(x)
    
    # Final refinement
    x = layers.Conv2D(64, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Classification head
    x = layers.Conv2D(num_classes, 1, padding='same')(x)
    outputs = layers.UpSampling2D((4,4), interpolation='bilinear')(x)
    
    return Model(inputs, outputs)

model = build_model(input_shape=train_X.shape[1:])

# 6) Pérdida, métricas --------------------------------------------------------
# Pesos de clase
class_counts = np.bincount(train_y.flatten())
total_pixels = class_counts.sum()
class_weights_np = total_pixels / (num_classes * np.maximum(class_counts,1))
class_weights = tf.constant(class_weights_np, dtype=tf.float32)

def weighted_sparse_ce(y_true, y_pred):
    y_true_i = tf.cast(y_true, tf.int32)
    w = tf.gather(class_weights, y_true_i)
    ce = tf.keras.losses.sparse_categorical_crossentropy(y_true_i, y_pred, from_logits=True)
    return tf.reduce_mean(ce * w)

def dice_coef(y_true, y_pred, smooth=1):
    y_t = tf.one_hot(tf.cast(y_true,'int32'), depth=num_classes)
    y_t = tf.reshape(y_t, [-1, num_classes])
    y_p = tf.reshape(y_pred, [-1, num_classes])
    y_p = tf.nn.softmax(y_p, axis=-1)
    inter = tf.reduce_sum(y_t * y_p, axis=0)
    union = tf.reduce_sum(y_t, axis=0) + tf.reduce_sum(y_p, axis=0)
    return tf.reduce_mean((2.*inter + smooth)/(union + smooth))

def boundary_loss(y_true, y_pred, alpha=1.0):
    """Add boundary-aware loss"""
    # y_true is 3D (batch, H, W), y_pred is 4D (batch, H, W, C)
    # Add a channel dimension to y_true to make it 4D
    y_true = tf.expand_dims(y_true, axis=-1)

    # Now all tensors are 4D, and the rest of the code will work
    
    # Compute gradients to detect boundaries
    dy_true = tf.abs(y_true[:, 1:, :, :] - y_true[:, :-1, :, :])
    dx_true = tf.abs(y_true[:, :, 1:, :] - y_true[:, :, :-1, :])

    dy_pred = tf.abs(y_pred[:, 1:, :, :] - y_pred[:, :-1, :, :])
    dx_pred = tf.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])

    # Pad to match original size with 4D padding rules
    dy_true = tf.pad(dy_true, [[0,0], [0,1], [0,0], [0,0]])
    dx_true = tf.pad(dx_true, [[0,0], [0,0], [0,1], [0,0]])
    dy_pred = tf.pad(dy_pred, [[0,0], [0,1], [0,0], [0,0]])
    dx_pred = tf.pad(dx_pred, [[0,0], [0,0], [0,1], [0,0]])

    boundary_loss = tf.reduce_mean(tf.abs(dy_true - dy_pred) + tf.abs(dx_true - dx_pred))
    return alpha * boundary_loss

def combined_loss(y_true, y_pred):
    ce_loss = weighted_sparse_ce(y_true, y_pred)
    dice_loss = 1 - dice_coef(y_true, y_pred)
    bound_loss = boundary_loss(tf.cast(y_true, tf.float32), tf.nn.softmax(y_pred))
    return ce_loss + dice_loss + 0.1 * bound_loss

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
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=combined_loss,
    metrics=['accuracy', MeanIoU(num_classes=num_classes)]
)

cbs1 = [
    EarlyStopping(monitor='val_mean_iou', mode='max', patience=10, restore_best_weights=True),
    ModelCheckpoint('phase1_efficientnetv2_deeplab.h5', save_best_only=True, monitor='val_mean_iou', mode='max'),
    TensorBoard(log_dir='./logs/phase1', update_freq='epoch')
]
model.fit(train_X, train_y,
          batch_size=16, epochs=10,
          validation_data=(val_X, val_y),
          callbacks=cbs1)

# --- FASE 1: ENTRENAR SOLO LA CABEZA ---
print("Iniciando Fase 1: Entrenando solo la cabeza del modelo...")

# Congelar el backbone
for layer in model.layers:
    if 'efficientnetv2' in layer.name.lower():
        layer.trainable = False

# Compilar con una tasa de aprendizaje más alta para la cabeza
model.compile(
    optimizer=tfa.optimizers.AdamW(learning_rate=2e-4, weight_decay=1e-4),
    loss=combined_loss,
    metrics=['accuracy', MeanIoU(num_classes=num_classes)]
)

# Callbacks para la Fase 1
cbs_phase1 = [
    EarlyStopping(monitor='val_mean_iou', mode='max', patience=8, restore_best_weights=True),
    ModelCheckpoint('phase1_head_training.h5', save_best_only=True, monitor='val_mean_iou', mode='max'),
    TensorBoard(log_dir='./logs/phase1_head', update_freq='epoch')
]

# ¡¡AQUÍ FALTA EL ENTRENAMIENTO!!
model.fit(train_X, train_y,
          batch_size=16, # Puede ser un batch size mayor
          epochs=15,    # Entrenar la cabeza por varias épocas
          validation_data=(val_X, val_y),
          callbacks=cbs_phase1)


# --- FASE 2: AJUSTE FINO DE TODO EL MODELO ---
print("\nIniciando Fase 2: Ajuste fino de todo el modelo...")

# Descongelar todas las capas
for layer in model.layers:
    layer.trainable = True

# Re-compilar con una tasa de aprendizaje muy baja
model.compile(
    optimizer=tfa.optimizers.AdamW(learning_rate=1e-5, weight_decay=1e-4),
    loss=combined_loss,
    metrics=['accuracy', MeanIoU(num_classes=num_classes)]
)

# Callbacks para la Fase 2 (los que ya tenías)
cbs2 = [
    EarlyStopping(monitor='val_mean_iou', mode='max', patience=10, restore_best_weights=True),
    ModelCheckpoint('final_efficientnetv2_deeplab.h5', save_best_only=True, monitor='val_mean_iou', mode='max'),
    TensorBoard(log_dir='./logs/phase2_finetune', update_freq='epoch'),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1)
]

# Entrenar el modelo completo
model.fit(train_X, train_y,
          batch_size=8, # Un batch size menor es común en fine-tuning
          epochs=50,   # Entrenar por más épocas
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
        plt.subplot(num_samples,3,3*i+2); plt.imshow(y[ix],cmap=cmap,vmin=0,vmax=num_classes-1); plt.axis('off'); plt.title('GT')
        plt.subplot(num_samples,3,3*i+3); plt.imshow(masks[i],cmap=cmap,vmin=0,vmax=num_classes-1); plt.axis('off'); plt.title('Pred')
    plt.tight_layout(); plt.show()

visualize_predictions(model, val_X, val_y)
print("Entrenamiento y evaluación completados.")