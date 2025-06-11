# -*- coding: utf-8 -*-
"""
claude_efficientnetv2_deeplab_iou_optimized.py

Script completo: carga datos, construye y entrena el modelo DeepLabV3+ usando EfficientNetV2S,
con optimizaciones específicas para maximizar la métrica Mean IoU.

Implementa las siguientes recomendaciones:
1. Re-balanceo de pesos en la función de pérdida para priorizar términos de IoU.
2. Callbacks de EarlyStopping y LR Scheduler monitorean `val_mean_iou`.
3. Composición de mini-lotes con sobremuestreo de clases minoritarias.
4. Inclusión de un término de pérdida de bordes (Boundary-aware).
5. Filtros más profundos (320 canales) en el módulo ASPP.
6. Aumentación en validación (TTA) durante el entrenamiento.
7. Eliminación de la métrica de 'accuracy' durante la compilación del modelo.
8. Carga explícita del mejor checkpoint entre fases de entrenamiento.
"""

import os
os.environ['MPLBACKEND'] = 'agg'  # Backend explícito para matplotlib

# --- PRIMERO TENSORFLOW ---
import tensorflow as tf

# --- Verificación de GPU ---
print("Versión de TensorFlow:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPUs disponibles: {[gpu.name for gpu in gpus]}")
    except RuntimeError as e:
        print(e)
else:
    print("No se encontró ninguna GPU. El entrenamiento se ejecutará en la CPU.")

# --- RESTO DE IMPORTS ---
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import cv2  # Importar después de TF
from tqdm import tqdm
import albumentations as A
from tensorflow.keras import layers, backend as K
from tensorflow.keras.layers import (
    Input, Conv2D, SeparableConv2D, Conv2DTranspose,
    BatchNormalization, Activation, Dropout,
    Lambda, Concatenate, UpSampling2D
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
)
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------------
# Monkey-patch SE module en keras_efficientnet_v2
# -------------------------------------------------------------------------------
import keras_efficientnet_v2.efficientnet_v2 as _efn2
from lovasz_losses_tf import lovasz_softmax

def patched_se_module(inputs, se_ratio=0.25, name=""):
    data_format = K.image_data_format()
    h_axis, w_axis = (1, 2) if data_format == 'channels_last' else (2, 3)
    se = Lambda(lambda x: tf.reduce_mean(x, axis=[h_axis, w_axis], keepdims=True),
                name=name + "se_reduce_pool")(inputs)
    channels = inputs.shape[-1]
    reduced_filters = max(1, int(channels / se_ratio))
    se = Conv2D(reduced_filters, 1, padding='same', name=name + "se_reduce_conv")(se)
    se = Activation('swish', name=name + "se_reduce_act")(se)
    se = Conv2D(channels, 1, padding='same', name=name + "se_expand_conv")(se)
    se = Activation('sigmoid', name=name + "se_excite")(se)
    return layers.Multiply(name=name + "se_excite_mul")([inputs, se])

_efn2.se_module = patched_se_module
from keras_efficientnet_v2 import EfficientNetV2S

# -------------------------------------------------------------------------------
# 1) Funciones de pérdida mejoradas (Recomendaciones 1 y 4)
# -------------------------------------------------------------------------------
class_counts = None  # Se inicializa luego de cargar datos

# Pérdida CE ponderada píxel-wise
def weighted_sparse_ce(y_true, y_pred):
    y_true_i = tf.cast(y_true, tf.int32)
    w = tf.gather(class_weights, y_true_i)
    ce = tf.keras.losses.sparse_categorical_crossentropy(y_true_i, y_pred, from_logits=True)
    return tf.reduce_mean(ce * w)

# Soft Jaccard (IoU) loss
def soft_jaccard_loss(y_true, y_pred, smooth=1e-5):
    y_true_o = tf.one_hot(tf.cast(y_true, tf.int32), depth=num_classes)
    y_pred_s = tf.nn.softmax(y_pred, axis=-1)
    inter = tf.reduce_sum(y_true_o * y_pred_s, axis=[1,2])
    union = tf.reduce_sum(y_true_o + y_pred_s, axis=[1,2]) - inter
    jaccard = (inter + smooth) / (union + smooth)
    return 1.0 - tf.reduce_mean(jaccard)

# Focal loss para ejemplos difíciles
def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    y_true_i = tf.cast(y_true, tf.int32)
    ce = tf.keras.losses.sparse_categorical_crossentropy(y_true_i, y_pred, from_logits=True)
    pt = tf.exp(-ce)
    focal = alpha * tf.pow(1 - pt, gamma) * ce
    return tf.reduce_mean(focal)

# Recomendación 4: Término de pérdida consciente de los bordes
def soft_boundary_loss(y_true, y_pred):
    y_true_e = tf.cast(y_true, tf.int32)
    y_pred_p = tf.nn.softmax(y_pred, axis=-1)
    # Magnitud de bordes basada en Sobel para la máscara real
    sobel_x = tf.image.sobel_edges(tf.expand_dims(tf.cast(y_true_e, tf.float32), -1))[..., 0]
    edges = tf.cast(tf.reduce_max(tf.abs(sobel_x), axis=-1) > 0, tf.float32)
    # Suma ponderada de las probabilidades predichas en las ubicaciones de los bordes
    pred_edg = tf.reduce_sum(y_pred_p * tf.expand_dims(edges, -1), axis=[1, 2])
    return 1.0 - tf.reduce_mean(pred_edg)

# Recomendación 1 y 4: Pérdida combinada con re-balanceo de pesos y término de bordes
def enhanced_combined_loss(y_true, y_pred):
    jac   = 0.45 * soft_jaccard_loss(y_true, y_pred)
    lov   = 0.30 * lovasz_softmax(
               y_pred, tf.cast(y_true, tf.int32), per_image=True) # per_image=True es clave
    focal = 0.15 * focal_loss(y_true, y_pred)
    ce    = 0.10 * weighted_sparse_ce(y_true, y_pred)
    bd    = 0.05 * soft_boundary_loss(y_true, y_pred)
    return jac + lov + focal + ce + bd

# -------------------------------------------------------------------------------
# 2) Pipelines de augmentación mejorada
# -------------------------------------------------------------------------------
def get_training_augmentation():
    return A.Compose([
        A.RandomScale(scale_limit=0.1, p=0.3),
        A.PadIfNeeded(min_height=256, min_width=256, border_mode=cv2.BORDER_REFLECT),
        A.RandomCrop(height=256, width=256),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.2),
        A.RandomGamma(gamma_limit=(90, 110), p=0.1),
        A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=5, p=0.1),
        A.CoarseDropout(max_holes=4, max_height=16, max_width=16,
                        min_holes=1, min_height=8, min_width=8, p=0.1),
    ])

def get_validation_augmentation():
    return A.Compose([])

# -------------------------------------------------------------------------------
# 3) Carga de datos
# -------------------------------------------------------------------------------

def load_augmented_dataset(img_dir, mask_dir, target_size=(256,256), augment=False):
    images, masks = [], []
    files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg','.png'))]
    aug = get_training_augmentation() if augment else get_validation_augmentation()
    mask_size = (256,256)
    with ThreadPoolExecutor() as exe:
        futures = {exe.submit(lambda fn: (
            img_to_array(load_img(os.path.join(img_dir,fn), target_size=target_size)),
            img_to_array(load_img(os.path.join(mask_dir,fn.rsplit('.',1)[0] + '_mask.png'),
                                  color_mode='grayscale', target_size=mask_size))
        ), fn): fn for fn in files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Cargando datos"):
            img_arr, mask_arr = future.result()
            if augment:
                augm = aug(image=img_arr.astype('uint8'), mask=mask_arr)
                img_arr, mask_arr = augm['image'], augm['mask']
            images.append(img_arr)
            masks.append(mask_arr)
    X = np.array(images, dtype='float32') / 255.0
    y = np.array(masks, dtype='int32')
    if y.ndim == 4 and y.shape[-1] == 1:
        y = np.squeeze(y, axis=-1)
    return X, y

# Carga y división
train_X, train_y = load_augmented_dataset('Balanced/train/images', 'Balanced/train/masks', augment=True)
val_X,   val_y   = load_augmented_dataset('Balanced/val/images',   'Balanced/val/masks',   augment=False)

# Constantes basadas en datos
num_classes = int(np.max(train_y) + 1)
class_counts = np.bincount(train_y.flatten())

# Recomendación 3: Composición de mini-lotes - mantener visibles las clases minoritarias
class_prob = class_counts / class_counts.sum()
rare_class_indices = np.argsort(class_prob)[:2]  # Dos clases más raras
# Encontrar imágenes que contienen cualquiera de las clases raras
presence_mask = np.isin(train_y.reshape(len(train_y), -1), rare_class_indices)
rare_idx = np.any(presence_mask, axis=1)
print(f"Sobremuestreo: Encontradas {np.sum(rare_idx)} imágenes con clases raras. Duplicándolas.")
extra_imgs = train_X[rare_idx]
extra_masks = train_y[rare_idx]
train_X = np.concatenate([train_X, extra_imgs], axis=0)
train_y = np.concatenate([train_y, extra_masks], axis=0)
# Barajar el conjunto de datos combinado es crucial
shuffle_idx = np.random.permutation(len(train_X))
train_X = train_X[shuffle_idx]
train_y = train_y[shuffle_idx]
print(f"Nuevo tamaño del conjunto de entrenamiento: {len(train_X)} imágenes.")

# Calcular pesos de clase después del sobremuestreo
total_pixels = train_y.size
class_weights_np = total_pixels / (num_classes * np.maximum(np.bincount(train_y.flatten()), 1))
class_weights = tf.constant(class_weights_np, dtype=tf.float32)


# -------------------------------------------------------------------------------
# 4) ASPP mejorado (Recomendación 5)
# -------------------------------------------------------------------------------
def ASPP(x, out=320, rates=(6, 12, 18, 24, 36)): # out=320 para filtros más profundos
    branches = [Activation('relu')(BatchNormalization()(Conv2D(out, 1, use_bias=False)(x)))]
    for r in rates:
        d = Activation('relu')(BatchNormalization()(
            SeparableConv2D(out, 3, dilation_rate=r, padding='same', use_bias=False)(x)
        ))
        branches.append(d)
    pool = layers.GlobalAveragePooling2D(keepdims=True)(x)
    pool = Activation('relu')(BatchNormalization()(Conv2D(out, 1, use_bias=False)(pool)))
    pool = Lambda(lambda a: tf.image.resize(a[0], tf.shape(a[1])[1:3]))([pool, x])
    branches.append(pool)
    y = Activation('relu')(BatchNormalization()(Conv2D(out, 1, use_bias=False)(Concatenate()(branches))))
    y = Dropout(0.1)(y)
    return y

# -------------------------------------------------------------------------------
# 5) Definición del modelo con skip connections
# -------------------------------------------------------------------------------
def build_model(shape=(256,256,3)):
    backbone = EfficientNetV2S(input_shape=shape, num_classes=0,
                               pretrained='imagenet', include_preprocessing=False)
    for l in backbone.layers:
        if isinstance(l,(Conv2D, SeparableConv2D)):
            l.kernel_regularizer = tf.keras.regularizers.l2(1e-4)
    inp = backbone.input
    high     = backbone.get_layer('post_swish').output   # 8×8
    mid      = backbone.get_layer('add_6').output        # 16×16
    low      = backbone.get_layer('add_2').output        # 32×32
    very_low = backbone.get_layer('stem_swish').output   # 64×64

    # Recomendación 5: ASPP mejorado con filtros más profundos
    x = ASPP(high, out=320)
    
    # Stage 1: 8->32
    x = Activation('relu')(BatchNormalization()(Conv2DTranspose(256,4,strides=4,padding='same',use_bias=False)(x)))
    mid_p = Activation('relu')(BatchNormalization()(Conv2D(256,1,padding='same',use_bias=False)(mid)))
    x = Activation('relu')(BatchNormalization()(SeparableConv2D(256,3,padding='same',use_bias=False)(Concatenate()([x,mid_p]))))
    x = Dropout(0.1)(x)

    # Stage 2: 32->64
    x = Activation('relu')(BatchNormalization()(Conv2DTranspose(128,2,strides=2,padding='same',use_bias=False)(x)))
    low_p = Activation('relu')(BatchNormalization()(Conv2D(128,1,padding='same',use_bias=False)(low)))
    x = Activation('relu')(BatchNormalization()(SeparableConv2D(128,3,padding='same',use_bias=False)(Concatenate()([x,low_p]))))
    x = Dropout(0.1)(x)

    # Stage 3: 64->128
    x = Activation('relu')(BatchNormalization()(Conv2DTranspose(64,2,strides=2,padding='same',use_bias=False)(x)))
    very_low_p = Activation('relu')(BatchNormalization()(Conv2D(64,1,padding='same',use_bias=False)(very_low)))
    x = Activation('relu')(BatchNormalization()(SeparableConv2D(96,3,padding='same',use_bias=False)(Concatenate()([x,very_low_p]))))

    # Final upsampling: 128->256
    x = UpSampling2D(size=(2,2), interpolation='bilinear')(x)
    
    # Refinamiento final
    x = Activation('relu')(BatchNormalization()(SeparableConv2D(64,3,padding='same',use_bias=False)(x)))
    x = Activation('relu')(BatchNormalization()(SeparableConv2D(32,3,padding='same',use_bias=False)(x)))
    out = Conv2D(num_classes,1,padding='same',dtype='float32')(x)
    return Model(inp, out)

# -------------------------------------------------------------------------------
# 6) Métrica MeanIoU personalizada
# -------------------------------------------------------------------------------
class MeanIoU(tf.keras.metrics.Metric):
    def __init__(self, num_classes, name='mean_iou', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.total_cm = self.add_weight(
            shape=(num_classes, num_classes),
            name='total_confusion_matrix',
            initializer='zeros'
        )
    def update_state(self, y_true, y_pred, sample_weight=None):
        preds = tf.argmax(y_pred, axis=-1)
        if y_true.ndim == 4 and y_true.shape[-1] == 1:
            y_true = tf.squeeze(y_true, axis=-1)
        y_t = tf.reshape(tf.cast(y_true, tf.int32), [-1])
        y_p = tf.reshape(preds, [-1])
        cm = tf.math.confusion_matrix(y_t, y_p, num_classes=self.num_classes, dtype=tf.float32)
        self.total_cm.assign_add(cm)
    def result(self):
        tp = tf.linalg.tensor_diag_part(self.total_cm)
        sum_r = tf.reduce_sum(self.total_cm, axis=0)
        sum_c = tf.reduce_sum(self.total_cm, axis=1)
        denom = sum_r + sum_c - tp
        iou = tf.math.divide_no_nan(tp, denom)
        return tf.reduce_mean(iou)
    def reset_states(self):
        self.total_cm.assign(tf.zeros_like(self.total_cm))

# -------------------------------------------------------------------------------
# 7) Estrategia de entrenamiento mejorada (Recomendaciones 2, 6, 7)
# -------------------------------------------------------------------------------

# Recomendación 6: Aumentación en validación (TTA ligero)
# Generador que aplica volteos aleatorios a los lotes de validación
val_gen = tf.data.Dataset.from_tensor_slices((val_X, val_y))\
       .batch(8).map(lambda x,y: (tf.image.random_flip_left_right(x), y),
                     num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

GPU_DEVICE = '/GPU:0' if gpus else '/CPU:0'
print(f"\n--- Iniciando entrenamiento en el dispositivo: {GPU_DEVICE} ---")

with tf.device(GPU_DEVICE):
    model = build_model(train_X.shape[1:])
    model.summary()
    print("Modelo construido y ubicado en el dispositivo.")

    # Fase 1: entrenar cabeza
    print("\n--- Fase 1: Entrenamiento de la cabeza decodificadora ---")
    for layer in model.layers:
        if 'efficientnetv2' in layer.name.lower():
            layer.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=3e-4, weight_decay=1e-4),
        loss=enhanced_combined_loss,
        # Recomendación 7: Monitorear exclusivamente IoU
        metrics=[MeanIoU(num_classes=num_classes)]
    )
    # Recomendación 2: Callbacks monitorean val_mean_iou
    cbs1 = [
        EarlyStopping(monitor='val_mean_iou', mode='max', patience=10, restore_best_weights=True),
        ModelCheckpoint('phase1_head_training.keras',save_best_only=True,monitor='val_mean_iou',mode='max'),
        TensorBoard(log_dir='./logs/phase1_head', update_freq='epoch'),
        ReduceLROnPlateau(monitor='val_mean_iou', mode='max', factor=0.6, patience=4, min_lr=1e-6, verbose=1)
    ]
    model.fit(train_X, train_y, batch_size=12, epochs=25,
              # Recomendación 6: Usar generador de validación
              validation_data=val_gen, callbacks=cbs1)

    # Fase 2: fine-tuning parcial (último 20%)
    print("\n--- Fase 2: Fine-tuning parcial del encoder ---")
    backbone = [l for l in model.layers if 'efficientnetv2' in l.name.lower()]
    if backbone:
        b = backbone[0]
        total = len(b.layers)
        start = int(0.8 * total)
        for i, l in enumerate(b.layers): l.trainable = i >= start
    
    # <<< RECOMENDACIÓN IMPLEMENTADA >>>
    # Cargar los mejores pesos de la fase 1 antes de continuar
    print("Cargando los mejores pesos de la Fase 1 desde 'phase1_head_training.keras'")
    model.load_weights('phase1_head_training.keras')

    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=5e-6, weight_decay=1e-4),
        loss=enhanced_combined_loss,
        # Recomendación 7: Monitorear exclusivamente IoU
        metrics=[MeanIoU(num_classes=num_classes)]
    )
    # Recomendación 2: Callbacks monitorean val_mean_iou
    cbs2 = [
        EarlyStopping(monitor='val_mean_iou', mode='max', patience=10, restore_best_weights=True),
        ModelCheckpoint('final_efficientnetv2_deeplab.keras', save_best_only=True,monitor='val_mean_iou',mode='max'),
        TensorBoard(log_dir='./logs/phase2_finetune', update_freq='epoch'),
        ReduceLROnPlateau(monitor='val_mean_iou', mode='max', factor=0.6, patience=6, min_lr=1e-8, verbose=1)
    ]
    model.fit(train_X, train_y, batch_size=6, epochs=60,
              # Recomendación 6: Usar generador de validación
              validation_data=val_gen, callbacks=cbs2)

    # Fase 3: fine-tuning completo
    print("\n--- Fase 3: Fine-tuning completo del modelo ---")
    for layer in model.layers:
        layer.trainable = True

    # <<< RECOMENDACIÓN IMPLEMENTADA >>>
    # Cargar los mejores pesos de la fase 2 antes de continuar
    print("Cargando los mejores pesos de la Fase 2 desde 'final_efficientnetv2_deeplab.keras'")
    model.load_weights('final_efficientnetv2_deeplab.keras')

    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-6, weight_decay=5e-5),
        loss=enhanced_combined_loss,
        # Recomendación 7: Monitorear exclusivamente IoU
        metrics=[MeanIoU(num_classes=num_classes)]
    )
    # Recomendación 2: Callbacks monitorean val_mean_iou
    cbs3 = [
        EarlyStopping(monitor='val_mean_iou', mode='max', patience=10, restore_best_weights=True),
        ModelCheckpoint('final_full_finetune.keras', save_best_only=True,monitor='val_mean_iou',mode='max'),
        TensorBoard(log_dir='./logs/phase3_full', update_freq='epoch'),
        ReduceLROnPlateau(monitor='val_mean_iou', mode='max', factor=0.6, patience=8, min_lr=1e-9, verbose=1)
    ]
    model.fit(train_X, train_y, batch_size=4, epochs=40,
              # Recomendación 6: Usar generador de validación
              validation_data=val_gen, callbacks=cbs3)

# -------------------------------------------------------------------------------
# 8) Test Time Augmentation (TTA)
# -------------------------------------------------------------------------------
def predict_with_tta(model, X, num_tta=5):
    preds = []
    # Original prediction
    pred = model.predict(X, verbose=0)
    preds.append(pred)
    # Augmented predictions
    for _ in range(num_tta-1):
        X_aug = np.copy(X)
        # Random horizontal flip
        if np.random.rand() > 0.5:
             X_aug = np.flip(X_aug, axis=2)
        
        p_aug = model.predict(X_aug, verbose=0)
        
        # Un-augment the prediction if augmentation was applied
        if np.array_equal(X_aug, np.flip(X, axis=2)):
             p_aug = np.flip(p_aug, axis=2)
             
        preds.append(p_aug)
    return np.mean(preds, axis=0)

def evaluate_model_with_tta(model, X, y):
    preds = predict_with_tta(model, X)
    mask = np.argmax(preds, axis=-1)
    ious = []
    for cls in range(num_classes):
        yt = (y==cls).astype(int)
        yp = (mask==cls).astype(int)
        inter = np.sum(yt * yp)
        union = np.sum(yt) + np.sum(yp) - inter
        ious.append(inter/union if union>0 else 0)
    print(f"Mean IoU (con TTA): {np.mean(ious):.4f}")
    return ious

# -------------------------------------------------------------------------------
# 9) Evaluación y visualización
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

# La función de visualización necesita matplotlib, se deja como referencia
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
        plt.imshow(y[i], vmin=0, vmax=num_classes-1)
        plt.title("Máscara Real")
        plt.axis('off')

        plt.subplot(num_samples, 3, i*3 + 3)
        plt.imshow(pred_masks[i], vmin=0, vmax=num_classes-1)
        plt.title("Máscara Predicha")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("predictions_visualization.png")
    print("Visualización de predicciones guardada en 'predictions_visualization.png'")


print("\n--- Evaluación Final del Modelo ---")
# Cargar el mejor modelo final guardado en disco para la evaluación
print("Cargando el mejor modelo final desde 'final_full_finetune.keras' para la evaluación.")
model.load_weights('final_full_finetune.keras')

metrics = evaluate_model(model, val_X, val_y)
print(f"Mean IoU (final): {metrics['mean_iou']:.4f}")
print(f"Pixel Accuracy (final): {metrics['pixel_accuracy']:.4f}")
for i, iou in enumerate(metrics['class_ious']): print(f" IoU Clase {i}: {iou:.4f}")

evaluate_model_with_tta(model, val_X, val_y)

visualize_predictions(model, val_X, val_y)
print("\nEntrenamiento y evaluación completados.")