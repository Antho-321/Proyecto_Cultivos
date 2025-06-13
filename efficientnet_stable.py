# -*- coding: utf-8 -*-
"""
claude_efficientnetv2_deeplab_iou_optimized.py

Script completo: carga datos, construye y entrena el modelo DeepLabV3+ usando 
EfficientNetV2S como backbone. Esta versión implementa un bucle de entrenamiento
personalizado con la clase IoUOptimizedModel para optimizar directamente una 
pérdida combinada centrada en IoU (Lovasz, Tversky, Boundary Loss) a través
de un entrenamiento en tres fases.
"""

# 1) Imports ------------------------------------------------------------------
import os
os.environ['MPLBACKEND'] = 'agg'  # Set the backend explicitly

# --- PRIMERO TENSORFLOW ---
# Importa TensorFlow ANTES que otras librerías pesadas como cv2 o albumentations
import tensorflow as tf

# --- Verificación de GPU (ahora usando el tf ya importado) ---
print("Versión de TensorFlow:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPUs disponibles: {[gpu.name for gpu in gpus]}")
        device = '/GPU:0'
    except RuntimeError as e:
        print(e)
        device = '/CPU:0'
else:
    print("No se encontró ninguna GPU. El entrenamiento se ejecutará en la CPU.")
    device = '/CPU:0'

# --- RESTO DE IMPORTS ---
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import cv2  # Ahora cv2 se importa DESPUÉS de TensorFlow
from tqdm import tqdm
import albumentations as A
from tensorflow import keras
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
# Monkey-patch para usar un Lambda layer en SE module de keras_efficientnet_v2
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

# 2) Augmentation pipelines --------------------------------------------------
def get_training_augmentation():
    return A.Compose([
        A.RandomScale(scale_limit=0.2, p=0.5),
        A.PadIfNeeded(min_height=256, min_width=256, border_mode=cv2.BORDER_REFLECT),
        A.RandomCrop(height=256, width=256),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.Affine(translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)}, scale=(0.9,1.1), rotate=(-15,15), p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.RandomGamma(gamma_limit=(80,120), p=0.2),
        A.GaussianBlur(blur_limit=3, p=0.1),
        A.GaussNoise(p=0.1),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.2),
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, min_holes=1, min_height=16, min_width=16, p=0.2),
    ])

def get_validation_augmentation():
    return A.Compose([])

# 3) Data loading -------------------------------------------------------------
def load_augmented_dataset(img_dir, mask_dir, target_size=(256,256), augment=False):
    images, masks = [], []
    files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg','.png'))]
    aug = get_training_augmentation() if augment else get_validation_augmentation()
    
    mask_target_size = (256, 256)

    with ThreadPoolExecutor() as exe:
        futures = {exe.submit(lambda fn: (
            img_to_array(load_img(os.path.join(img_dir,fn), target_size=target_size)),
            img_to_array(load_img(os.path.join(mask_dir,fn.rsplit('.',1)[0] + '_mask.png'),
                                  color_mode='grayscale', target_size=mask_target_size))
        ), fn): fn for fn in files}
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Loading {'train' if augment else 'val'} data"):
            img_arr, mask_arr = future.result()
            if augment:
                augm = aug(image=img_arr.astype('uint8'), mask=mask_arr)
                img_arr, mask_arr = augm['image'], augm['mask']
            images.append(img_arr)
            masks.append(mask_arr)
            
    X = np.array(images, dtype='float32') / 255.0
    y = np.array(masks, dtype='int32')

    if y.shape[-1] == 1:
        y = np.squeeze(y, axis=-1)
        
    return X, y

# 4) Load / split ------------------------------------------------------------
train_X, train_y = load_augmented_dataset('Balanced/train/images', 'Balanced/train/masks', target_size=(256, 256), augment=True)
val_X,   val_y   = load_augmented_dataset('Balanced/val/images',   'Balanced/val/masks',   target_size=(256, 256), augment=False)

# 5) Número de clases & modelo ----------------------------------------------
num_classes = int(np.max(train_y) + 1)
print(f"\nNúmero de clases detectado: {num_classes}")

# 6) Definición del Modelo (Arquitectura) ------------------------------------

def ASPP(x, out=128, rates=(6,12,18,24)):
    branches = [Activation('relu')(BatchNormalization()(Conv2D(out,1,use_bias=False)(x)))]
    for r in rates:
        d = Activation('relu')(BatchNormalization()(SeparableConv2D(out,3,dilation_rate=r,padding='same',use_bias=False)(x)))
        branches.append(d)
    pool = layers.GlobalAveragePooling2D(keepdims=True)(x)
    pool = Activation('relu')(BatchNormalization()(Conv2D(out,1,use_bias=False)(pool)))
    pool = Lambda(lambda a: tf.image.resize(a[0], tf.shape(a[1])[1:3]))([pool,x])
    branches.append(pool)
    y = Activation('relu')(BatchNormalization()(Conv2D(out,1,use_bias=False)(Concatenate()(branches))))
    return y

def build_model(shape=(256,256,3), num_classes_arg=None):
    # Usar num_classes_arg para evitar conflictos con la variable global
    if num_classes_arg is None:
        raise ValueError("num_classes_arg must be provided to build_model")

    backbone = EfficientNetV2S(input_shape=shape, num_classes=0, pretrained='imagenet', include_preprocessing=False)
    for l in backbone.layers:
        if isinstance(l,(Conv2D,SeparableConv2D)):
            l.kernel_regularizer = tf.keras.regularizers.l2(1e-4)

    inp = backbone.input
    high     = backbone.get_layer('post_swish').output   # 8×8 en un input de 256x256
    mid      = backbone.get_layer('add_6').output       # 16×16
    low      = backbone.get_layer('add_2').output       # 32×32
    very_low = backbone.get_layer('stem_swish').output  # 64x64

    x = ASPP(high)                                                      # 8×8 → 8×8
    x = Activation('relu')(BatchNormalization()(Conv2DTranspose(256,4,strides=4,padding='same',use_bias=False)(x))) # 8→32

    mid_p = Activation('relu')(BatchNormalization()(Conv2D(128,1,padding='same',use_bias=False)(mid)))
    x = Activation('relu')(BatchNormalization()(SeparableConv2D(256,3,padding='same',use_bias=False)(Concatenate()([x,mid_p]))))

    x = Activation('relu')(BatchNormalization()(Conv2DTranspose(128,2,strides=2,padding='same',use_bias=False)(x))) # 32→64
    low_p = Activation('relu')(BatchNormalization()(Conv2D(64,1,padding='same',use_bias=False)(low)))
    x = Activation('relu')(BatchNormalization()(SeparableConv2D(128,3,padding='same',use_bias=False)(Concatenate()([x,low_p]))))

    x = Activation('relu')(BatchNormalization()(Conv2DTranspose(64,2,strides=2,padding='same',use_bias=False)(x)))  # 64→128
    very_low_p = Activation('relu')(BatchNormalization()(Conv2D(48,1,padding='same',use_bias=False)(very_low)))
    x = Activation('relu')(BatchNormalization()(SeparableConv2D(96,3,padding='same',use_bias=False)(Concatenate()([x,very_low_p]))))

    # =========================================================================
    # SOLUCIÓN IMPLEMENTADA AQUÍ
    # El tensor llega con 128x128. Para llegar a 256x256 se necesita un factor x2.
    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)   # 128→256
    # =========================================================================
    
    out = Conv2D(num_classes_arg,1,padding='same',dtype='float32')(x)
    return Model(inp,out), backbone


# 7) Pérdidas Mejoradas y Métricas Personalizadas ------------------------------

def tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3, smooth=1e-7):
    """
    Tversky Loss para segmentación.
    - alpha: Peso para Falsos Positivos (FP).
    - beta:  Peso para Falsos Negativos (FN).
    - alpha + beta = 1. Para penalizar más los FN (clases pequeñas), usar alpha < 0.5.
    """
    y_true_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), depth=num_classes, axis=-1)
    y_pred_probs = tf.nn.softmax(y_pred, axis=-1)

    y_true_flat = tf.reshape(y_true_one_hot, [-1, num_classes])
    y_pred_flat = tf.reshape(y_pred_probs, [-1, num_classes])
    
    tp = tf.reduce_sum(y_true_flat * y_pred_flat, axis=0)
    fn = tf.reduce_sum(y_true_flat * (1 - y_pred_flat), axis=0)
    fp = tf.reduce_sum((1 - y_true_flat) * y_pred_flat, axis=0)
    
    tversky_index = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    
    return 1.0 - tf.reduce_mean(tversky_index)

def adaptive_boundary_enhanced_dice_loss_tf(y_true, y_pred, gamma=2.0, smooth=1e-7):
    """
    Pérdida de bordes basada en el Dice score, calculado sobre el gradiente
    morfológico de las máscaras.
    """
    y_true_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), depth=num_classes, axis=-1)
    y_pred_probs = tf.nn.softmax(y_pred, axis=-1)

    # El kernel debe ser [alto, ancho, canales]
    kernel = tf.ones((3, 3, num_classes), dtype=tf.float32)

    y_true_dilated = tf.nn.dilation2d(y_true_one_hot, filters=kernel, strides=(1, 1, 1, 1), padding="SAME", data_format="NHWC")
    y_true_eroded = tf.nn.erosion2d(y_true_one_hot, filters=kernel, strides=(1, 1, 1, 1), padding="SAME", data_format="NHWC")
    y_true_boundary = y_true_dilated - y_true_eroded

    y_pred_boundary = tf.nn.max_pool2d(y_pred_probs, ksize=3, strides=1, padding='SAME') - y_pred_probs
    
    y_true_boundary_flat = tf.reshape(y_true_boundary, [-1])
    y_pred_boundary_flat = tf.reshape(y_pred_boundary, [-1])
    
    intersection = tf.reduce_sum(y_true_boundary_flat * y_pred_boundary_flat)
    union = tf.reduce_sum(y_true_boundary_flat) + tf.reduce_sum(y_pred_boundary_flat)
    
    dice_score = (2. * intersection + smooth) / (union + smooth)
    return 1.0 - dice_score

def ultimate_iou_loss(y_true, y_pred):
    """
    Pérdida combinada que optimiza directamente IoU y los bordes.
    """
    lov = 0.50 * lovasz_softmax(y_pred, tf.cast(y_true, tf.int32), per_image=True)
    abe_dice = 0.30 * adaptive_boundary_enhanced_dice_loss_tf(y_true, y_pred, gamma=2.0)
    tver = 0.20 * tversky_loss(y_true, y_pred, alpha=0.4, beta=0.6) # beta > alpha penaliza más los FNs
    
    return lov + abe_dice + tver

class MeanIoU(tf.keras.metrics.Metric):
    def __init__(self,num_classes,name='mean_iou',**kwargs):
        super().__init__(name=name,**kwargs)
        self.num_classes = num_classes
        self.total_cm = self.add_weight(
            shape=(num_classes,num_classes),
            name='total_confusion_matrix',
            initializer='zeros'
        )
    def update_state(self,y_true,y_pred,sample_weight=None):
        preds = tf.argmax(y_pred,axis=-1)
        if len(y_true.shape) == 4 and y_true.shape[-1] == 1:
            y_true = tf.squeeze(y_true, axis=-1)
            
        y_t = tf.reshape(tf.cast(y_true,tf.int32),[-1])
        y_p = tf.reshape(preds,[-1])
        cm = tf.math.confusion_matrix(y_t,y_p,num_classes=self.num_classes,dtype=tf.float32)
        self.total_cm.assign_add(cm)
    def result(self):
        cm = self.total_cm
        tp = tf.linalg.tensor_diag_part(cm)
        sum_r = tf.reduce_sum(cm,axis=0)
        sum_c = tf.reduce_sum(cm,axis=1)
        denom = sum_r + sum_c - tp
        iou = tf.math.divide_no_nan(tp,denom)
        return tf.reduce_mean(iou)
    def reset_states(self):
        self.total_cm.assign(tf.zeros_like(self.total_cm))

# 8) Modelo Personalizado con Bucle de Entrenamiento Optimizado para IoU -----
class IoUOptimizedModel(keras.Model):
    def __init__(self, shape, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.seg_model, self.backbone = build_model(shape, num_classes_arg=num_classes)
        
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.iou_metric = MeanIoU(num_classes=num_classes, name='enhanced_mean_iou')

    def call(self, inputs, training=False):
        return self.seg_model(inputs, training=training)

    @property
    def metrics(self):
        return [self.loss_tracker, self.iou_metric]

    def train_step(self, data):
        x, y_true = data

        with tf.GradientTape() as tape:
            y_pred = self.seg_model(x, training=True)
            loss = ultimate_iou_loss(y_true, y_pred)

        trainable_vars = self.seg_model.trainable_variables
        grads = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(grads, trainable_vars))

        self.loss_tracker.update_state(loss)
        self.iou_metric.update_state(y_true, y_pred)
        
        return {m.name: m.result() for m in self.metrics}
        
    def test_step(self, data):
        x, y_true = data
        y_pred = self.seg_model(x, training=False)
        
        loss = ultimate_iou_loss(y_true, y_pred)
        
        self.loss_tracker.update_state(loss)
        self.iou_metric.update_state(y_true, y_pred)
        
        return {m.name: m.result() for m in self.metrics}

# 9) Entrenamiento por Fases con el Modelo Personalizado ---------------------
print(f"\n--- Iniciando entrenamiento en el dispositivo: {device} ---")

# Directorio para guardar los checkpoints
checkpoint_dir = './checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)
ckpt1_path = os.path.join(checkpoint_dir, 'phase1_iou_model.keras')
ckpt2_path = os.path.join(checkpoint_dir, 'phase2_iou_model.keras')
ckpt3_path = os.path.join(checkpoint_dir, 'phase3_iou_model.keras')

# Generador de validación simple
val_gen = (val_X, val_y)

with tf.device(device):
    iou_aware_model = IoUOptimizedModel(shape=train_X.shape[1:], num_classes=num_classes)
    
    # Referencias para compatibilidad con la lógica de congelamiento
    model = iou_aware_model.seg_model
    backbone = iou_aware_model.backbone
    
    iou_aware_model.seg_model.summary()
    
    MONITOR_METRIC = 'val_enhanced_mean_iou'
    
    # --- Fase 1: entrenar cabeza ---
    print("\n--- Fase 1: Entrenando solo el decoder (backbone congelado) ---")
    backbone.trainable = False
    
    iou_aware_model.compile(optimizer=tf.keras.optimizers.AdamW(5e-4, weight_decay=1e-4))
    
    cbs1 = [
        ModelCheckpoint(ckpt1_path, save_best_only=True, monitor=MONITOR_METRIC, mode='max', verbose=1),
        EarlyStopping(monitor=MONITOR_METRIC, mode='max', patience=10, restore_best_weights=False),
        TensorBoard(log_dir='./logs/iou_phase1_head', update_freq='epoch')
    ]
    
    iou_aware_model.fit(train_X, train_y, batch_size=12, epochs=30,
                        validation_data=val_gen, callbacks=cbs1)
    
    print(f"Cargando mejores pesos de la fase 1 desde: {ckpt1_path}")
    iou_aware_model.load_weights(ckpt1_path)

    # --- Fase 2: fine-tune parcial ---
    print("\n--- Fase 2: Fine-tuning del 20% superior del backbone ---")
    backbone.trainable = True
    fine_tune_at = int(len(backbone.layers) * 0.80)
    for layer in backbone.layers[:fine_tune_at]:
        layer.trainable = False

    iou_aware_model.compile(optimizer=tf.keras.optimizers.AdamW(2e-5, weight_decay=1e-4))
    
    cbs2 = [
        ModelCheckpoint(ckpt2_path, save_best_only=True, monitor=MONITOR_METRIC, mode='max', verbose=1),
        EarlyStopping(monitor=MONITOR_METRIC, mode='max', patience=12, restore_best_weights=False),
        TensorBoard(log_dir='./logs/iou_phase2_partial', update_freq='epoch')
    ]

    iou_aware_model.fit(train_X, train_y, batch_size=6, epochs=40,
                        validation_data=val_gen, callbacks=cbs2)

    print(f"Cargando mejores pesos de la fase 2 desde: {ckpt2_path}")
    iou_aware_model.load_weights(ckpt2_path)

    # --- Fase 3: full fine-tune ---
    print("\n--- Fase 3: Fine-tuning de todo el modelo (backbone + decoder) ---")
    for layer in backbone.layers:
        layer.trainable = True
        
    iou_aware_model.compile(optimizer=tf.keras.optimizers.AdamW(5e-6, weight_decay=5e-5))
    
    cbs3 = [
        ModelCheckpoint(ckpt3_path, save_best_only=True, monitor=MONITOR_METRIC, mode='max', verbose=1),
        EarlyStopping(monitor=MONITOR_METRIC, mode='max', patience=15, restore_best_weights=False),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1),
        TensorBoard(log_dir='./logs/iou_phase3_full', update_freq='epoch')
    ]
    
    iou_aware_model.fit(train_X, train_y, batch_size=4, epochs=50,
                        validation_data=val_gen, callbacks=cbs3)
    
    print(f"Cargando pesos finales desde: {ckpt3_path}")
    iou_aware_model.load_weights(ckpt3_path)
    
    # Asignar el modelo interno final para la evaluación
    eval_model = iou_aware_model.seg_model

# 10) Evaluación & visualización ----------------------------------------------
print("\n--- Evaluación del modelo final en CPU/GPU disponible ---")
def evaluate_model(model_to_eval,X,y):
    preds = model_to_eval.predict(X)
    mask = np.argmax(preds,axis=-1)
    ious = []
    for cls in range(num_classes):
        yt = (y==cls).astype(int)
        yp = (mask==cls).astype(int)
        inter = np.sum(yt*yp)
        union = np.sum(yt)+np.sum(yp)-inter
        ious.append(inter/union if union>0 else 0)
    return {'mean_iou':np.mean(ious),'class_ious':ious,'pixel_accuracy':np.mean(mask==y)}

metrics = evaluate_model(eval_model,val_X,val_y)
print(f"Mean IoU: {metrics['mean_iou']:.4f}")
print(f"Pixel Accuracy: {metrics['pixel_accuracy']:.4f}")
for i,iou in enumerate(metrics['class_ious']): print(f" Class {i}: {iou:.4f}")

def visualize_predictions(model_to_eval,X,y,num_samples=5):
    idxs = np.random.choice(len(X),num_samples,replace=False)
    preds = model_to_eval.predict(X[idxs])
    masks = np.argmax(preds,axis=-1)
    cmap = plt.cm.get_cmap('tab10',num_classes)
    plt.figure(figsize=(12,4*num_samples))
    for i,ix in enumerate(idxs):
        plt.subplot(num_samples,3,3*i+1); plt.imshow(X[ix]); plt.axis('off'); plt.title('Image')
        plt.subplot(num_samples,3,3*i+2); plt.imshow(y[ix],cmap=cmap,vmin=0,vmax=num_classes-1); plt.axis('off'); plt.title('GT Mask')
        plt.subplot(num_samples,3,3*i+3); plt.imshow(masks[i],cmap=cmap,vmin=0,vmax=num_classes-1); plt.axis('off'); plt.title('Pred Mask')
    plt.tight_layout()
    plt.savefig('predictions_visualization_iou_optimized.png')
    plt.close()
    print("\nVisualización de predicciones guardada en 'predictions_visualization_iou_optimized.png'")

visualize_predictions(eval_model,val_X,val_y)
print("\nEntrenamiento y evaluación completados.")