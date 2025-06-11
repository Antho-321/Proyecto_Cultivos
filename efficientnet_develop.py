# -*- coding: utf-8 -*-
"""
claude_efficientnetv2_deeplab.py
Script completo: carga datos, construye y entrena el modelo DeepLabV3+ usando EfficientNetV2S como backbone,
con monkey-patch de SE module, pérdida ponderada píxel-wise, métricas y visualización.
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
    except RuntimeError as e:
        print(e)
else:
    print("No se encontró ninguna GPU. El entrenamiento se ejecutará en la CPU.")

# --- RESTO DE IMPORTS ---
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import cv2  # Ahora cv2 se importa DESPUÉS de TensorFlow
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
# Monkey-patch para usar un Lambda layer en SE module de keras_efficientnet_v2
# -------------------------------------------------------------------------------
import keras_efficientnet_v2.efficientnet_v2 as _efn2
from lovasz_losses_tf import lovasz_hinge, lovasz_softmax

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
    
    # Redimensionar las máscaras al tamaño objetivo de entrada del modelo
    mask_target_size = (256, 256)

    with ThreadPoolExecutor() as exe:
        futures = {exe.submit(lambda fn: (
            img_to_array(load_img(os.path.join(img_dir,fn), target_size=target_size)),
            img_to_array(load_img(os.path.join(mask_dir,fn.rsplit('.',1)[0] + '_mask.png'),
                                  color_mode='grayscale', target_size=mask_target_size))
        ), fn): fn for fn in files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Loading data"):
            img_arr, mask_arr = future.result()
            if augment:
                augm = aug(image=img_arr.astype('uint8'), mask=mask_arr)
                img_arr, mask_arr = augm['image'], augm['mask']
            images.append(img_arr)
            masks.append(mask_arr)
            
    X = np.array(images, dtype='float32') / 255.0
    y = np.array(masks, dtype='int32') # Ya no se necesita np.squeeze

    # Asegurarse que `y` tenga la forma (batch, height, width) si viene con un canal extra
    if y.shape[-1] == 1:
        y = np.squeeze(y, axis=-1)
        
    return X, y

# 4) Load / split ------------------------------------------------------------
# Usar (256, 256) como tamaño de entrada para que coincida con las máscaras
train_X, train_y = load_augmented_dataset('Balanced/train/images', 'Balanced/train/masks', target_size=(256, 256), augment=True)
val_X,   val_y   = load_augmented_dataset('Balanced/val/images',   'Balanced/val/masks',   target_size=(256, 256), augment=False)

# 5) Número de clases & modelo ----------------------------------------------
num_classes = int(np.max(train_y) + 1)

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

# 6) Modelo ------------------------------------------------------------------

def build_model(shape=(256,256,3)):
    backbone = EfficientNetV2S(input_shape=shape, num_classes=0, pretrained='imagenet', include_preprocessing=False)
    for l in backbone.layers:
        if isinstance(l,(Conv2D,SeparableConv2D)):
            l.kernel_regularizer = tf.keras.regularizers.l2(1e-4)

    inp = backbone.input
    high     = backbone.get_layer('post_swish').output   # 8×8 en un input de 256x256
    mid      = backbone.get_layer('add_6').output       # 16×16
    low      = backbone.get_layer('add_2').output       # 32×32
    very_low = backbone.get_layer('stem_swish').output   # 64x64

    x = ASPP(high)                                          # 8×8 → 8×8
    x = Activation('relu')(BatchNormalization()(Conv2DTranspose(256,4,strides=4,padding='same',use_bias=False)(x)))  # 8→32

    mid_p = Activation('relu')(BatchNormalization()(Conv2D(128,1,padding='same',use_bias=False)(mid)))
    x = Activation('relu')(BatchNormalization()(SeparableConv2D(256,3,padding='same',use_bias=False)(Concatenate()([x,mid_p]))))

    x = Activation('relu')(BatchNormalization()(Conv2DTranspose(128,2,strides=2,padding='same',use_bias=False)(x)))  # 32→64
    low_p = Activation('relu')(BatchNormalization()(Conv2D(64,1,padding='same',use_bias=False)(low)))
    x = Activation('relu')(BatchNormalization()(SeparableConv2D(128,3,padding='same',use_bias=False)(Concatenate()([x,low_p]))))

    x = Activation('relu')(BatchNormalization()(Conv2DTranspose(64,2,strides=2,padding='same',use_bias=False)(x)))   # 64→128
    very_low_p = Activation('relu')(BatchNormalization()(Conv2D(48,1,padding='same',use_bias=False)(very_low)))
    x = Activation('relu')(BatchNormalization()(SeparableConv2D(96,3,padding='same',use_bias=False)(Concatenate()([x,very_low_p]))))

    # =========================================================================
    # SOLUCIÓN IMPLEMENTADA AQUÍ
    # El tensor llega con 128x128. Para llegar a 256x256 se necesita un factor x2.
    # La línea original errónea sería UpSampling2D(size=(8,8)), resultando en 1024x1024.
    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)  # 128→256
    # =========================================================================
    
    out = Conv2D(num_classes,1,padding='same',dtype='float32')(x)
    return Model(inp,out)

# 7) Pérdida y métricas ------------------------------------------------------
class_counts = np.bincount(train_y.flatten())
total_pixels = class_counts.sum()
class_weights_np = total_pixels / (num_classes * np.maximum(class_counts,1))
class_weights = tf.constant(class_weights_np, dtype=tf.float32)

def weighted_sparse_ce(y_true,y_pred):
    y_true_i = tf.cast(y_true,tf.int32)
    w = tf.gather(class_weights,y_true_i)
    ce = tf.keras.losses.sparse_categorical_crossentropy(y_true_i,y_pred,from_logits=True)
    return tf.reduce_mean(ce * w)

def soft_jaccard_loss(y_true,y_pred,smooth=1e-5):
    y_true = tf.one_hot(tf.cast(y_true,tf.int32),depth=num_classes)
    y_pred = tf.nn.softmax(y_pred,axis=-1)
    inter = tf.reduce_sum(y_true*y_pred,axis=[1,2])
    union = tf.reduce_sum(y_true+y_pred,axis=[1,2]) - inter
    jaccard = (inter+smooth)/(union+smooth)
    return 1.0 - tf.reduce_mean(jaccard)

def combined_loss(y_true,y_pred):
    ce  = 0.5 * weighted_sparse_ce(y_true,y_pred)
    jac = 0.2 * soft_jaccard_loss(y_true,y_pred)
    lov = 0.3 * lovasz_softmax(y_pred, tf.cast(y_true,tf.int32), per_image=False)
    return ce + jac + lov

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
        # Asegurarse que y_true no tenga una dimensión de canal extra
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

# 8) GPU-accelerated training ------------------------------------------------
GPU_DEVICE = '/GPU:0' if gpus else '/CPU:0'
print(f"\n--- Iniciando entrenamiento en el dispositivo: {GPU_DEVICE} ---")
with tf.device(GPU_DEVICE):
    model = build_model(input_shape=train_X.shape[1:])
    model.summary() # Imprime un resumen para verificar las formas
    print("Modelo construido y ubicado en el dispositivo.")

    # FASE 1: entrena solo la cabeza
    for layer in model.layers:
        if 'efficientnetv2' in layer.name.lower():
            layer.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=2e-4, weight_decay=1e-4),
        loss=combined_loss,
        metrics=['accuracy', MeanIoU(num_classes=num_classes)]
    )
    cbs1 = [
        EarlyStopping(monitor='val_mean_iou',mode='max',patience=8,restore_best_weights=True),
        ModelCheckpoint('phase1_head_training.keras',save_best_only=True,monitor='val_mean_iou',mode='max'),
        TensorBoard(log_dir='./logs/phase1_head',update_freq='epoch')
    ]
    model.fit(train_X,train_y,batch_size=16,epochs=15,validation_data=(val_X,val_y),callbacks=cbs1)

    # FASE 2: fine-tuning completo
    for layer in model.layers:
        layer.trainable = True
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-5, weight_decay=1e-4),
        loss=combined_loss,
        metrics=['accuracy', MeanIoU(num_classes=num_classes)]
    )
    cbs2 = [
        EarlyStopping(monitor='val_mean_iou',mode='max',patience=10,restore_best_weights=True),
        ModelCheckpoint('final_efficientnetv2_deeplab.keras',save_best_only=True,monitor='val_mean_iou',mode='max'),
        TensorBoard(log_dir='./logs/phase2_finetune',update_freq='epoch'),
        ReduceLROnPlateau(monitor='val_loss',factor=0.5,patience=5,min_lr=1e-7,verbose=1)
    ]
    model.fit(train_X,train_y,batch_size=8,epochs=50,validation_data=(val_X,val_y),callbacks=cbs2)

# 9) Evaluación & visualización ----------------------------------------------
print("\n--- Evaluación del modelo final en CPU/GPU disponible ---")
def evaluate_model(model,X,y):
    preds = model.predict(X)
    mask = np.argmax(preds,axis=-1)
    ious = []
    for cls in range(num_classes):
        yt = (y==cls).astype(int)
        yp = (mask==cls).astype(int)
        inter = np.sum(yt*yp)
        union = np.sum(yt)+np.sum(yp)-inter
        ious.append(inter/union if union>0 else 0)
    return {'mean_iou':np.mean(ious),'class_ious':ious,'pixel_accuracy':np.mean(mask==y)}
metrics = evaluate_model(model,val_X,val_y)
print(f"Mean IoU: {metrics['mean_iou']:.4f}")
print(f"Pixel Accuracy: {metrics['pixel_accuracy']:.4f}")
for i,iou in enumerate(metrics['class_ious']): print(f" Class {i}: {iou:.4f}")

def visualize_predictions(model,X,y,num_samples=3):
    idxs = np.random.choice(len(X),num_samples,replace=False)
    preds = model.predict(X[idxs])
    masks = np.argmax(preds,axis=-1)
    cmap = plt.cm.get_cmap('tab10',num_classes)
    plt.figure(figsize=(12,4*num_samples))
    for i,ix in enumerate(idxs):
        plt.subplot(num_samples,3,3*i+1); plt.imshow(X[ix]); plt.axis('off'); plt.title('Image')
        plt.subplot(num_samples,3,3*i+2); plt.imshow(y[ix],cmap=cmap,vmin=0,vmax=num_classes-1); plt.axis('off'); plt.title('GT Mask')
        plt.subplot(num_samples,3,3*i+3); plt.imshow(masks[i],cmap=cmap,vmin=0,vmax=num_classes-1); plt.axis('off'); plt.title('Pred Mask')
    plt.tight_layout()
    plt.savefig('predictions_visualization.png')
    plt.close()

visualize_predictions(model,val_X,val_y)
print("Entrenamiento y evaluación completados.")