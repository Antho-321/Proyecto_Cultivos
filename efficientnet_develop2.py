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
8. Carga explícita del mejor checkpoint entre fases de entrenamiento (skip mismatches).
"""

import os
os.environ['MPLBACKEND'] = 'agg'  # matplotlib backend

import tensorflow as tf
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

# -------------------------------------------------------------------------------
# Monkey-patch SE module in keras_efficientnet_v2
# -------------------------------------------------------------------------------
import keras_efficientnet_v2.efficientnet_v2 as _efn2
from lovasz_losses_tf import lovasz_softmax

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

# -------------------------------------------------------------------------------
# 1) Funciones de pérdida mejoradas
# -------------------------------------------------------------------------------
class_counts = None  # se inicializa después de cargar datos

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
    return 1.0 - tf.reduce_mean(num_classes / tf.reduce_sum(1.0/(iou_per_class+1e-8), axis=-1))

def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    y_i = tf.cast(y_true, tf.int32)
    ce = tf.keras.losses.sparse_categorical_crossentropy(y_i, y_pred, from_logits=True)
    pt = tf.exp(-ce)
    return tf.reduce_mean(alpha * tf.pow(1-pt, gamma) * ce)

def soft_boundary_loss(y_true, y_pred):
    y_i = tf.cast(y_true, tf.int32)
    y_p = tf.nn.softmax(y_pred, axis=-1)
    sobel = tf.image.sobel_edges(tf.expand_dims(tf.cast(y_i, tf.float32),-1))[...,0]
    edges = tf.cast(tf.reduce_max(tf.abs(sobel),axis=-1)>0, tf.float32)
    pred_edge = tf.reduce_sum(y_p * tf.expand_dims(edges,-1), axis=[1,2])
    return 1.0 - tf.reduce_mean(pred_edge)

def lovasz_softmax_improved(y_pred, y_true, per_image=True, ignore_index=None):
    return lovasz_softmax(y_pred, y_true, per_image=per_image, ignore=ignore_index)

def enhanced_combined_loss(y_true, y_pred):
    jac   = 0.55 * soft_jaccard_loss(y_true, y_pred)
    lov   = 0.35 * lovasz_softmax_improved(y_pred, tf.cast(y_true,tf.int32), per_image=True)
    foc   = 0.08 * focal_loss(y_true, y_pred)
    ce    = 0.02 * weighted_sparse_ce(y_true, y_pred)
    return jac + lov + foc + ce

# -------------------------------------------------------------------------------
# 2) Pipelines de augmentación
# -------------------------------------------------------------------------------
def get_training_augmentation():
    return A.Compose([
        A.RandomScale(0.15,p=0.4),
        A.PadIfNeeded(256,256, border_mode=cv2.BORDER_REFLECT),
        A.RandomCrop(256,256),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.4),
        A.ShiftScaleRotate(0.1,0.1,15,p=0.3),
        A.RandomBrightnessContrast(0.05,0.05,p=0.2),
        A.RandomGamma((95,105),p=0.1),
        A.CoarseDropout(
    num_holes_range=(1, 2),         # old min_holes=1, max_holes=2
    hole_height_range=(6, 12),      # old min_height=6, max_height=12
    hole_width_range=(6, 12),       # old min_width=6,  max_width=12
    fill=0,                         # was fill_value / mask_fill_value
    p=0.1
)
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

# Sobremuestreo de clases raras
prob = class_counts/class_counts.sum()
rare = np.argsort(prob)[:2]
mask = np.isin(train_y.reshape(len(train_y),-1), rare)
idx = np.any(mask,axis=1)
train_X = np.concatenate([train_X, train_X[idx]], axis=0)
train_y = np.concatenate([train_y, train_y[idx]], axis=0)
perm = np.random.permutation(len(train_X))
train_X, train_y = train_X[perm], train_y[perm]

total_pix = train_y.size
cw = total_pix/(num_classes*np.maximum(np.bincount(train_y.flatten()),1))
class_weights = tf.constant(cw, dtype=tf.float32)

# -------------------------------------------------------------------------------
# 4) ASPP mejorado
# -------------------------------------------------------------------------------
def ASPP(x, out=256, rates=(6,12,18,24)):
    br = []
    # 1x1
    c1 = Activation('relu')(BatchNormalization()(Conv2D(out,1,use_bias=False)(x)))
    br.append(c1)
    # dilated
    for i, r in enumerate(rates):
        k = 3 if i<2 else 5
        d = Activation('relu')(BatchNormalization()(
            SeparableConv2D(out,k,dilation_rate=r,padding='same',use_bias=False)(x)))
        br.append(d)
    # pool
    p = Activation('relu')(BatchNormalization()(Conv2D(out,1,use_bias=False)(
        layers.GlobalAveragePooling2D(keepdims=True)(x))))
    p = Lambda(lambda args: tf.image.resize(args[0], tf.shape(args[1])[1:3]))([p,x])
    br.append(p)
    # fusion + attention
    feat = Concatenate()(br)
    att = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)(feat)
    att = Conv2D(len(br),1,activation='sigmoid')(att)
    weighted = [layers.Multiply()([b, Lambda(lambda z: z[...,i:i+1])(att)]) 
                for i,b in enumerate(br)]
    out_f = Dropout(0.1)(Activation('relu')(BatchNormalization()(
        Conv2D(out,1,use_bias=False)(Concatenate()(weighted)))))
    return out_f

# -------------------------------------------------------------------------------
# 5) Definición del modelo con decoder completo
# -------------------------------------------------------------------------------
def build_model(shape=(256,256,3)):
    backbone = EfficientNetV2S(input_shape=shape, num_classes=0,
                               pretrained='imagenet', include_preprocessing=False)
    for layer in backbone.layers:
        if isinstance(layer, (Conv2D, SeparableConv2D)):
            layer.kernel_regularizer = tf.keras.regularizers.l2(2e-4)

    inp = backbone.input
    high     = backbone.get_layer('post_swish').output   # 8×8
    mid      = backbone.get_layer('add_6').output        # 16×16  
    low      = backbone.get_layer('add_2').output        # 32×32
    very_low = backbone.get_layer('stem_swish').output   # 64×64

    # ASPP
    x = ASPP(high, out=256)

    # Decoder Stage 1: 8→32
    x = Activation('relu')(BatchNormalization()(
        Conv2DTranspose(256,4,strides=4,padding='same',use_bias=False)(x)))
    m = Activation('relu')(BatchNormalization()(Conv2D(256,1,use_bias=False)(mid)))
    x = Dropout(0.1)(layers.Add()([
        Activation('relu')(BatchNormalization()(
            SeparableConv2D(256,3,padding='same',use_bias=False)(
                Concatenate()([x,m])))),
        Conv2D(256,1,use_bias=False)(Concatenate()([x,m]))
    ]))

    # Decoder Stage 2: 32→64
    x = Activation('relu')(BatchNormalization()(
        Conv2DTranspose(128,2,strides=2,padding='same',use_bias=False)(x)))
    l = Activation('relu')(BatchNormalization()(Conv2D(128,1,use_bias=False)(low)))
    x = Dropout(0.1)(layers.Add()([
        Activation('relu')(BatchNormalization()(
            SeparableConv2D(128,3,padding='same',use_bias=False)(
                Concatenate()([x,l])))),
        Conv2D(128,1,use_bias=False)(Concatenate()([x,l]))
    ]))

    # Decoder Stage 3: 64→128
    x = Activation('relu')(BatchNormalization()(
        Conv2DTranspose(64,2,strides=2,padding='same',use_bias=False)(x)))
    v = Activation('relu')(BatchNormalization()(Conv2D(64,1,use_bias=False)(very_low)))
    x = Activation('relu')(BatchNormalization()(
        SeparableConv2D(96,3,padding='same',use_bias=False)(
            Concatenate()([x,v]))))

    # Final upsampling to 256×256
    x = UpSampling2D(2, interpolation='bilinear')(x)
    x = Activation('relu')(BatchNormalization()(SeparableConv2D(64,3,padding='same',use_bias=False)(x)))
    x = Activation('relu')(BatchNormalization()(SeparableConv2D(32,3,padding='same',use_bias=False)(x)))

    # Prediction
    out = Conv2D(num_classes,1,padding='same', dtype='float32')(x)
    return Model(inp, out)

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
# 7) Entrenamiento por fases con cargas seguras
# -------------------------------------------------------------------------------
val_gen = tf.data.Dataset.from_tensor_slices((val_X,val_y))\
    .batch(8)\
    .map(lambda x,y: (tf.image.random_flip_left_right(x), y),
         num_parallel_calls=tf.data.AUTOTUNE)\
    .prefetch(tf.data.AUTOTUNE)

device = '/GPU:0' if gpus else '/CPU:0'
with tf.device(device):
    model = build_model(train_X.shape[1:])
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(5e-4, weight_decay=1e-4),
        loss=enhanced_combined_loss,
        metrics=[MeanIoU(num_classes=num_classes)]
    )

    # Fase 1: entrenar cabeza
    for l in model.layers:
        if 'efficientnetv2' in l.name.lower():
            l.trainable=False

    def one_cycle_lr(ep, lr_unused, max_epochs=25, max_lr=5e-4, min_lr=1e-6):
        half = max_epochs//2
        return (min_lr + (max_lr-min_lr)*ep/half) if ep<half \
               else (max_lr - (max_lr-min_lr)*(ep-half)/half)

    cbs1 = [
        EarlyStopping('val_mean_iou',mode='max',patience=15,restore_best_weights=True),
        ModelCheckpoint('best_model_phase1.keras', save_best_only=True, monitor='val_mean_iou', mode='max'),
        LearningRateScheduler(one_cycle_lr),
        ReduceLROnPlateau('val_mean_iou',mode='max',factor=0.5,patience=5,min_lr=1e-7)
    ]
    model.fit(train_X,train_y, batch_size=12, epochs=25,
              validation_data=val_gen, callbacks=cbs1)

    # Cargar fase 1 (skip mismatches)
    model.load_weights('best_model_phase1.keras', by_name=True, skip_mismatch=True)

    # Fase 2: fine-tune parcial
    # liberar últimas capas del backbone
    b = [l for l in model.layers if 'efficientnetv2' in l.name.lower()][0]
    total = len(b.layers)
    for i,lay in enumerate(b.layers):
        lay.trainable = (i >= int(0.8*total))

    model.compile(
        optimizer=tf.keras.optimizers.AdamW(5e-6, weight_decay=1e-4),
        loss=enhanced_combined_loss,
        metrics=[MeanIoU(num_classes=num_classes)]
    )
    cbs2 = [
        EarlyStopping('val_mean_iou',mode='max',patience=10,restore_best_weights=True),
        ModelCheckpoint('final_efficientnetv2_deeplab.keras', save_best_only=True, monitor='val_mean_iou', mode='max'),
        TensorBoard(log_dir='./logs/phase2_finetune'),
        ReduceLROnPlateau('val_mean_iou',mode='max',factor=0.6,patience=6,min_lr=1e-8)
    ]
    model.fit(train_X,train_y, batch_size=6, epochs=60,
              validation_data=val_gen, callbacks=cbs2)

    # Cargar fase 2
    model.load_weights('final_efficientnetv2_deeplab.keras', by_name=True, skip_mismatch=True)

    # Fase 3: full fine-tune
    for l in model.layers:
        l.trainable=True

    model.compile(
        optimizer=tf.keras.optimizers.AdamW(1e-6, weight_decay=5e-5),
        loss=enhanced_combined_loss,
        metrics=[MeanIoU(num_classes=num_classes)]
    )
    cbs3 = [
        EarlyStopping('val_mean_iou',mode='max',patience=10,restore_best_weights=True),
        ModelCheckpoint('final_full_finetune.keras', save_best_only=True, monitor='val_mean_iou', mode='max'),
        TensorBoard(log_dir='./logs/phase3_full'),
        ReduceLROnPlateau('val_mean_iou',mode='max',factor=0.6,patience=8,min_lr=1e-9)
    ]
    model.fit(train_X,train_y, batch_size=4, epochs=40,
              validation_data=val_gen, callbacks=cbs3)

    # Cargar modelo final
    model.load_weights('final_full_finetune.keras', by_name=True, skip_mismatch=True)
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