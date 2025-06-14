# -*- coding: utf-8 -*-
"""
claude_efficientnetv2_deeplab_iou_optimized.py

Entrena DeepLabV3+ con EfficientNetV2S como backbone usando un
bucle de entrenamiento personalizado (IoUOptimizedModel).
"""

# -----------------------------------------------------------------------------
# 1) Imports y configuración de entorno
# -----------------------------------------------------------------------------
import os
os.environ['MPLBACKEND'] = 'agg'

# --- TensorFlow primerísimo ---
import tensorflow as tf
print("Versión de TensorFlow:", tf.__version__)

# Configuración de GPU/CPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    device = '/GPU:0'
else:
    device = '/CPU:0'
print(f"Dispositivo elegido para entrenamiento: {device}")

# --- Resto de imports ---
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import cv2
from tqdm import tqdm
import albumentations as A
from tensorflow import keras
from tensorflow.keras import layers, backend as K
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, Activation, Lambda,
    Concatenate, UpSampling2D, GlobalAveragePooling2D, Dense, Multiply
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
)
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

# Para type hints
from typing import Union, List, Dict

# -----------------------------------------------------------------------------
# 2) Monkey-patch para SE en EfficientNetV2
# -----------------------------------------------------------------------------
import keras_efficientnet_v2.efficientnet_v2 as _efn2
from keras_efficientnet_v2 import EfficientNetV2S

def patched_se_module(inputs, se_ratio=0.25, name=""):
    df = K.image_data_format()
    h_axis, w_axis = (1,2) if df=='channels_last' else (2,3)
    se = Lambda(lambda x: tf.reduce_mean(x, axis=[h_axis,w_axis], keepdims=True),
                name=name+"se_reduce_pool")(inputs)
    ch = inputs.shape[-1]
    reduced = max(1, int(ch/se_ratio))
    se = Conv2D(reduced,1,padding='same',name=name+"se_reduce_conv")(se)
    se = Activation('swish',name=name+"se_reduce_act")(se)
    se = Conv2D(ch,1,padding='same',name=name+"se_expand_conv")(se)
    se = Activation('sigmoid',name=name+"se_excite")(se)
    return Multiply(name=name+"se_excite_mul")([inputs,se])

_efn2.se_module = patched_se_module

# -----------------------------------------------------------------------------
# 4) Carga de datos
# -----------------------------------------------------------------------------
def load_dataset(image_directory, mask_directory):
    images = []
    masks  = []

    for image_filename in os.listdir(image_directory):
        if image_filename.lower().endswith(('.jpg', '.png')):
            # Cargar imagen
            img_path = os.path.join(image_directory, image_filename)
            img = load_img(img_path)
            images.append(img_to_array(img))

            # Cargar máscara asociada
            name_wo_ext   = os.path.splitext(image_filename)[0]
            mask_filename = f"{name_wo_ext}_mask.png"
            mask_path     = os.path.join(mask_directory, mask_filename)

            if os.path.exists(mask_path):
                mask = load_img(mask_path, color_mode='grayscale')
                m = img_to_array(mask).astype(np.int32)
                masks.append(np.squeeze(m, axis=-1))  # (H, W)
            else:
                print(f"No se encontró la máscara para: {image_filename}")

    # Convertimos YA aquí en arrays y devolvemos
    images = np.array(images)     # shape (N, H, W, C)
    masks  = np.array(masks)      # shape (N, H, W)
    return images, masks

# -----------------------------------------------------------------------------
# 5) Cálculo de pesos de clase
# -----------------------------------------------------------------------------
def calculate_class_weights(
    masks: Union[np.ndarray, List[np.ndarray]],
    verbose: bool=True
) -> Dict[int, float]:
    arr = np.array(masks)
    flat = arr.flatten()
    classes, counts = np.unique(flat, return_counts=True)
    n_classes = int(flat.max())+1
    total = flat.size

    weights: Dict[int,float] = {}
    if verbose: print("\n--- Pesos por clase ---")
    for cls, cnt in zip(classes, counts):
        w = total/(n_classes*cnt)
        weights[cls] = w
        if verbose:
            pct = cnt/total*100
            print(f"Clase {cls}: peso={w:.4f}  ({pct:.2f}%)")
    for cls in set(range(n_classes))-set(classes):
        weights[cls] = 0.0
        if verbose: print(f"Clase {cls}: peso=0.0000  (ausente)")
    return weights

# -----------------------------------------------------------------------------
# 6) Bloques de modelo y ensamblaje
# -----------------------------------------------------------------------------
def SqueezeAndExcitation(tensor, ratio=16, name="se"):
    filters = tensor.shape[-1]
    se = GlobalAveragePooling2D(name+ "_gap")(tensor)
    se = Dense(filters//ratio, activation="relu", use_bias=False, name=name+ "_reduce")(se)
    se = Dense(filters,       activation="sigmoid", use_bias=False, name=name+ "_expand")(se)
    return Multiply(name+ "_scale")([tensor,se])

def conv_block(x, filters, kernel_size, dilation_rate=1):
    x = Conv2D(filters, kernel_size,
               dilation_rate=dilation_rate,
               padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    return Activation('swish')(x)

def WASP(x, out_channels=256, dilation_rates=(1,2,4,8),
         use_global_pool=True, anti_gridding=True,
         use_attention=False, name="WASP"):
    convs = []
    # rama 1×1
    y = Conv2D(
        out_channels,
        1,
        padding='same',
        use_bias=False,
        name = name + "_conv1x1"   # ahora es keyword, no positional
    )(x)
    y = Activation('relu',name+ "_relu1")(BatchNormalization(name+ "_bn1")(y))
    convs.append(y)
    prev = y
    # cascada dilatada
    for r in dilation_rates:
        br = Conv2D(
            out_channels,
            3,
            dilation_rate=r,
            padding='same',
            use_bias=False,
            name = name + f"_conv_d{r}"
        )(prev)
        br = Activation('relu',name+ f"_relu_d{r}")(BatchNormalization(name+ f"_bn_d{r}")(br))
        if anti_gridding and r>1:
            br = Activation('relu',name+ f"_ag_relu{r}")(
                 BatchNormalization(name+ f"_ag_bn{r}")(
                 layers.DepthwiseConv2D(3, padding='same', use_bias=False, name=name+f"_ag_dw{r}")(br)))
        convs.append(br); prev=br
    # global pool
    if use_global_pool:
        gp = GlobalAveragePooling2D(keepdims=True, name=name + "_gap_gp")(x)
        gp = Activation('relu',name+ "_gp_relu")(
             BatchNormalization(name+ "_gp_bn")(
             Conv2D(out_channels, 1, padding='same', use_bias=False, name=name+ "_gp_conv")(gp)))
        gp = Lambda(lambda t: tf.image.resize(t[0], tf.shape(t[1])[1:3], method="bilinear"),
                    name+ "_gp_resize")([gp,x])
        convs.append(gp)
    # fusión
    y = Concatenate(name+ "_concat")(convs)
    y = Activation('relu',name+ "_fuse_relu")(
         BatchNormalization(name+ "_fuse_bn")(
         Conv2D(out_channels, 1, padding='same', use_bias=False, name=name+ "_fuse_conv")(y)))
    return SqueezeAndExcitation(y, name=name+ "_se") if use_attention else y

def build_model(shape=(256,256,3), num_classes_arg=None):
    backbone = EfficientNetV2S(input_shape=shape, num_classes=0,
                               pretrained='imagenet', include_preprocessing=False)
    inp = backbone.input
    bottleneck = backbone.get_layer('post_swish').output

    s1, s2, s3, s4 = None, None, None, None

    # Búsqueda hacia atrás para encontrar las últimas capas 'add' de cada tamaño
    for layer in reversed(backbone.layers):
        if 'add' in layer.name:
            shape = layer.output.shape[1]
            if shape == 128 and s1 is None:
                s1 = layer.output  # 128x128
            elif shape == 64 and s2 is None:
                s2 = layer.output  # 64x64
            elif shape == 32 and s3 is None:
                s3 = layer.output  # 32x32
            elif shape == 16 and s4 is None:
                s4 = layer.output  # 16x16
    
    # Verificar que todas las capas fueron encontradas
    if any(s is None for s in [s1, s2, s3, s4]):
        raise ValueError("No se pudieron encontrar todas las capas de skip connection requeridas.")

    x = WASP(bottleneck, out_channels=256, dilation_rates=(2,4,6),
             use_attention=True, name="WASP")

    # decoder tipo U-Net
    for (size, skip_feat, filters) in zip(
        (16,32,64,128), (s4,s3,s2,s1), (128,64,48,32)
    ):
        x = UpSampling2D(2, interpolation='bilinear')(x)
        x = Concatenate()([x, skip_feat])
        x = conv_block(x, filters, 3)
        x = conv_block(x, filters, 3)

    x = UpSampling2D(2, interpolation='bilinear')(x)
    out = Conv2D(num_classes_arg,1, padding='same', activation='softmax', dtype='float32')(x)
    return Model(inp, out), backbone

# -----------------------------------------------------------------------------
# 7) Pérdidas personalizadas
# -----------------------------------------------------------------------------

def focal_loss(alpha=None, gamma=2.0):
    def loss_fn(y_true, y_pred):
        y_int = tf.cast(y_true[...,0],tf.int32)
        y_oh  = tf.one_hot(y_int, depth=y_pred.shape[-1])
        pt    = tf.reduce_sum(y_oh * y_pred, axis=-1) + 1e-7
        mod   = tf.pow(1-pt, gamma)
        at    = tf.reduce_sum(y_oh*tf.constant(alpha,dtype=tf.float32), axis=-1) if alpha else 1.0
        return tf.reduce_mean(-at * mod * tf.math.log(pt))
    return loss_fn

def weighted_focal_loss(class_weights, gamma=2.0):
    alpha = [class_weights[i] for i in sorted(class_weights)]
    return focal_loss(alpha=alpha, gamma=gamma)

# -----------------------------------------------------------------------------
# 8) Métrica personalizada
# -----------------------------------------------------------------------------
class MeanIoU(tf.keras.metrics.Metric):
    def __init__(self, num_classes, name='mean_iou', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.total_cm = self.add_weight(
            'total_confusion_matrix', shape=(num_classes,num_classes), initializer='zeros'
        )
    def update_state(self, y_true, y_pred, sample_weight=None):
        preds = tf.argmax(y_pred, axis=-1)
        if y_true.shape[-1]==1: y_true = tf.squeeze(y_true, -1)
        y_t = tf.reshape(tf.cast(y_true,tf.int32),[-1])
        y_p = tf.reshape(preds,[-1])
        cm = tf.math.confusion_matrix(y_t, y_p, self.num_classes, dtype=tf.float32)
        self.total_cm.assign_add(cm)
    def result(self):
        cm = self.total_cm
        tp = tf.linalg.tensor_diag_part(cm)
        sum_r = tf.reduce_sum(cm,axis=0)
        sum_c = tf.reduce_sum(cm,axis=1)
        iou = tp / (sum_r + sum_c - tp)
        return tf.reduce_mean(iou)
    def reset_states(self):
        self.total_cm.assign(tf.zeros_like(self.total_cm))

# -----------------------------------------------------------------------------
# 9) Modelo personalizado con entrenamiento IoU-aware
# -----------------------------------------------------------------------------
class IoUOptimizedModel(keras.Model):
    def __init__(self, shape, num_classes, class_weights, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.seg_model, self.backbone = build_model(shape, num_classes_arg=num_classes)
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.iou_metric   = MeanIoU(num_classes=num_classes)
        self.loss_fn      = weighted_focal_loss(class_weights, gamma=2.0)

    @property
    def metrics(self):
        return [self.loss_tracker, self.iou_metric]

    def call(self, inputs, training=False):
        return self.seg_model(inputs, training=training)

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            preds = self.seg_model(x, training=True)
            loss  = self.loss_fn(y, preds)
        grads = tape.gradient(loss, self.seg_model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.seg_model.trainable_variables))
        self.loss_tracker.update_state(loss)
        self.iou_metric.update_state(y, preds)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data
        preds = self.seg_model(x, training=False)
        loss  = self.loss_fn(y, preds)
        self.loss_tracker.update_state(loss)
        self.iou_metric.update_state(y, preds)
        return {m.name: m.result() for m in self.metrics}

# -----------------------------------------------------------------------------
# 10) Main: carga datos, calcula pesos, compila y entrena
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Carga y split
    train_X, train_y = load_dataset("Balanced/train/images",
                                              "Balanced/train/masks")
    val_X,   val_y   = load_dataset("Balanced/val/images",
                                              "Balanced/val/masks")

    # Clases y pesos
    num_classes = int(train_y.max()) + 1
    class_weights = calculate_class_weights(train_y, verbose=False)

    # Instancio modelo IoU-aware
    with tf.device(device):
        model = IoUOptimizedModel(
            shape=train_X.shape[1:],
            num_classes=num_classes,
            class_weights=class_weights
        )
        # Build explícito
        model.build(input_shape=(None, *train_X.shape[1:]))
        model.seg_model.summary()

        # --- Fase 1: head only ---
        model.backbone.trainable = False
        model.compile(optimizer=keras.optimizers.AdamW(5e-4, weight_decay=1e-4))
        callbacks = [
            ModelCheckpoint("chk1.keras", save_best_only=True, monitor="val_mean_iou", mode="max"),
            EarlyStopping(monitor="val_mean_iou", mode="max", patience=10),
            TensorBoard(log_dir="./logs/phase1")
        ]
        model.fit(train_X, train_y, validation_data=(val_X,val_y),
                  batch_size=12, epochs=5, callbacks=callbacks)

        # --- Fase 2: fine-tune top 20% ---
        model.load_weights("chk1.keras")
        model.backbone.trainable = True
        for layer in model.backbone.layers[:int(len(model.backbone.layers)*0.8)]:
            layer.trainable = False
        model.compile(optimizer=keras.optimizers.AdamW(2e-5, weight_decay=1e-4))
        model.fit(train_X, train_y, validation_data=(val_X,val_y),
                  batch_size=6, epochs=5,
                  callbacks=[ModelCheckpoint("chk2.keras", save_best_only=True, monitor="val_mean_iou", mode="max")])

        # --- Fase 3: full fine-tune ---
        model.load_weights("chk2.keras")
        for layer in model.backbone.layers:
            layer.trainable = True
        model.compile(optimizer=keras.optimizers.AdamW(5e-6, weight_decay=5e-5))
        model.fit(train_X, train_y, validation_data=(val_X,val_y),
                  batch_size=4, epochs=5,
                  callbacks=[ModelCheckpoint("chk3.keras", save_best_only=True, monitor="val_mean_iou", mode="max"),
                             ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5)])

    # -----------------------------------------------------------------------------
    # 11) Evaluación y visualización
    # -----------------------------------------------------------------------------
    def print_class_iou(model: tf.keras.Model, X: np.ndarray, Y: np.ndarray, class_names=None):
        preds = model.predict(X)
        labels = np.argmax(preds, axis=-1)
        C = model.output_shape[-1]
        if class_names is None:
            class_names = [f"Clase {i}" for i in range(C)]
        print("\n--- IoU por Clase ---")
        for i in range(C):
            inter = np.logical_and(Y==i, labels==i).sum()
            uni   = np.logical_or(Y==i, labels==i).sum()
            iou   = inter/uni if uni>0 else np.nan
            print(f"{class_names[i]}: {iou:.4f}")
        print(f"Mean IoU: {np.nanmean([inter/uni for i, (inter,uni) in enumerate(zip([np.logical_and(Y==i,labels==i).sum() for i in range(C)],[np.logical_or(Y==i,labels==i).sum() for i in range(C)]))]):.4f}")

    def visualize_predictions(model, X, Y, samples=5):
        idx = np.random.choice(len(X), samples, replace=False)
        preds = model.predict(X[idx])
        masks = np.argmax(preds, axis=-1)
        cmap = plt.cm.get_cmap('tab10', Y.max()+1)
        plt.figure(figsize=(12,4*samples))
        for i, j in enumerate(idx):
            plt.subplot(samples,3,3*i+1); plt.imshow(X[j]); plt.axis('off')
            plt.subplot(samples,3,3*i+2); plt.imshow(Y[j], cmap=cmap); plt.axis('off')
            plt.subplot(samples,3,3*i+3); plt.imshow(masks[i], cmap=cmap); plt.axis('off')
        plt.tight_layout(); plt.savefig('preds.png'); plt.close()

    # Cargar pesos finales y evaluar
    final_model = model
    final_model.load_weights("chk3.keras")
    print_class_iou(final_model, val_X, val_y)
    visualize_predictions(final_model, val_X, val_y)
