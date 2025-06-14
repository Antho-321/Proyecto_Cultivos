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
    Lambda, Concatenate, UpSampling2D, ReLU, Add, GlobalAveragePooling2D,
    Reshape, Dense, Multiply, GlobalMaxPooling2D
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
)
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
from iou_por_clase import print_class_iou
import numpy as np

# -------------------------------------------------------------------------------
# Monkey-patch para usar un Lambda layer en SE module de keras_efficientnet_v2
# -------------------------------------------------------------------------------
import keras_efficientnet_v2.efficientnet_v2 as _efn2
from lovasz_losses_tf import lovasz_softmax
from distribucion_por_clase import calculate_class_weights

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
        A.CoarseDropout(
            num_holes_range=(8, 8),
            hole_height_range=(16, 32),
            hole_width_range=(16, 32),
            fill_value=0,
            p=0.2,
        )
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

weights_dict_silent = calculate_class_weights(train_y, verbose=False)

# 5) Número de clases & modelo ----------------------------------------------
num_classes = int(np.max(train_y) + 1)
print(f"\nNúmero de clases detectado: {num_classes}")

def focal_loss(alpha=None, gamma=2.0):
    def loss(y_true, y_pred):
        # 1) Asegúrate de que y_true es [B,H,W]
        y_true = tf.cast(y_true, tf.int32)
        # 2) Crea one-hot [B,H,W,C]
        y_true_oh = tf.one_hot(y_true, depth=y_pred.shape[-1])
        # 3) Ahora sí p_t coincide en [B,H,W]
        p_t = tf.reduce_sum(y_true_oh * y_pred, axis=-1) + 1e-7
        modulating_factor = tf.pow(1.0 - p_t, gamma)
        if alpha is not None:
            alpha_t = tf.reduce_sum(y_true_oh * tf.constant(alpha, dtype=tf.float32), axis=-1)
        else:
            alpha_t = 1.0
        loss = -alpha_t * modulating_factor * tf.math.log(p_t)
        return tf.reduce_mean(loss)
    return loss

def weighted_focal_loss(class_weights, gamma=2.0):
    alpha = [class_weights[i] for i in sorted(class_weights)]
    return focal_loss(alpha=alpha, gamma=gamma)

# 6) Definición del Modelo (Arquitectura) ------------------------------------

def SqueezeAndExcitation(tensor, ratio=16, name="se"):
    """Bloque de atención Squeeze-and-Excitation (SE)."""
    filters = tensor.shape[-1]
    se = layers.GlobalAveragePooling2D(name=f"{name}_gap")(tensor)
    se = layers.Dense(filters // ratio, activation="relu", use_bias=False,
                      name=f"{name}_reduce")(se)
    se = layers.Dense(filters, activation="sigmoid", use_bias=False,
                      name=f"{name}_expand")(se)
    se = layers.Multiply(name=f"{name}_scale")([tensor, se])
    return se


def conv_block(x, filters, kernel_size, strides=1, padding='same', dilation_rate=1): # <-- Add dilation_rate here
    """Un bloque convolucional que soporta dilatación, con Conv2D, BatchNorm y Swish."""
    x = Conv2D(
        filters, 
        kernel_size, 
        strides=strides, 
        padding=padding, 
        dilation_rate=dilation_rate, # <-- Pass it to Conv2D
        use_bias=False
    )(x)
    x = BatchNormalization()(x)
    x = Activation('swish')(x)
    return x

def WASP(x,
         out_channels=256,
         dilation_rates=(1, 2, 4, 8),
         use_global_pool=True,
         anti_gridding=True,
         use_attention=False,
         name="WASP"):
    """
    Waterfall Atrous Spatial Pooling.
    
    Args
    ----
    x : 4-D tensor
        Entrada (feature map) de forma (B, H, W, C).
    out_channels : int
        Nº de filtros en cada rama convolucional.
    dilation_rates : iterable[int]
        Tasas de dilatación (p.e. (2,4,6)). Se aplican en cascada:  
        cada rama recibe la salida de la anterior (“waterfall”).
    use_global_pool : bool
        Añade rama de pooling global + 1×1 conv (estilo ASPP).
    anti_gridding : bool
        Si r > 1, aplica DepthwiseConv 3×3 extra para suavizar artefactos.
    use_attention : bool
        Aplica un bloque SE al tensor fusionado.
    name : str
        Prefijo para los nombres de capa.
    
    Returns
    -------
    y : 4-D tensor
        Mapa de características fusionado.
    """
    convs = []

    # 0) Rama 1×1 (sin dilatación) — punto de partida para la cascada
    y = layers.Conv2D(out_channels, 1, padding="same", use_bias=False,
                      name=f"{name}_conv1x1")(x)
    y = layers.BatchNormalization(name=f"{name}_bn1")(y)
    y = layers.Activation("relu", name=f"{name}_relu1")(y)
    convs.append(y)
    prev = y  # ← se alimenta a la siguiente rama

    # 1) Cascada atrous
    for r in dilation_rates:
        branch = layers.Conv2D(out_channels, 3, padding="same", dilation_rate=r,
                               use_bias=False, name=f"{name}_conv_d{r}")(prev)
        branch = layers.BatchNormalization(name=f"{name}_bn_d{r}")(branch)
        branch = layers.Activation("relu", name=f"{name}_relu_d{r}")(branch)

        # Anti-gridding (opcional) — Depthwise 3×3
        if anti_gridding and r > 1:
            branch = layers.DepthwiseConv2D(3, padding="same", use_bias=False,
                                            name=f"{name}_ag_dw{r}")(branch)
            branch = layers.BatchNormalization(name=f"{name}_ag_bn{r}")(branch)
            branch = layers.Activation("relu", name=f"{name}_ag_relu{r}")(branch)

        convs.append(branch)
        prev = branch  # “waterfall”: la salida pasa a la siguiente iteración

    # 2) Rama de pooling global (estilo ASPP)
    if use_global_pool:
        gp = layers.GlobalAveragePooling2D(keepdims=True, name=f"{name}_gap")(x)
        gp = layers.Conv2D(out_channels, 1, padding="same", use_bias=False,
                           name=f"{name}_gp_conv")(gp)
        gp = layers.BatchNormalization(name=f"{name}_gp_bn")(gp)
        gp = layers.Activation("relu", name=f"{name}_gp_relu")(gp)
        # Redimensiona al tamaño espacial de `x`
        gp = layers.Lambda(lambda t: tf.image.resize(
                t[0], tf.shape(t[1])[1:3], method="bilinear"),
                name=f"{name}_gp_resize")([gp, x])
        convs.append(gp)

    # 3) Fusión por concatenación + 1×1 conv
    y = layers.Concatenate(name=f"{name}_concat")(convs)
    y = layers.Conv2D(out_channels, 1, padding="same", use_bias=False,
                      name=f"{name}_fuse_conv")(y)
    y = layers.BatchNormalization(name=f"{name}_fuse_bn")(y)
    y = layers.Activation("relu", name=f"{name}_fuse_relu")(y)

    # 4) Atención opcional
    if use_attention:
        y = SqueezeAndExcitation(y, name=f"{name}_se")

    return y

def build_model(shape=(256, 256, 3), num_classes_arg=None):
    """
    Construye un modelo de segmentación con un backbone EfficientNetV2S
    y un decodificador tipo U-Net que usa una búsqueda dinámica de capas
    y un ASPP_mejorado.
    """
    # --- Definición del backbone ---
    backbone = EfficientNetV2S(  # Remove tf.keras.applications.
        input_shape=shape,
        num_classes=0,
        pretrained='imagenet',
        include_preprocessing=False
    )

    # --- Extracción dinámica de las skip connections ---
    inp = backbone.input
    bottleneck = backbone.get_layer('post_swish').output  # Salida del encoder, 8x8

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
        
    # --- Cuello de botella con ASPP Mejorado ---
    # Se aplican tasas de dilatación pequeñas, adecuadas para el mapa de 8x8
    x = WASP(bottleneck,
         out_channels=256,
         dilation_rates=(2, 4, 6),
         use_global_pool=True,
         anti_gridding=True,
         use_attention=True,
         name="WASP")

    # --- Rama del decodificador (Upsampling con lógica U-Net) ---

    # Bloque 1: De 8x8 a 16x16
    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    x = Concatenate()([x, s4]) # Concatenar con la skip connection de 16x16
    x = conv_block(x, filters=128, kernel_size=3)
    x = conv_block(x, filters=128, kernel_size=3)

    # Bloque 2: De 16x16 a 32x32
    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    x = Concatenate()([x, s3]) # Concatenar con la skip connection de 32x32
    x = conv_block(x, filters=64, kernel_size=3)
    x = conv_block(x, filters=64, kernel_size=3)

    # Bloque 3: De 32x32 a 64x64
    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    x = Concatenate()([x, s2]) # Concatenar con la skip connection de 64x64
    x = conv_block(x, filters=48, kernel_size=3)
    x = conv_block(x, filters=48, kernel_size=3)

    # Bloque 4: De 64x64 a 128x128
    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    x = Concatenate()([x, s1]) # Concatenar con la skip connection de 128x128
    x = conv_block(x, filters=32, kernel_size=3)
    x = conv_block(x, filters=32, kernel_size=3)

    # Upsampling final para alcanzar la resolución de entrada (256x256)
    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    
    # Capa de salida
    out = Conv2D(num_classes_arg, 1, padding='same', activation='softmax', dtype='float32')(x)

    return Model(inp, out), backbone


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

# Replace your existing function with this one
def adaptive_boundary_enhanced_dice_loss_tf(y_true, y_pred, gamma=2.0, smooth=1e-7):
    """
    Pérdida de bordes basada en el Dice score, calculado sobre el gradiente
    morfológico de las máscaras.
    VERSIÓN COMPATIBLE CON XLA usando Max-Pooling en lugar de dilation/erosion.
    """
    y_true_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), depth=num_classes, axis=-1)
    y_pred_probs = tf.nn.softmax(y_pred, axis=-1)

    # Usar max_pool2d para aproximar la dilatación (XLA-friendly)
    y_true_dilated = tf.nn.max_pool2d(y_true_one_hot, ksize=3, strides=1, padding='SAME')
    
    # Usar max_pool2d con input negado para aproximar la erosión (XLA-friendly)
    y_true_eroded = -tf.nn.max_pool2d(-y_true_one_hot, ksize=3, strides=1, padding='SAME')

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

loss_focal = weighted_focal_loss(weights_dict_silent, gamma=2.0)

# 8) Modelo Personalizado con Bucle de Entrenamiento Optimizado para IoU -----
class IoUOptimizedModel(keras.Model):
    def __init__(self, shape, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.seg_model, self.backbone = build_model(shape, num_classes_arg=num_classes)

        # métricas existentes
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.iou_metric   = MeanIoU(num_classes=num_classes, name='enhanced_mean_iou')
        # *** nueva pérdida focal ***
        self.loss_focal = loss_focal

    def train_step(self, data):
        x, y_true = data
        with tf.GradientTape() as tape:
            y_pred = self.seg_model(x, training=True)
            # sustituye ultimate_iou_loss por focal_loss
            loss = self.loss_focal(y_true, y_pred)

        grads = tape.gradient(loss, self.seg_model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.seg_model.trainable_variables))

        self.loss_tracker.update_state(loss)
        self.iou_metric.update_state(y_true, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y_true = data
        y_pred = self.seg_model(x, training=False)
        loss = self.loss_focal(y_true, y_pred)
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
    # 1. Instantiate the model
    iou_aware_model = IoUOptimizedModel(shape=train_X.shape[1:], num_classes=num_classes)
    
    # 2. Add this line to explicitly build the model
    # The input shape is (batch_size, height, width, channels). We use None for the batch size
    # to indicate it can be flexible.
    iou_aware_model.build(input_shape=(None, *train_X.shape[1:]))
    
    # 3. Now the rest of your code will work as expected
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
    
    iou_aware_model.fit(train_X, train_y, batch_size=12, epochs=5,
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

    iou_aware_model.fit(train_X, train_y, batch_size=6, epochs=5,
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
    
    iou_aware_model.fit(train_X, train_y, batch_size=4, epochs=5,
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

print("Calculando IoU por clase (usando índices como etiquetas):")
class_ious = print_class_iou(
    model=eval_model, 
    X=val_X, 
    y_true=val_y
)

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