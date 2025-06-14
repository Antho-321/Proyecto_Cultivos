# -*- coding: utf-8 -*-
"""
claude_efficientnetv2_swin_deeplab_iou.py

Script completo que integra un bloque Swin Transformer después del encoder 
EfficientNetV2S. El modelo DeepLabV3+ modificado utiliza este bloque para
refinar las características antes de pasarlas al módulo WASP y al decodificador.
El entrenamiento se realiza con un bucle personalizado para optimizar 
directamente una pérdida combinada centrada en IoU.
"""

# --- 1) Imports ------------------------------------------------------------------
import os
os.environ['MPLBACKEND'] = 'agg'  # Establecer el backend explícitamente

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
import cv2
from tqdm import tqdm
import albumentations as A
from tensorflow.keras import layers, backend as K
from tensorflow.keras.layers import (
    Input, Conv2D, SeparableConv2D, Conv2DTranspose,
    BatchNormalization, Activation, Dropout,
    Lambda, Concatenate, UpSampling2D, ReLU, Add, GlobalAveragePooling2D,
    Reshape, Dense, Multiply, GlobalMaxPooling2D, LayerNormalization
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

# --- 2) Definición de los componentes Swin Transformer -------------------------

class Mlp(tf.keras.layers.Layer):
    """Capa de Perceptrón Multicapa (Feed-Forward Network) para el Swin Transformer."""
    def __init__(self, hidden_features=None, out_features=None, drop=0., **kwargs):
        super().__init__(**kwargs)
        self.fc1 = Dense(hidden_features, activation='gelu')
        self.drop1 = Dropout(drop)
        self.fc2 = Dense(out_features)
        self.drop2 = Dropout(drop)

    def call(self, x):
        x = self.fc1(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class WindowAttention(tf.keras.layers.Layer):
    """Atención Multi-Cabeza en Ventana (W-MSA) con sesgos de posición relativa."""
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None,
                 attn_drop=0., proj_drop=0., **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = Dense(dim * 3, use_bias=qkv_bias, name="qkv_proj")
        self.attn_drop = Dropout(attn_drop)
        self.proj = Dense(dim, name="attn_proj")
        self.proj_drop = Dropout(proj_drop)

        # Tabla e índices de posiciones relativas
        num_rel = (2*window_size[0]-1)*(2*window_size[1]-1)
        self.relative_position_bias_table = self.add_weight(
            shape=(num_rel, num_heads),
            initializer=tf.initializers.TruncatedNormal(stddev=0.02),
            trainable=True,
            name="relative_position_bias_table"
        )
        coords_h = tf.range(window_size[0])
        coords_w = tf.range(window_size[1])
        coords = tf.stack(tf.meshgrid(coords_h, coords_w, indexing='ij'))
        coords_flat = tf.reshape(coords, [2, -1])
        rel_coords = coords_flat[:, :, None] - coords_flat[:, None, :]
        rel_coords = tf.transpose(rel_coords, (1,2,0))
        rel_coords = rel_coords + [window_size[0]-1, window_size[1]-1]
        rel_coords = rel_coords * [(2*window_size[1]-1), 1]
        rel_index = tf.reduce_sum(rel_coords, axis=-1)
        self.relative_position_index = tf.Variable(
            rel_index, trainable=False, dtype=tf.int32, name="relative_position_index"
        )

    def call(self, x, mask=None):
        B_, N, C = tf.unstack(tf.shape(x))
        qkv = self.qkv(x)
        qkv = tf.reshape(qkv, (B_, N, 3, self.num_heads, C//self.num_heads))
        qkv = tf.transpose(qkv, (2,0,3,1,4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = tf.matmul(q, k, transpose_b=True)

        rel_bias = tf.gather(self.relative_position_bias_table,
                              tf.reshape(self.relative_position_index, [-1]))
        rel_bias = tf.reshape(rel_bias,
                               (self.window_size[0]*self.window_size[1],
                                self.window_size[0]*self.window_size[1],
                                -1))
        rel_bias = tf.transpose(rel_bias, (2,0,1))
        attn = attn + tf.expand_dims(rel_bias, 0)

        if mask is not None:
            nW = tf.shape(mask)[0]
            attn = tf.reshape(attn, (-1, nW, self.num_heads, N, N))
            attn = attn + tf.cast(tf.reshape(mask, (1,nW,1,N,N)), attn.dtype)
            attn = tf.reshape(attn, (-1, self.num_heads, N, N))

        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = tf.matmul(attn, v)
        x = tf.transpose(x, (0,2,1,3))
        x = tf.reshape(x, (B_, N, C))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinTransformerBlock(tf.keras.layers.Layer):
    """Bloque Swin Transformer que alterna entre W-MSA y SW-MSA."""
    def __init__(self, dim, input_resolution, num_heads,
                 window_size=7, shift_size=0, mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.,
                 norm_layer=LayerNormalization, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = (window_size, window_size)
        self.shift_size = (shift_size, shift_size)
        if min(input_resolution) <= window_size:
            self.shift_size = (0, 0)
            self.window_size = input_resolution

        self.norm1 = norm_layer()
        self.attn = WindowAttention(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop
        )
        self.drop_path = Dropout(drop_path) if drop_path>0 else tf.identity
        self.norm2 = norm_layer()
        self.mlp = Mlp(hidden_features=int(dim*mlp_ratio), out_features=dim, drop=drop)

        # Crear máscara de atención desplazada (si shift_size > 0)
        if any(s>0 for s in self.shift_size):
            H, W = input_resolution
            wsH, wsW = self.window_size
            ssH, ssW = self.shift_size
            img_mask = np.zeros((1, H, W, 1), dtype=np.float32)
            h_slices = (slice(0, -wsH), slice(-wsH, -ssH), slice(-ssH, None))
            w_slices = (slice(0, -wsW), slice(-wsW, -ssW), slice(-ssW, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            
            mask_windows = Lambda(
                lambda t: tf.reshape(
                    tf.transpose(
                        tf.reshape(t, (-1, H//wsH, wsH, W//wsW, wsW, 1)),
                        (0,1,3,2,4,5)
                    ),
                    (-1, wsH*wsW)
                )
            )(tf.constant(img_mask))
            attn_mask = tf.expand_dims(mask_windows,1) - tf.expand_dims(mask_windows,2)
            attn_mask = tf.where(attn_mask!=0, -100.0, 0.0)
            self.attn_mask = tf.Variable(attn_mask, trainable=False, dtype=tf.float32)
        else:
            self.attn_mask = None

    def call(self, x):
        H, W = self.input_resolution
        B, L, C = tf.unstack(tf.shape(x))
        shortcut = x
        x = self.norm1(x)
        x = Reshape((H, W, C))(x)

        # shift
        if any(s>0 for s in self.shift_size):
            ssH, ssW = self.shift_size
            x = tf.roll(x, shift=[-ssH, -ssW], axis=[1,2])

        # partition windows
        wsH, wsW = self.window_size
        x_windows = Lambda(
            lambda t: tf.reshape(
                tf.transpose(
                    tf.reshape(t, (-1, H//wsH, wsH, W//wsW, wsW, C)),
                    (0,1,3,2,4,5)
                ),
                (-1, wsH*wsW, C)
            )
        )(x)

        # W-MSA or SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # merge windows
        x = Lambda(
            lambda t: tf.reshape(
                tf.transpose(
                    tf.reshape(t, (-1, H//wsH, W//wsW, wsH, wsW, C)),
                    (0,1,3,2,4,5)
                ),
                (-1, H, W, C)
            )
        )(attn_windows)

        # reverse shift
        if any(s>0 for s in self.shift_size):
            ssH, ssW = self.shift_size
            x = tf.roll(x, shift=[ssH, ssW], axis=[1,2])

        x = Reshape((L, C))(x)
        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

# --- 3) Componentes del modelo original (WASP, etc.) --------------------------

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

def SqueezeAndExcitation(tensor, ratio=16, name="se"):
    filters = tensor.shape[-1]
    se = layers.GlobalAveragePooling2D(name=f"{name}_gap")(tensor)
    se = layers.Dense(filters // ratio, activation="relu", use_bias=False, name=f"{name}_reduce")(se)
    se = layers.Dense(filters, activation="sigmoid", use_bias=False, name=f"{name}_expand")(se)
    se = layers.Multiply(name=f"{name}_scale")([tensor, se])
    return se

def conv_block(x, filters, kernel_size, strides=1, padding='same', dilation_rate=1):
    x = Conv2D(filters, kernel_size, strides=strides, padding=padding, dilation_rate=dilation_rate, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('swish')(x)
    return x

def WASP(x, out_channels=256, dilation_rates=(1, 2, 4, 8), use_global_pool=True,
         anti_gridding=True, use_attention=False, name="WASP"):
    convs = []
    y = layers.Conv2D(out_channels, 1, padding="same", use_bias=False, name=f"{name}_conv1x1")(x)
    y = layers.BatchNormalization(name=f"{name}_bn1")(y)
    y = layers.Activation("relu", name=f"{name}_relu1")(y)
    convs.append(y)
    prev = y

    for r in dilation_rates:
        branch = layers.Conv2D(out_channels, 3, padding="same", dilation_rate=r, use_bias=False, name=f"{name}_conv_d{r}")(prev)
        branch = layers.BatchNormalization(name=f"{name}_bn_d{r}")(branch)
        branch = layers.Activation("relu", name=f"{name}_relu_d{r}")(branch)
        if anti_gridding and r > 1:
            branch = layers.DepthwiseConv2D(3, padding="same", use_bias=False, name=f"{name}_ag_dw{r}")(branch)
            branch = layers.BatchNormalization(name=f"{name}_ag_bn{r}")(branch)
            branch = layers.Activation("relu", name=f"{name}_ag_relu{r}")(branch)
        convs.append(branch)
        prev = branch

    if use_global_pool:
        gp = layers.GlobalAveragePooling2D(keepdims=True, name=f"{name}_gap")(x)
        gp = layers.Conv2D(out_channels, 1, padding="same", use_bias=False, name=f"{name}_gp_conv")(gp)
        gp = layers.BatchNormalization(name=f"{name}_gp_bn")(gp)
        gp = layers.Activation("relu", name=f"{name}_gp_relu")(gp)
        gp = layers.Lambda(lambda t: tf.image.resize(t[0], tf.shape(t[1])[1:3], method="bilinear"), name=f"{name}_gp_resize")([gp, x])
        convs.append(gp)

    y = layers.Concatenate(name=f"{name}_concat")(convs)
    y = layers.Conv2D(out_channels, 1, padding="same", use_bias=False, name=f"{name}_fuse_conv")(y)
    y = layers.BatchNormalization(name=f"{name}_fuse_bn")(y)
    y = layers.Activation("relu", name=f"{name}_fuse_relu")(y)
    if use_attention:
        y = SqueezeAndExcitation(y, name=f"{name}_se")
    return y

# --- 4) Construcción del Modelo Integrado -------------------------------------

def build_model(shape=(256, 256, 3), num_classes_arg=None):
    """
    Construye el modelo de segmentación con EfficientNetV2S, un bloque Swin Transformer,
    WASP y un decodificador tipo U-Net.
    """
    # --- Encoder ---
    backbone = EfficientNetV2S(
        input_shape=shape,
        num_classes=0,
        pretrained='imagenet',
        include_preprocessing=False
    )
    inp = backbone.input
    bottleneck = backbone.get_layer('post_swish').output  # Salida del encoder, e.g., (B, 8, 8, C)

    # --- Extracción dinámica de las skip connections ---
    s1, s2, s3, s4 = None, None, None, None
    for layer in reversed(backbone.layers):
        if 'add' in layer.name:
            out_shape = layer.output.shape[1]
            if out_shape == 128 and s1 is None: s1 = layer.output
            elif out_shape == 64 and s2 is None: s2 = layer.output
            elif out_shape == 32 and s3 is None: s3 = layer.output
            elif out_shape == 16 and s4 is None: s4 = layer.output
    if any(s is None for s in [s1, s2, s3, s4]):
        raise ValueError("No se pudieron encontrar todas las capas de skip connection requeridas.")

    # --- Bloque Intermedio: Swin Transformer ---
    H, W, C = map(int, bottleneck.shape[1:])
    
    # Preparamos la entrada para el Swin Transformer (de 4D a 3D)
    x = Reshape((H*W, C), name="swin_in_reshape")(bottleneck)
    
    # Aplicamos un bloque Swin Transformer. 
    # Usamos shift_size=0 para un bloque sin desplazamiento (W-MSA).
    # window_size=2 es apropiado para un mapa de 8x8.
    x = SwinTransformerBlock(
        dim=C,
        input_resolution=(H, W),
        num_heads=4,
        window_size=2,
        shift_size=0, # Podrías alternar 0 y window_size//2 si apilas varios bloques
        mlp_ratio=4.,
        name="swin_block_1"
    )(x)
    
    # Devolvemos la salida a su forma 4D para el resto del modelo
    x = Reshape((H, W, C), name="swin_out_reshape")(x)

    # --- Cuello de botella con WASP ---
    x = WASP(
        x,  # La entrada a WASP ahora son las características refinadas por Swin
        out_channels=256,
        dilation_rates=(2, 4, 6),
        use_global_pool=True,
        anti_gridding=True,
        use_attention=True,
        name="WASP"
    )

    # --- Decoder (U-Net) ---
    # Bloque 1: De 8x8 -> 16x16
    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    x = Concatenate()([x, s4])
    x = conv_block(x, filters=128, kernel_size=3)
    x = conv_block(x, filters=128, kernel_size=3)

    # Bloque 2: De 16x16 -> 32x32
    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    x = Concatenate()([x, s3])
    x = conv_block(x, filters=64, kernel_size=3)
    x = conv_block(x, filters=64, kernel_size=3)

    # Bloque 3: De 32x32 -> 64x64
    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    x = Concatenate()([x, s2])
    x = conv_block(x, filters=48, kernel_size=3)
    x = conv_block(x, filters=48, kernel_size=3)

    # Bloque 4: De 64x64 -> 128x128
    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    x = Concatenate()([x, s1])
    x = conv_block(x, filters=32, kernel_size=3)
    x = conv_block(x, filters=32, kernel_size=3)

    # Upsampling final
    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    
    # --- Capa de Salida ---
    out = Conv2D(num_classes_arg, 1, padding='same', activation='softmax', dtype='float32')(x)

    return Model(inp, out), backbone

# ---------------------------------------------------------------------------------
# El resto del script (carga de datos, pérdidas, entrenamiento) es idéntico al original.
# ---------------------------------------------------------------------------------

# 5) Carga de datos
print("Cargando datos...")
train_X, train_y = load_augmented_dataset('Balanced/train/images', 'Balanced/train/masks', target_size=(256, 256), augment=True)
val_X,   val_y   = load_augmented_dataset('Balanced/val/images',   'Balanced/val/masks',   target_size=(256, 256), augment=False)

# 6) Número de clases
num_classes = int(np.max(train_y) + 1)
print(f"\nNúmero de clases detectado: {num_classes}")

# 7) Pérdidas y Métricas
def tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3, smooth=1e-7):
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
    y_true_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), depth=num_classes, axis=-1)
    y_pred_probs = tf.nn.softmax(y_pred, axis=-1)
    y_true_dilated = tf.nn.max_pool2d(y_true_one_hot, ksize=3, strides=1, padding='SAME')
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
    lov = 0.50 * lovasz_softmax(y_pred, tf.cast(y_true, tf.int32), per_image=True)
    abe_dice = 0.30 * adaptive_boundary_enhanced_dice_loss_tf(y_true, y_pred, gamma=2.0)
    tver = 0.20 * tversky_loss(y_true, y_pred, alpha=0.4, beta=0.6)
    return lov + abe_dice + tver

class MeanIoU(tf.keras.metrics.Metric):
    def __init__(self, num_classes, name='mean_iou', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.total_cm = self.add_weight(shape=(num_classes, num_classes), name='total_confusion_matrix', initializer='zeros')
    def update_state(self, y_true, y_pred, sample_weight=None):
        preds = tf.argmax(y_pred, axis=-1)
        if len(y_true.shape) == 4 and y_true.shape[-1] == 1:
            y_true = tf.squeeze(y_true, axis=-1)
        y_t = tf.reshape(tf.cast(y_true, tf.int32), [-1])
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

# 8) Modelo Personalizado para Entrenamiento
class IoUOptimizedModel(tf.keras.Model):
    def __init__(self, shape, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.seg_model, self.backbone = build_model(shape, num_classes_arg=num_classes)
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
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

# 9) Entrenamiento por Fases
print(f"\n--- Iniciando entrenamiento en el dispositivo: {device} ---")
checkpoint_dir = './checkpoints_swin'
os.makedirs(checkpoint_dir, exist_ok=True)
ckpt1_path = os.path.join(checkpoint_dir, 'phase1_swin_model.keras')
ckpt2_path = os.path.join(checkpoint_dir, 'phase2_swin_model.keras')
ckpt3_path = os.path.join(checkpoint_dir, 'phase3_swin_model.keras')
val_gen = (val_X, val_y)

with tf.device(device):
    iou_aware_model = IoUOptimizedModel(shape=train_X.shape[1:], num_classes=num_classes)
    iou_aware_model.build(input_shape=(None, *train_X.shape[1:]))
    backbone = iou_aware_model.backbone
    iou_aware_model.seg_model.summary()
    
    MONITOR_METRIC = 'val_enhanced_mean_iou'
    
    # Fase 1
    print("\n--- Fase 1: Entrenando solo el decoder y Swin (backbone congelado) ---")
    backbone.trainable = False
    iou_aware_model.compile(optimizer=tf.keras.optimizers.AdamW(5e-4, weight_decay=1e-4))
    cbs1 = [
        ModelCheckpoint(ckpt1_path, save_best_only=True, monitor=MONITOR_METRIC, mode='max', verbose=1),
        EarlyStopping(monitor=MONITOR_METRIC, mode='max', patience=10, restore_best_weights=False),
        TensorBoard(log_dir='./logs/swin_phase1', update_freq='epoch')
    ]
    iou_aware_model.fit(train_X, train_y, batch_size=12, epochs=20, validation_data=val_gen, callbacks=cbs1)
    iou_aware_model.load_weights(ckpt1_path)

    # Fase 2
    print("\n--- Fase 2: Fine-tuning del 20% superior del backbone ---")
    backbone.trainable = True
    fine_tune_at = int(len(backbone.layers) * 0.80)
    for layer in backbone.layers[:fine_tune_at]: layer.trainable = False
    iou_aware_model.compile(optimizer=tf.keras.optimizers.AdamW(2e-5, weight_decay=1e-4))
    cbs2 = [
        ModelCheckpoint(ckpt2_path, save_best_only=True, monitor=MONITOR_METRIC, mode='max', verbose=1),
        EarlyStopping(monitor=MONITOR_METRIC, mode='max', patience=12, restore_best_weights=False),
        TensorBoard(log_dir='./logs/swin_phase2', update_freq='epoch')
    ]
    iou_aware_model.fit(train_X, train_y, batch_size=6, epochs=20, validation_data=val_gen, callbacks=cbs2)
    iou_aware_model.load_weights(ckpt2_path)

    # Fase 3
    print("\n--- Fase 3: Fine-tuning de todo el modelo ---")
    for layer in backbone.layers: layer.trainable = True
    iou_aware_model.compile(optimizer=tf.keras.optimizers.AdamW(5e-6, weight_decay=5e-5))
    cbs3 = [
        ModelCheckpoint(ckpt3_path, save_best_only=True, monitor=MONITOR_METRIC, mode='max', verbose=1),
        EarlyStopping(monitor=MONITOR_METRIC, mode='max', patience=15, restore_best_weights=False),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1),
        TensorBoard(log_dir='./logs/swin_phase3', update_freq='epoch')
    ]
    iou_aware_model.fit(train_X, train_y, batch_size=4, epochs=20, validation_data=val_gen, callbacks=cbs3)
    iou_aware_model.load_weights(ckpt3_path)
    eval_model = iou_aware_model.seg_model

# 10) Evaluación y Visualización
print("\n--- Evaluación del modelo final ---")
def evaluate_model(model_to_eval, X, y):
    preds = model_to_eval.predict(X, batch_size=4)
    mask = np.argmax(preds, axis=-1)
    ious = []
    for cls in range(num_classes):
        yt, yp = (y==cls).astype(int), (mask==cls).astype(int)
        inter = np.sum(yt * yp)
        union = np.sum(yt) + np.sum(yp) - inter
        ious.append(inter / union if union > 0 else 0)
    return {'mean_iou': np.mean(ious), 'class_ious': ious, 'pixel_accuracy': np.mean(mask == y)}

metrics = evaluate_model(eval_model, val_X, val_y)
print(f"Mean IoU: {metrics['mean_iou']:.4f}")
print(f"Pixel Accuracy: {metrics['pixel_accuracy']:.4f}")
for i, iou in enumerate(metrics['class_ious']): print(f" Class {i}: {iou:.4f}")

def visualize_predictions(model_to_eval, X, y, num_samples=5):
    idxs = np.random.choice(len(X), num_samples, replace=False)
    preds = model_to_eval.predict(X[idxs])
    masks = np.argmax(preds, axis=-1)
    cmap = plt.cm.get_cmap('tab10', num_classes)
    plt.figure(figsize=(12, 4 * num_samples))
    for i, ix in enumerate(idxs):
        plt.subplot(num_samples, 3, 3 * i + 1); plt.imshow(X[ix]); plt.axis('off'); plt.title('Image')
        plt.subplot(num_samples, 3, 3 * i + 2); plt.imshow(y[ix], cmap=cmap, vmin=0, vmax=num_classes - 1); plt.axis('off'); plt.title('GT Mask')
        plt.subplot(num_samples, 3, 3 * i + 3); plt.imshow(masks[i], cmap=cmap, vmin=0, vmax=num_classes - 1); plt.axis('off'); plt.title('Pred Mask')
    plt.tight_layout()
    plt.savefig('predictions_visualization_swin_optimized.png')
    plt.close()
    print("\nVisualización de predicciones guardada en 'predictions_visualization_swin_optimized.png'")

visualize_predictions(eval_model, val_X, val_y)
print("\nEntrenamiento y evaluación completados.")