# -*- coding: utf-8 -*-
"""
claude_efficientnetv2_swin_deeplab_iou.py

Script completo que integra un bloque Swin Transformer después del encoder 
EfficientNetV2S. El modelo DeepLabV3+ modificado utiliza este bloque para
refinar las características antes de pasarlas al módulo WASP y al decodificador.
El entrenamiento se realiza con un bucle personalizado para optimizar 
directamente una pérdida combinada centrada en IoU.

Correcciones implementadas:
1. Se usa una longitud de secuencia estática en SwinTransformerBlock para evitar `OperatorNotAllowedInGraphError`.
2. Se fuerza la creación de la variable `relative_position_index` en la GPU para evitar `InvalidArgumentError`.
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
# Asumimos que el módulo `keras_efficientnet_v2` y `lovasz_losses_tf` están instalados.
# Si no, se pueden instalar con:
# pip install keras-efficientnet-v2
# pip install git+https://github.com/bermanmaxim/LovaszSoftmax.git
try:
    import keras_efficientnet_v2.efficientnet_v2 as _efn2
    from lovasz_losses_tf import lovasz_softmax
    from keras_efficientnet_v2 import EfficientNetV2S
    
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

except ImportError as e:
    print(f"Error al importar un módulo necesario: {e}")
    print("Por favor, instala las dependencias: pip install keras-efficientnet-v2 tensorflow-addons")
    # Definimos placeholders para que el script no falle inmediatamente
    EfficientNetV2S = tf.keras.applications.EfficientNetV2S
    lovasz_softmax = lambda y_pred, y_true, **kwargs: tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)


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

        # --- SOLUCIÓN GPU: INICIO ---
        # Forzamos la creación del peso (variable no entrenable) en la GPU
        # para evitar un error de lectura de recursos entre dispositivos.
        # Nota: Esto asume que se está entrenando en /GPU:0 y fallará si no hay GPU.
        # Una solución más robusta usaría el método build() de la capa.
        if gpus:
            with tf.device('/GPU:0'):
                self.relative_position_index = self.add_weight(
                    shape=rel_index.shape,
                    dtype=tf.int32,
                    trainable=False,
                    initializer=tf.constant_initializer(rel_index),
                    name='relative_position_index'
                )
        else:
             self.relative_position_index = self.add_weight(
                    shape=rel_index.shape,
                    dtype=tf.int32,
                    trainable=False,
                    initializer=tf.constant_initializer(rel_index),
                    name='relative_position_index'
                )
        # --- SOLUCIÓN GPU: FIN ---

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
        
        # Pre-calculamos la longitud de la secuencia (L = H*W) como un entero de Python.
        self.seq_len = input_resolution[0] * input_resolution[1]
        
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
            self.attn_mask = tf.constant(attn_mask, dtype=tf.float32) # Usar constante también aquí por si acaso
        else:
            self.attn_mask = None

    def call(self, x):
        H, W = self.input_resolution
        C = self.dim

        shortcut = x
        x = self.norm1(x)
        
        x = Reshape((H, W, C))(x)

        if any(s > 0 for s in self.shift_size):
            ssH, ssW = self.shift_size
            x = tf.roll(x, shift=[-ssH, -ssW], axis=[1, 2])

        wsH, wsW = self.window_size
        x_windows = Lambda(
            lambda t: tf.reshape(
                tf.transpose(
                    tf.reshape(t, (-1, H // wsH, wsH, W // wsW, wsW, C)),
                    (0, 1, 3, 2, 4, 5)
                ),
                (-1, wsH * wsW, C)
            )
        )(x)

        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        x = Lambda(
            lambda t: tf.reshape(
                tf.transpose(
                    tf.reshape(t, (-1, H // wsH, W // wsW, wsH, wsW, C)),
                    (0, 1, 3, 2, 4, 5)
                ),
                (-1, H, W, C)
            )
        )(attn_windows)

        if any(s > 0 for s in self.shift_size):
            ssH, ssW = self.shift_size
            x = tf.roll(x, shift=[ssH, ssW], axis=[1, 2])

        # Aplanamos de nuevo usando la longitud de secuencia estática.
        x = Reshape((self.seq_len, C))(x)
        
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
    try:
        files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg','.png'))]
    except FileNotFoundError:
        print(f"Directorio no encontrado: {img_dir}. Saltando la carga de datos.")
        return np.array([]), np.array([])
        
    aug = get_training_augmentation() if augment else get_validation_augmentation()
    
    mask_target_size = (256, 256)

    with ThreadPoolExecutor() as exe:
        futures = {exe.submit(lambda fn: (
            img_to_array(load_img(os.path.join(img_dir,fn), target_size=target_size)),
            img_to_array(load_img(os.path.join(mask_dir,fn.rsplit('.',1)[0] + '_mask.png'),
                                  color_mode='grayscale', target_size=mask_target_size))
        ), fn): fn for fn in files}
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Loading {'train' if augment else 'val'} data"):
            try:
                img_arr, mask_arr = future.result()
                if augment:
                    augm = aug(image=img_arr.astype('uint8'), mask=mask_arr)
                    img_arr, mask_arr = augm['image'], augm['mask']
                images.append(img_arr)
                masks.append(mask_arr)
            except Exception as e:
                print(f"Error procesando un archivo: {e}")

    if not images:
        return np.array([]), np.array([])
            
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
    # --- Encoder ---
    backbone = EfficientNetV2S(
        input_shape=shape,
        weights='imagenet', # Usar 'weights' en lugar de 'pretrained'
        include_top=False, # Usar 'include_top'
    )
    backbone.trainable = True # Asegurarse de que el backbone sea entrenable
    inp = backbone.input
    
    # Identificar la capa de salida del encoder correcta
    # 'post_swish' puede no existir, buscar una capa de salida apropiada.
    # Vamos a usar una capa con un downsampling conocido, p.ej. /16 o /32
    # Para EfficientNetV2S con input (256, 256), la salida después de block6a es (8, 8, 256)
    encoder_output_layer_name = 'add_35' # Un nombre típico de capa de skip connection profunda. Ajustar si es necesario.
    try:
        bottleneck = backbone.get_layer(encoder_output_layer_name).output
    except ValueError:
        print(f"Capa '{encoder_output_layer_name}' no encontrada. Usando la salida final del backbone.")
        bottleneck = backbone.output

    # --- Extracción dinámica de las skip connections ---
    # Los nombres de las capas pueden variar. Es mejor buscarlas por su factor de reducción de tamaño.
    # Input: 256 -> s1: 128, s2: 64, s3: 32, s4: 16
    layer_names = [l.name for l in backbone.layers]
    s1_name = next((name for name in reversed(layer_names) if 'add' in name and backbone.get_layer(name).output_shape[1] == 128), None)
    s2_name = next((name for name in reversed(layer_names) if 'add' in name and backbone.get_layer(name).output_shape[1] == 64), None)
    s3_name = next((name for name in reversed(layer_names) if 'add' in name and backbone.get_layer(name).output_shape[1] == 32), None)
    s4_name = next((name for name in reversed(layer_names) if 'add' in name and backbone.get_layer(name).output_shape[1] == 16), None)

    if not all([s1_name, s2_name, s3_name, s4_name]):
        raise ValueError("No se pudieron encontrar todas las capas de skip connection requeridas.")

    s1 = backbone.get_layer(s1_name).output
    s2 = backbone.get_layer(s2_name).output
    s3 = backbone.get_layer(s3_name).output
    s4 = backbone.get_layer(s4_name).output

    # --- Bloque Intermedio: Swin Transformer ---
    H, W, C = map(int, bottleneck.shape[1:])
    x = Reshape((H*W, C), name="swin_in_reshape")(bottleneck)
    x = SwinTransformerBlock(
        dim=C, input_resolution=(H, W), num_heads=4, window_size=2,
        shift_size=0, mlp_ratio=4., name="swin_block_1"
    )(x)
    x = Reshape((H, W, C), name="swin_out_reshape")(x)

    # --- Cuello de botella con WASP ---
    x = WASP(x, out_channels=256, dilation_rates=(2, 4, 6), use_global_pool=True,
             anti_gridding=True, use_attention=True, name="WASP")

    # --- Decoder (U-Net) ---
    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x); x = Concatenate()([x, s4]); x = conv_block(x, 128, 3); x = conv_block(x, 128, 3)
    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x); x = Concatenate()([x, s3]); x = conv_block(x, 64, 3); x = conv_block(x, 64, 3)
    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x); x = Concatenate()([x, s2]); x = conv_block(x, 48, 3); x = conv_block(x, 48, 3)
    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x); x = Concatenate()([x, s1]); x = conv_block(x, 32, 3); x = conv_block(x, 32, 3)
    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    
    out = Conv2D(num_classes_arg, 1, padding='same', activation='softmax', dtype='float32')(x)
    return Model(inp, out), backbone

# ---------------------------------------------------------------------------------
# El resto del script...
# ---------------------------------------------------------------------------------

# 5) Carga de datos
print("Cargando datos...")
# Crear directorios dummy si no existen para evitar errores en la ejecución inicial
os.makedirs('Balanced/train/images', exist_ok=True)
os.makedirs('Balanced/train/masks', exist_ok=True)
os.makedirs('Balanced/val/images', exist_ok=True)
os.makedirs('Balanced/val/masks', exist_ok=True)

train_X, train_y = load_augmented_dataset('Balanced/train/images', 'Balanced/train/masks', target_size=(256, 256), augment=True)
val_X,   val_y   = load_augmented_dataset('Balanced/val/images',   'Balanced/val/masks',   target_size=(256, 256), augment=False)

if train_X.size == 0 or val_X.size == 0:
    print("\nNo se cargaron datos. Creando datos dummy para la construcción del modelo.")
    train_X = np.random.rand(4, 256, 256, 3).astype(np.float32)
    train_y = np.random.randint(0, 2, (4, 256, 256)).astype(np.int32)
    val_X = np.random.rand(2, 256, 256, 3).astype(np.float32)
    val_y = np.random.randint(0, 2, (2, 256, 256)).astype(np.int32)
    num_classes = 2
else:
    num_classes = int(np.max(train_y) + 1)

print(f"\nNúmero de clases: {num_classes}")

# 7) Pérdidas y Métricas
def tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3, smooth=1e-7):
    y_true_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), depth=num_classes)
    y_pred_probs = y_pred # La salida del modelo ya es softmax
    tp = tf.reduce_sum(y_true_one_hot * y_pred_probs, axis=list(range(1, 4)))
    fn = tf.reduce_sum(y_true_one_hot * (1 - y_pred_probs), axis=list(range(1, 4)))
    fp = tf.reduce_sum((1 - y_true_one_hot) * y_pred_probs, axis=list(range(1, 4)))
    tversky_index = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return 1.0 - tf.reduce_mean(tversky_index)

def ultimate_iou_loss(y_true, y_pred):
    y_true_int = tf.cast(y_true, tf.int32)
    lov = 0.6 * lovasz_softmax(y_pred, y_true_int)
    tver = 0.4 * tversky_loss(y_true, y_pred)
    return lov + tver

class MeanIoU(tf.keras.metrics.Metric):
    def __init__(self, num_classes, name='mean_iou', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.total_cm = self.add_weight(shape=(num_classes, num_classes), name='total_confusion_matrix', initializer='zeros')
    def update_state(self, y_true, y_pred, sample_weight=None):
        preds = tf.argmax(y_pred, axis=-1)
        y_true_squeezed = tf.squeeze(y_true, axis=-1) if len(y_true.shape) == 4 else y_true
        cm = tf.math.confusion_matrix(tf.reshape(y_true_squeezed, [-1]), tf.reshape(preds, [-1]), num_classes=self.num_classes)
        self.total_cm.assign_add(tf.cast(cm, self.total_cm.dtype))
    def result(self):
        tp = tf.linalg.tensor_diag_part(self.total_cm)
        denom = tf.reduce_sum(self.total_cm, axis=1) + tf.reduce_sum(self.total_cm, axis=0) - tp
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
        self.iou_metric = MeanIoU(num_classes=num_classes, name='mean_iou')
    def call(self, inputs, training=False):
        return self.seg_model(inputs, training=training)
    @property
    def metrics(self):
        return [self.loss_tracker, self.iou_metric]
    def train_step(self, data):
        x, y_true = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y_true, y_pred, regularization_losses=self.losses)
        trainable_vars = self.trainable_variables
        grads = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(grads, trainable_vars))
        self.loss_tracker.update_state(loss)
        self.iou_metric.update_state(y_true, y_pred)
        return {m.name: m.result() for m in self.metrics}
    def test_step(self, data):
        x, y_true = data
        y_pred = self(x, training=False)
        loss = self.compiled_loss(y_true, y_pred, regularization_losses=self.losses)
        self.loss_tracker.update_state(loss)
        self.iou_metric.update_state(y_true, y_pred)
        return {m.name: m.result() for m in self.metrics}

# 9) Entrenamiento
print(f"\n--- Iniciando construcción del modelo en: {device} ---")
checkpoint_dir = './checkpoints_swin'
os.makedirs(checkpoint_dir, exist_ok=True)
ckpt_path = os.path.join(checkpoint_dir, 'final_swin_model.keras')
val_gen = tf.data.Dataset.from_tensor_slices((val_X, val_y)).batch(4)
train_gen = tf.data.Dataset.from_tensor_slices((train_X, train_y)).shuffle(buffer_size=len(train_X)).batch(4)

with tf.device(device):
    model = IoUOptimizedModel(shape=train_X.shape[1:], num_classes=num_classes)
    model.build(input_shape=(None, *train_X.shape[1:]))
    model.seg_model.summary()

    model.compile(optimizer=tf.keras.optimizers.AdamW(1e-4), loss=ultimate_iou_loss)
    
    cbs = [
        ModelCheckpoint(ckpt_path, save_best_only=True, monitor='val_mean_iou', mode='max', verbose=1),
        EarlyStopping(monitor='val_mean_iou', mode='max', patience=15, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1),
        TensorBoard(log_dir='./logs/swin_final', update_freq='epoch')
    ]
    
    print("\n--- Iniciando Entrenamiento ---")
    # Ejecutamos solo unas pocas épocas para demostración
    model.fit(train_gen, epochs=3, validation_data=val_gen, callbacks=cbs)
    
    # Cargar el mejor modelo guardado
    try:
        model.load_weights(ckpt_path)
    except Exception as e:
        print(f"No se pudieron cargar los pesos del checkpoint: {e}")

# 10) Evaluación y Visualización
print("\n--- Evaluación del modelo final ---")
def evaluate_model(model_to_eval, X, y):
    preds = model_to_eval.predict(X, batch_size=4)
    mask = np.argmax(preds, axis=-1)
    y_squeezed = np.squeeze(y) if y.ndim == 4 else y
    ious = []
    for cls in range(num_classes):
        yt, yp = (y_squeezed==cls), (mask==cls)
        inter = np.sum(yt & yp)
        union = np.sum(yt | yp)
        ious.append(inter / union if union > 0 else 1.0) # IoU es 1 si no hay GT ni pred
    return {'mean_iou': np.mean(ious), 'class_ious': ious, 'pixel_accuracy': np.mean(mask == y_squeezed)}

metrics = evaluate_model(model, val_X, val_y)
print(f"Mean IoU: {metrics['mean_iou']:.4f}")
print(f"Pixel Accuracy: {metrics['pixel_accuracy']:.4f}")
for i, iou in enumerate(metrics['class_ious']): print(f" Class {i}: {iou:.4f}")

def visualize_predictions(model_to_eval, X, y, num_samples=5):
    if len(X) < num_samples: num_samples = len(X)
    idxs = np.random.choice(len(X), num_samples, replace=False)
    preds = model_to_eval.predict(X[idxs])
    masks = np.argmax(preds, axis=-1)
    y_squeezed = np.squeeze(y) if y.ndim == 4 else y
    
    cmap = plt.cm.get_cmap('viridis', num_classes)
    plt.figure(figsize=(15, 5 * num_samples))
    for i, ix in enumerate(idxs):
        plt.subplot(num_samples, 3, 3 * i + 1); plt.imshow(X[ix]); plt.title('Image'); plt.axis('off')
        plt.subplot(num_samples, 3, 3 * i + 2); plt.imshow(y_squeezed[ix], cmap=cmap, vmin=0, vmax=num_classes - 1); plt.title('Ground Truth'); plt.axis('off')
        plt.subplot(num_samples, 3, 3 * i + 3); plt.imshow(masks[i], cmap=cmap, vmin=0, vmax=num_classes - 1); plt.title('Prediction'); plt.axis('off')
    plt.tight_layout()
    plt.savefig('predictions_visualization.png')
    plt.close()
    print("\nVisualización de predicciones guardada en 'predictions_visualization.png'")

visualize_predictions(model, val_X, val_y)
print("\nScript completado.")