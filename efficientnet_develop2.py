# -*- coding: utf-8 -*-
"""
efficientnetv2_swin_deeplab_iou_fixed.py

Modelo DeepLab-v3+ con EfficientNetV2-S + Swin-Transformer + WASP.
Correcciones:
  • relative_position_index se define como tf.constant (no variable) → sin conflicto CPU↔GPU
  • Decoder reorganizado: Concat antes de upsampling (sin errores de forma)
"""

# --------------------------------------------------------------------------- #
# 1) IMPORTS Y CONFIGURACIÓN BÁSICA
# --------------------------------------------------------------------------- #
import os
os.environ['MPLBACKEND'] = 'agg'      # evita abrir ventana de matplotlib

import tensorflow as tf
print("TensorFlow:", tf.__version__)

# Selección de dispositivo
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    device = '/GPU:0'
    print("Usando GPU:", gpus[0].name)
else:
    device = '/CPU:0'
    print("Entrenando en CPU")

# Resto de imports
import numpy as np
import cv2, albumentations as A, matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from tensorflow.keras import layers
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                         ReduceLROnPlateau, TensorBoard)
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from lovasz_losses_tf import lovasz_softmax
EfficientNetV2S = tf.keras.applications.EfficientNetV2S


# --------------------------------------------------------------------------- #
# 2) SWIN-TRANSFORMER LAYERS
# --------------------------------------------------------------------------- #
class Mlp(tf.keras.layers.Layer):
    def __init__(self, hidden_features, out_features, drop=0., **kw):
        super().__init__(**kw)
        self.fc1  = Dense(hidden_features, activation='gelu')
        self.drop = Dropout(drop)
        self.fc2  = Dense(out_features)

    def call(self, x):
        return self.fc2(self.drop(self.fc1(x)))


class WindowAttention(tf.keras.layers.Layer):
    """W-MSA con sesgo de posición relativa."""
    def __init__(self, dim, window_size, num_heads,
                 qkv_bias=True, qk_scale=None,
                 attn_drop=0., proj_drop=0., **kw):
        super().__init__(**kw)
        self.dim          = dim
        self.window_size  = window_size         # (h, w)
        self.num_heads    = num_heads
        head_dim          = dim // num_heads
        self.scale        = qk_scale or head_dim ** -0.5

        self.qkv  = Dense(dim*3, use_bias=qkv_bias, name="qkv")
        self.proj = Dense(dim, name="proj")
        self.attn_drop = Dropout(attn_drop)
        self.proj_drop = Dropout(proj_drop)

        # tabla de sesgo relativa entrenable
        num_rel = (2*window_size[0]-1)*(2*window_size[1]-1)
        self.relative_position_bias_table = self.add_weight(
            shape=(num_rel, num_heads),
            initializer=tf.keras.initializers.TruncatedNormal(.02),
            trainable=True,
            name="relative_position_bias_table"
        )

        # índice (constante) de posiciones relativas  ----------------------- #
        coords_h = tf.range(window_size[0])
        coords_w = tf.range(window_size[1])
        coords   = tf.stack(tf.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flat = tf.reshape(coords, [2, -1])
        rel_coords  = coords_flat[:, :, None] - coords_flat[:, None, :]  # 2, N, N
        rel_coords  = tf.transpose(rel_coords, (1,2,0))
        rel_coords  = rel_coords + [window_size[0]-1, window_size[1]-1]
        rel_coords  = rel_coords * [2*window_size[1]-1, 1]
        rel_index   = tf.reduce_sum(rel_coords, axis=-1)

        # ---> **CORRECCIÓN PRINCIPAL**: tf.constant (no Variable)
        self.relative_position_index = tf.constant(
            rel_index, dtype=tf.int32, name="relative_position_index"
        )

    def call(self, x, mask=None):
        B_, N, C = tf.unstack(tf.shape(x))
        qkv = self.qkv(x)
        qkv = tf.reshape(qkv, (B_, N, 3, self.num_heads, C // self.num_heads))
        qkv = tf.transpose(qkv, (2,0,3,1,4))      # 3, B, heads, N, d
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = tf.matmul(q, k, transpose_b=True)  # B, heads, N, N

        rel_bias = tf.gather(self.relative_position_bias_table,
                              tf.reshape(self.relative_position_index, [-1]))
        rel_bias = tf.reshape(rel_bias,
                                (self.window_size[0]*self.window_size[1],
                                 self.window_size[0]*self.window_size[1],
                                 -1))              # N, N, heads
        rel_bias = tf.transpose(rel_bias, (2,0,1)) # heads, N, N
        attn     = attn + rel_bias[None, ...]      # broadcast a B

        if mask is not None:
            nW = tf.shape(mask)[0]
            attn = tf.reshape(attn, (-1, nW, self.num_heads, N, N))
            attn = attn + tf.cast(mask[None, ..., None, :, :], attn.dtype)
            attn = tf.reshape(attn, (-1, self.num_heads, N, N))

        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = tf.matmul(attn, v)                    # B, heads, N, d
        x = tf.transpose(x, (0,2,1,3))
        x = tf.reshape(x, (B_, N, C))
        x = self.proj_drop(self.proj(x))
        return x


class SwinTransformerBlock(tf.keras.layers.Layer):
    def __init__(self, dim, input_resolution, num_heads,
                 window_size=7, shift_size=0, mlp_ratio=4.,
                 drop=0., attn_drop=0., drop_path=0.,
                 qkv_bias=True, qk_scale=None,
                 norm_layer=LayerNormalization, **kw):
        super().__init__(**kw)
        self.dim   = dim
        self.H, self.W = input_resolution
        self.window_size = (window_size, window_size)
        self.shift_size  = (shift_size, shift_size)

        self.norm1 = norm_layer()
        self.attn  = WindowAttention(dim, self.window_size, num_heads,
                                     qkv_bias, qk_scale, attn_drop, drop)
        self.drop_path = Dropout(drop_path) if drop_path > 0. else tf.identity
        self.norm2 = norm_layer()
        self.mlp   = Mlp(int(dim*mlp_ratio), dim, drop)

        # máscara para SW-MSA (si shift_size > 0)
        if any(s > 0 for s in self.shift_size):
            img_mask = np.zeros((1, self.H, self.W, 1), np.float32)
            wsH, wsW = self.window_size
            ssH, ssW = self.shift_size
            h_slices = (slice(0, -wsH), slice(-wsH, -ssH), slice(-ssH, None))
            w_slices = (slice(0, -wsW), slice(-wsW, -ssW), slice(-ssW, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = tf.reshape(
                tf.transpose(
                    tf.reshape(img_mask, (-1, self.H//wsH, wsH,
                                              self.W//wsW, wsW, 1)),
                    (0,1,3,2,4,5)),
                (-1, wsH*wsW)
            )
            attn_mask = tf.expand_dims(mask_windows, 1) - tf.expand_dims(mask_windows, 2)
            attn_mask = tf.where(attn_mask != 0, -100., 0.)
            self.attn_mask = tf.constant(attn_mask, tf.float32)
        else:
            self.attn_mask = None

    def call(self, x):
        B = tf.shape(x)[0]        # batch dinámico
        H, W = self.H, self.W
        C = x.shape[-1]           # 160 (conocido)

        x = self.norm1(x)
        x = tf.reshape(x, (B, H, W, C))

        # cyclic shift
        if any(s > 0 for s in self.shift_size):
            x = tf.roll(x, shift=[-self.shift_size[0], -self.shift_size[1]], axis=[1,2])

        # windows
        wsH, wsW = self.window_size
        x_windows = tf.reshape(
            tf.transpose(
                tf.reshape(x, (B, H//wsH, wsH, W//wsW, wsW, C)),
                (0,1,3,2,4,5)
            ),
            (-1, wsH*wsW, C)
        )

        attn_windows = self.attn(x_windows, self.attn_mask)

        # reverse windows
        x = tf.reshape(
            tf.transpose(
                tf.reshape(attn_windows, (B, H//wsH, W//wsW, wsH, wsW, C)),
                (0,1,3,2,4,5)),
            (B, H, W, C)
        )

        if any(s > 0 for s in self.shift_size):
            x = tf.roll(x, shift=[self.shift_size[0], self.shift_size[1]], axis=[1,2])

        x = tf.reshape(x, (B, L, C))
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# --------------------------------------------------------------------------- #
# 3) BLOQUES UTILITARIOS (WASP, conv_block, etc.)
# --------------------------------------------------------------------------- #
def conv_block(x, filters, k, s=1, d=1):
    x = Conv2D(filters, k, s, 'same', dilation_rate=d, use_bias=False)(x)
    x = BatchNormalization()(x)
    return Activation('swish')(x)


def WASP(x, out=256, dil=(1,2,4,8), gp=True, ag=True, att=False, name="WASP"):
    convs = []
    y = conv_block(x, out, 1); convs.append(y); prev = y
    for r in dil:
        b = conv_block(prev, out, 3, d=r)
        if ag and r > 1:
            b = conv_block(b, out, 3)
        convs.append(b); prev = b
    if gp:
        g = GlobalAveragePooling2D(keepdims=True)(x)
        g = conv_block(g, out, 1)
        g = Lambda(lambda t: tf.image.resize(t[0], tf.shape(t[1])[1:3],
                                             method='bilinear'))([g, x])
        convs.append(g)
    y = Concatenate()(convs)
    y = conv_block(y, out, 1)
    if att:
        se = GlobalAveragePooling2D()(y)
        se = Dense(out//16, activation='relu', use_bias=False)(se)
        se = Dense(out, activation='sigmoid', use_bias=False)(se)
        y = Multiply()([y, se])
    return y


# --------------------------------------------------------------------------- #
# 4) CONSTRUCCIÓN DEL MODELO
# --------------------------------------------------------------------------- #
def build_model(img_shape=(256,256,3), n_classes=2):
    backbone = EfficientNetV2S(input_shape=img_shape,
                               include_top=False, weights='imagenet')
    inp = backbone.input
    layer_names = [l.name for l in backbone.layers]

    # capas de skip auto-detectadas por resolución
    get = lambda sz: next(name for name in reversed(layer_names)
                          if 'add' in name and backbone.get_layer(name).output.shape[1]==sz)

    s1 = backbone.get_layer(get(128)).output   # 1/2  (128×128)
    s2 = backbone.get_layer(get(64 )).output   # 1/4
    s3 = backbone.get_layer(get(32 )).output   # 1/8
    s4 = backbone.get_layer(get(16 )).output   # 1/16

    # Swin + WASP en la resolución 1/16
    H,W,C = map(int, s4.shape[1:])
    x = Reshape((H*W, C))(s4)
    x = SwinTransformerBlock(C, (H,W), num_heads=4, window_size=2)(x)
    x = Reshape((H,W,C))(x)
    x = WASP(x, 256, (2,4,6), gp=True, att=True)

    # ---------- DECODER U-Net consistente ----------
    # 16×16 → 32×32
    x = Concatenate()([x, s4])
    x = conv_block(x, 128, 3); x = conv_block(x, 128, 3)
    x = UpSampling2D()(x)                  # 32×32

    # 32×32 → 64×64
    x = Concatenate()([x, s3])
    x = conv_block(x, 64, 3);  x = conv_block(x, 64, 3)
    x = UpSampling2D()(x)                  # 64×64

    # 64×64 → 128×128
    x = Concatenate()([x, s2])
    x = conv_block(x, 48, 3); x = conv_block(x, 48, 3)
    x = UpSampling2D()(x)                  # 128×128

    # 128×128 → 256×256
    x = Concatenate()([x, s1])
    x = conv_block(x, 32, 3); x = conv_block(x, 32, 3)
    x = UpSampling2D()(x)                  # 256×256

    out = Conv2D(n_classes, 1, padding='same', activation='softmax')(x)
    return Model(inp, out)


# --------------------------------------------------------------------------- #
# 5) FUNCIONES DE DATOS Y PÉRDIDAS
# --------------------------------------------------------------------------- #
def get_training_augmentation():
    return A.Compose([
        A.RandomScale(scale_limit=0.2, p=0.5),
        A.PadIfNeeded(min_height=256, min_width=256, border_mode=cv2.BORDER_REFLECT),
        A.RandomCrop(height=256, width=256),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.GaussNoise(p=0.1),
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.2),
    ])

def get_validation_augmentation():
    return A.Compose([])

def load_augmented_dataset(img_dir, mask_dir, target_size=(256,256), augment=False):
    images, masks = [], []
    try:
        files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg','.png'))]
    except FileNotFoundError:
        print(f"Directorio no encontrado: {img_dir}. Saltando.")
        return np.array([]), np.array([])

    aug = get_training_augmentation() if augment else get_validation_augmentation()

    with ThreadPoolExecutor() as exe:
        futures = {exe.submit(lambda fn: (
            img_to_array(load_img(os.path.join(img_dir,fn), target_size=target_size)),
            img_to_array(load_img(os.path.join(mask_dir,fn.rsplit('.',1)[0] + '_mask.png'),
                                  color_mode='grayscale', target_size=target_size))
        ), fn): fn for fn in files}
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Cargando {'train' if augment else 'val'}"):
            try:
                img_arr, mask_arr = future.result()
                if augment:
                    augm = aug(image=img_arr.astype('uint8'), mask=mask_arr)
                    img_arr, mask_arr = augm['image'], augm['mask']
                images.append(img_arr)
                masks.append(mask_arr)
            except Exception as e:
                print(f"Error procesando {futures[future]}: {e}")

    if not images:
        return np.array([]), np.array([])

    X = np.array(images, dtype='float32') / 255.0
    y = np.array(masks, dtype='int32')
    return X, y

def tversky_loss(y_true, y_pred, n_classes, alpha=0.7, beta=0.3, smooth=1e-7):
    y_true_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), depth=n_classes)
    y_pred_probs = y_pred
    tp = tf.reduce_sum(y_true_one_hot * y_pred_probs, axis=list(range(1, 4)))
    fn = tf.reduce_sum(y_true_one_hot * (1 - y_pred_probs), axis=list(range(1, 4)))
    fp = tf.reduce_sum((1 - y_true_one_hot) * y_pred_probs, axis=list(range(1, 4)))
    tversky_index = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return 1.0 - tf.reduce_mean(tversky_index)

def ultimate_iou_loss(y_true, y_pred, n_classes):
    y_true_int = tf.cast(y_true, tf.int32)
    loss_lov = 0.6 * lovasz_softmax(y_pred, y_true_int)
    loss_tver = 0.4 * tversky_loss(y_true, y_pred, n_classes)
    return loss_lov + loss_tver

# --------------------------------------------------------------------------- #
# 6) ENTRENAMIENTO DE DEMO
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # datos dummy de ejemplo (el loader real se puede usar igual)
    X = np.random.rand(4, 256, 256, 3).astype('float32')
    y = np.random.randint(0, 2, (4, 256, 256)).astype('int32')
    n_classes = 2

    # Para usar la pérdida personalizada:
    # loss_fn = lambda yt, yp: ultimate_iou_loss(yt, yp, n_classes)
    
    with tf.device(device):
        model = build_model((256,256,3), n_classes)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.summary()
        print("\nIniciando entrenamiento de prueba...")
        model.fit(X, y, epochs=10, batch_size=2)
        print("Entrenamiento de prueba completado.")