# -*- coding: utf-8 -*-
"""
efficientnetv2_swin_deeplab_phased_training.py

SÍNTESIS DEL MODELO:
- Arquitectura: DeepLab-v3+ con un backbone EfficientNetV2-S, enriquecido con
  un bloque Swin-Transformer y un módulo WASP (Wide Atrous Spatial Pyramid).
- Metodología de Entrenamiento: Bucle de entrenamiento personalizado que optimiza
  directamente una métrica de IoU combinada (Lovasz-Softmax + Tversky).
- Estrategia: Entrenamiento por fases para una convergencia robusta.
    1. Fase 1: Entrenamiento del decodificador (backbone congelado).
    2. Fase 2: Ajuste fino (fine-tuning) de todo el modelo con una tasa de
       aprendizaje reducida.

CORRECCIONES DEL SCRIPT ORIGINAL:
- relative_position_index en Swin-Transformer se define como tf.constant
  para evitar conflictos de dispositivo (CPU/GPU).
- El decodificador ha sido reorganizado para asegurar la consistencia de
  las formas de los tensores.
- Se implementa un bucle de entrenamiento personalizado para usar pérdidas
  no estándar de forma nativa.
"""

# --------------------------------------------------------------------------- #
# 1) IMPORTS Y CONFIGURACIÓN DEL ENTORNO
# --------------------------------------------------------------------------- #
import os
os.environ['MPLBACKEND'] = 'agg'  # Evita que matplotlib abra ventanas GUI

import tensorflow as tf
print("Versión de TensorFlow:", tf.__version__)

# Selección y configuración del dispositivo (GPU/CPU)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        device = '/GPU:0'
        print("Dispositivo activo:", tf.test.gpu_device_name())
    except RuntimeError as e:
        print("Error en la configuración de GPU:", e)
        device = '/CPU:0'
else:
    device = '/CPU:0'
    print("No se encontró GPU. Entrenando en CPU.")

# --- Resto de imports ---
import numpy as np
import cv2
import albumentations as A
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from tensorflow.keras import layers
from tensorflow.keras.layers import (Input, Conv2D, BatchNormalization, Activation,
                                     Concatenate, UpSampling2D, GlobalAveragePooling2D,
                                     Reshape, Dense, Multiply, Dropout, LayerNormalization)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# --- Componentes externos y del modelo ---
from lovasz_losses_tf import lovasz_softmax
EfficientNetV2S = tf.keras.applications.EfficientNetV2S

# --------------------------------------------------------------------------- #
# 2) CAPAS DEL SWIN-TRANSFORMER
# --------------------------------------------------------------------------- #
class Mlp(layers.Layer):
    """Capa de Perceptrón Multicapa (MLP) para el Transformer."""
    def __init__(self, hidden_features, out_features, drop=0., **kwargs):
        super().__init__(**kwargs)
        self.fc1 = Dense(hidden_features, activation='gelu')
        self.drop = Dropout(drop)
        self.fc2 = Dense(out_features)

    def call(self, x):
        return self.fc2(self.drop(self.fc1(x)))


class WindowAttention(layers.Layer):
    """Atención Multi-Cabeza en Ventana (W-MSA) con sesgo de posición relativa."""
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None,
                 attn_drop=0., proj_drop=0., **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.window_size = window_size  # (altura, ancho)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = Dense(dim * 3, use_bias=qkv_bias, name="qkv")
        self.attn_drop = Dropout(attn_drop)
        self.proj = Dense(dim, name="proj")
        self.proj_drop = Dropout(proj_drop)

        # Tabla de sesgo de posición relativa (entrenable)
        num_rel_positions = (2 * window_size[0] - 1) * (2 * window_size[1] - 1)
        self.relative_position_bias_table = self.add_weight(
            shape=(num_rel_positions, num_heads),
            initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=True,
            name="relative_position_bias_table"
        )

        # --- CORRECCIÓN CLAVE: Índice de posición relativa como constante ---
        coords_h = tf.range(window_size[0])
        coords_w = tf.range(window_size[1])
        coords = tf.stack(tf.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flat = tf.reshape(coords, [2, -1])
        relative_coords = coords_flat[:, :, None] - coords_flat[:, None, :]
        relative_coords = tf.transpose(relative_coords, perm=[1, 2, 0])
        relative_coords = relative_coords + [window_size[0] - 1, window_size[1] - 1]
        relative_coords = relative_coords * [2 * window_size[1] - 1, 1]
        relative_position_index = tf.reduce_sum(relative_coords, axis=-1)

        self.relative_position_index = tf.constant(
            relative_position_index, dtype=tf.int32, name="relative_position_index"
        )

    def call(self, x, mask=None):
        B_, N, C = tf.unstack(tf.shape(x))
        qkv = self.qkv(x)
        qkv = tf.reshape(qkv, (B_, N, 3, self.num_heads, C // self.num_heads))
        qkv = tf.transpose(qkv, perm=[2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = tf.matmul(q, k, transpose_b=True)

        relative_bias = tf.gather(self.relative_position_bias_table,
                                  tf.reshape(self.relative_position_index, [-1]))
        relative_bias = tf.reshape(relative_bias, (N, N, -1))
        relative_bias = tf.transpose(relative_bias, perm=[2, 0, 1])
        attn = attn + relative_bias[None, ...]

        if mask is not None:
            nW = tf.shape(mask)[0]
            attn = tf.reshape(attn, (-1, nW, self.num_heads, N, N))
            attn = attn + tf.cast(mask[None, ..., None, :, :], attn.dtype)
            attn = tf.reshape(attn, (-1, self.num_heads, N, N))

        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = tf.matmul(attn, v)
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        x = tf.reshape(x, (B_, N, C))
        x = self.proj_drop(self.proj(x))
        return x


class SwinTransformerBlock(layers.Layer):
    """Bloque de Swin Transformer con W-MSA y SW-MSA."""
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., drop=0., attn_drop=0., drop_path=0., qkv_bias=True,
                 qk_scale=None, norm_layer=LayerNormalization, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.H, self.W = input_resolution
        self.window_size = (window_size, window_size)
        self.shift_size = (shift_size, shift_size)

        self.norm1 = norm_layer(epsilon=1e-5)
        self.attn = WindowAttention(dim, self.window_size, num_heads, qkv_bias, qk_scale, attn_drop, drop)
        self.drop_path = Dropout(drop_path) if drop_path > 0. else tf.identity
        self.norm2 = norm_layer(epsilon=1e-5)
        self.mlp = Mlp(int(dim * mlp_ratio), dim, drop)

        if any(s > 0 for s in self.shift_size):
            img_mask = np.zeros((1, self.H, self.W, 1), np.float32)
            h_slices = (slice(0, -self.window_size[0]), slice(-self.window_size[0], -self.shift_size[0]), slice(-self.shift_size[0], None))
            w_slices = (slice(0, -self.window_size[1]), slice(-self.window_size[1], -self.shift_size[1]), slice(-self.shift_size[1], None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            # De NumPy a Tensor constante
            img_mask_tensor = tf.constant(img_mask, dtype=tf.float32)
            mask_windows = tf.reshape(
                tf.transpose(
                    tf.reshape(img_mask_tensor, (-1, self.H // self.window_size[0], self.window_size[0],
                                                 self.W // self.window_size[1], self.window_size[1], 1)),
                    perm=[0, 1, 3, 2, 4, 5]
                ),
                (-1, self.window_size[0] * self.window_size[1])
            )
            attn_mask = tf.expand_dims(mask_windows, 1) - tf.expand_dims(mask_windows, 2)
            attn_mask = tf.where(attn_mask != 0, -100.0, 0.0)
            self.attn_mask = tf.cast(attn_mask, tf.float32)
        else:
            self.attn_mask = None

    def call(self, x):
        B, L, C = x.shape
        H, W = self.H, self.W
        shortcut = x
        
        x = self.norm1(x)
        x = tf.reshape(x, (B, H, W, C))

        if any(s > 0 for s in self.shift_size):
            x = tf.roll(x, shift=[-self.shift_size[0], -self.shift_size[1]], axis=[1, 2])

        wsH, wsW = self.window_size
        x_windows = tf.reshape(
            tf.transpose(
                tf.reshape(x, (B, H // wsH, wsH, W // wsW, wsW, C)),
                perm=[0, 1, 3, 2, 4, 5]
            ),
            (-1, wsH * wsW, C)
        )

        attn_windows = self.attn(x_windows, self.attn_mask)

        x = tf.reshape(
            tf.transpose(
                tf.reshape(attn_windows, (B, H // wsH, W // wsW, wsH, wsW, C)),
                perm=[0, 1, 3, 2, 4, 5]
            ),
            (B, H, W, C)
        )

        if any(s > 0 for s in self.shift_size):
            x = tf.roll(x, shift=[self.shift_size[0], self.shift_size[1]], axis=[1, 2])

        x = tf.reshape(x, (B, L, C))
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

# --------------------------------------------------------------------------- #
# 3) BLOQUES DE CONSTRUCCIÓN (WASP, CONV_BLOCK)
# --------------------------------------------------------------------------- #
def conv_block(x, filters, k, s=1, d=1):
    """Bloque convolucional: Conv2D -> BatchNorm -> Swish."""
    x = Conv2D(filters, k, strides=s, padding='same', dilation_rate=d, use_bias=False)(x)
    x = BatchNormalization()(x)
    return Activation('swish')(x)


def WASP(x, out_filters=256, dilations=(1, 2, 4, 8), use_gap=True, use_attention=False, name="WASP"):
    """Wide Atrous Spatial Pyramid (WASP) module."""
    with tf.name_scope(name):
        convs = []
        # Rama 1x1
        y = conv_block(x, out_filters, 1)
        convs.append(y)
        
        # Ramas con dilatación
        for r in dilations:
            b = conv_block(x, out_filters, 3, d=r)
            convs.append(b)
            
        # Rama de Global Average Pooling
        if use_gap:
            g = GlobalAveragePooling2D(keepdims=True)(x)
            g = conv_block(g, out_filters, 1)
            g = tf.image.resize(g, tf.shape(x)[1:3], method='bilinear')
            convs.append(g)
            
        # Fusión y proyección final
        y = Concatenate()(convs)
        y = conv_block(y, out_filters, 1)
        
        # Atención de canal (Squeeze-and-Excitation) opcional
        if use_attention:
            se = GlobalAveragePooling2D()(y)
            se = Dense(out_filters // 16, activation='relu', use_bias=False)(se)
            se = Dense(out_filters, activation='sigmoid', use_bias=False)(se)
            se = Reshape((1, 1, out_filters))(se)
            y = Multiply()([y, se])
            
        return y

# --------------------------------------------------------------------------- #
# 4) CONSTRUCCIÓN DEL MODELO COMPLETO
# --------------------------------------------------------------------------- #
def build_model(img_shape=(256, 256, 3), n_classes=2):
    """Construye la arquitectura de segmentación completa."""
    backbone = EfficientNetV2S(input_shape=img_shape, include_top=False, weights='imagenet')
    inp = backbone.input
    
    # Detección automática de capas de skip-connection por resolución
    layer_names = [l.name for l in backbone.layers]
    def get_skip_layer(size):
        return next(name for name in reversed(layer_names)
                    if 'add' in name and backbone.get_layer(name).output.shape[1] == size)

    s1 = backbone.get_layer(get_skip_layer(128)).output  # Res: 1/2 (128x128)
    s2 = backbone.get_layer(get_skip_layer(64)).output   # Res: 1/4 (64x64)
    s3 = backbone.get_layer(get_skip_layer(32)).output   # Res: 1/8 (32x32)
    s4 = backbone.get_layer(get_skip_layer(16)).output   # Res: 1/16 (16x16)

    # --- Cuello de Botella con Swin + WASP a resolución 1/16 ---
    H, W, C = s4.shape[1:]
    x = Reshape((H * W, C))(s4)
    x = SwinTransformerBlock(C, (H, W), num_heads=4, window_size=2, name="SwinBlock")(x)
    x = Reshape((H, W, C))(x)
    x = WASP(x, 256, dilations=(2, 4, 6), use_gap=True, use_attention=True, name="WASP_Module")

    # --- DECODIFICADOR (estilo U-Net) ---
    # De 16x16 -> 32x32
    x = Concatenate()([UpSampling2D()(x), s4])
    x = conv_block(x, 128, 3); x = conv_block(x, 128, 3)

    # De 32x32 -> 64x64
    x = Concatenate()([UpSampling2D()(x), s3])
    x = conv_block(x, 64, 3);  x = conv_block(x, 64, 3)

    # De 64x64 -> 128x128
    x = Concatenate()([UpSampling2D()(x), s2])
    x = conv_block(x, 48, 3); x = conv_block(x, 48, 3)

    # De 128x128 -> 256x256
    x = Concatenate()([UpSampling2D()(x), s1])
    x = conv_block(x, 32, 3); x = conv_block(x, 32, 3)
    
    # Capa de salida final
    out = Conv2D(n_classes, 1, padding='same', activation='softmax', dtype='float32')(x)
    
    return Model(inputs=inp, outputs=out), backbone


# --------------------------------------------------------------------------- #
# 5) PÉRDIDAS Y MÉTRICAS PERSONALIZADAS
# --------------------------------------------------------------------------- #
def tversky_loss(y_true, y_pred, n_classes, alpha=0.7, beta=0.3, smooth=1e-7):
    """Calcula la pérdida Tversky, útil para datasets desbalanceados."""
    y_true_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), depth=n_classes)
    y_pred_probs = y_pred # La salida del modelo ya es softmax

    tp = tf.reduce_sum(y_true_one_hot * y_pred_probs, axis=list(range(1, 4)))
    fn = tf.reduce_sum(y_true_one_hot * (1 - y_pred_probs), axis=list(range(1, 4)))
    fp = tf.reduce_sum((1 - y_true_one_hot) * y_pred_probs, axis=list(range(1, 4)))
    
    tversky_index = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return 1.0 - tf.reduce_mean(tversky_index)


def ultimate_iou_loss(y_true, y_pred, n_classes):
    """Pérdida combinada para optimizar IoU: 60% Lovasz + 40% Tversky."""
    y_true_int = tf.cast(y_true, tf.int32)
    # y_pred ya está en formato de logits o softmax, lovasz_softmax lo maneja
    loss_lov = 0.6 * lovasz_softmax(y_pred, tf.squeeze(y_true_int, axis=-1))
    loss_tver = 0.4 * tversky_loss(y_true, y_pred, n_classes)
    return loss_lov + loss_tver

class MeanIoU(tf.keras.metrics.Metric):
    """Métrica de Mean Intersection over Union (mIoU) robusta."""
    def __init__(self, num_classes, name='mean_iou', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.total_cm = self.add_weight(
            name='total_confusion_matrix',
            shape=(num_classes, num_classes),
            initializer='zeros'
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_labels = tf.argmax(y_pred, axis=-1)
        
        # Asegurarse que y_true no tenga una dimensión de canal extra
        if len(y_true.shape) == 4 and y_true.shape[-1] == 1:
            y_true = tf.squeeze(y_true, axis=-1)
            
        cm = tf.math.confusion_matrix(
            tf.reshape(tf.cast(y_true, self.total_cm.dtype), [-1]),
            tf.reshape(tf.cast(y_pred_labels, self.total_cm.dtype), [-1]),
            num_classes=self.num_classes
        )
        self.total_cm.assign_add(cm)

    def result(self):
        tp = tf.linalg.tensor_diag_part(self.total_cm)
        sum_over_row = tf.reduce_sum(self.total_cm, axis=0)
        sum_over_col = tf.reduce_sum(self.total_cm, axis=1)
        union = sum_over_row + sum_over_col - tp
        iou = tf.math.divide_no_nan(tp, union)
        return tf.reduce_mean(iou)

    def reset_states(self):
        self.total_cm.assign(tf.zeros_like(self.total_cm))


# --------------------------------------------------------------------------- #
# 6) CLASE DE MODELO CON ENTRENAMIENTO PERSONALIZADO
# --------------------------------------------------------------------------- #
class ModeloConEntrenamientoPorFases(tf.keras.Model):
    def __init__(self, img_shape, n_classes, **kwargs):
        super().__init__(**kwargs)
        self.n_classes = n_classes
        self.seg_model, self.backbone = build_model(img_shape, n_classes)
        
        # Métricas para el seguimiento
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.iou_metric = MeanIoU(num_classes=n_classes, name='mean_iou')

    def call(self, inputs, training=False):
        return self.seg_model(inputs, training=training)

    @property
    def metrics(self):
        return [self.loss_tracker, self.iou_metric]

    def train_step(self, data):
        x, y_true = data
        with tf.GradientTape() as tape:
            y_pred = self.seg_model(x, training=True)
            loss = ultimate_iou_loss(y_true, y_pred, self.n_classes)

        trainable_vars = self.seg_model.trainable_variables
        grads = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(grads, trainable_vars))

        self.loss_tracker.update_state(loss)
        self.iou_metric.update_state(y_true, y_pred)
        return {m.name: m.result() for m in self.metrics}
        
    def test_step(self, data):
        x, y_true = data
        y_pred = self.seg_model(x, training=False)
        loss = ultimate_iou_loss(y_true, y_pred, self.n_classes)
        
        self.loss_tracker.update_state(loss)
        self.iou_metric.update_state(y_true, y_pred)
        return {m.name: m.result() for m in self.metrics}


# --------------------------------------------------------------------------- #
# 7) CARGA DE DATOS Y AUMENTO
# --------------------------------------------------------------------------- #
def get_training_augmentation(size=256):
    return A.Compose([
        A.RandomScale(scale_limit=0.2, p=0.5),
        A.PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_REFLECT),
        A.RandomCrop(height=size, width=size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.GaussNoise(p=0.1),
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.2),
    ])

def load_dataset_concurrently(img_dir, mask_dir, target_size=(256, 256), augment=False):
    images, masks = [], []
    try:
        files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png'))]
    except FileNotFoundError:
        print(f"Directorio no encontrado: {img_dir}. Saltando carga.")
        return np.array([]), np.array([])

    aug_pipeline = get_training_augmentation(target_size[0]) if augment else A.Compose([])

    def process_file(filename):
        try:
            img_path = os.path.join(img_dir, filename)
            mask_path = os.path.join(mask_dir, filename.rsplit('.', 1)[0] + '_mask.png')
            
            img_arr = img_to_array(load_img(img_path, target_size=target_size))
            mask_arr = img_to_array(load_img(mask_path, color_mode='grayscale', target_size=target_size))
            
            augmented = aug_pipeline(image=img_arr.astype('uint8'), mask=mask_arr)
            return augmented['image'], augmented['mask']
        except Exception as e:
            print(f"Error procesando {filename}: {e}")
            return None, None

    with ThreadPoolExecutor() as executor:
        desc = f"Cargando y {'aumentando' if augment else 'validando'} datos"
        futures = [executor.submit(process_file, fn) for fn in files]
        for future in tqdm(as_completed(futures), total=len(files), desc=desc):
            img, mask = future.result()
            if img is not None:
                images.append(img)
                masks.append(mask)

    if not images:
        return np.array([]), np.array([])

    X = np.array(images, dtype='float32') / 255.0
    y = np.array(masks, dtype='int32')
    return X, y


# --------------------------------------------------------------------------- #
# 8) BUCLE PRINCIPAL DE ENTRENAMIENTO POR FASES
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # --- Parámetros de configuración ---
    IMG_SHAPE = (256, 256, 3)
    BATCH_SIZE_PHASE1 = 8
    BATCH_SIZE_PHASE2 = 4
    EPOCHS_PHASE1 = 5
    EPOCHS_PHASE2 = 5
    
    # --- Rutas a los datos (modificar según sea necesario) ---
    # Se recomienda usar rutas absolutas para evitar problemas
    TRAIN_IMG_DIR = 'data/train/images'
    TRAIN_MASK_DIR = 'data/train/masks'
    VAL_IMG_DIR = 'data/val/images'
    VAL_MASK_DIR = 'data/val/masks'

    # --- Carga de datos ---
    print("Cargando datos de entrenamiento...")
    train_X, train_y = load_dataset_concurrently(TRAIN_IMG_DIR, TRAIN_MASK_DIR, IMG_SHAPE[:2], augment=True)
    print("Cargando datos de validación...")
    val_X, val_y = load_dataset_concurrently(VAL_IMG_DIR, VAL_MASK_DIR, IMG_SHAPE[:2], augment=False)

    if train_X.size == 0 or val_X.size == 0:
        print("No se pudieron cargar los datos. Creando datos dummy para una prueba de ejecución.")
        train_X = np.random.rand(8, *IMG_SHAPE).astype('float32')
        train_y = np.random.randint(0, 2, (8, IMG_SHAPE[0], IMG_SHAPE[1], 1)).astype('int32')
        val_X, val_y = train_X, train_y

    N_CLASSES = int(np.max(train_y)) + 1
    print(f"\nDatos cargados. {len(train_X)} imágenes de entrenamiento, {len(val_X)} de validación.")
    print(f"Número de clases detectado: {N_CLASSES}")
    
    # --- Directorios para guardar resultados ---
    os.makedirs('./checkpoints', exist_ok=True)
    os.makedirs('./logs', exist_ok=True)
    CKPT_PHASE1 = './checkpoints/phase1_best_model.keras'
    CKPT_PHASE2 = './checkpoints/phase2_best_model.keras'
    MONITOR_METRIC = 'val_mean_iou'

    with tf.device(device):
        # --- Instanciar el modelo personalizado ---
        modelo = ModeloConEntrenamientoPorFases(img_shape=IMG_SHAPE, n_classes=N_CLASSES)
        modelo.build(input_shape=(None, *IMG_SHAPE))
        modelo.seg_model.summary()

        # ------------------- FASE 1: Entrenar el decodificador -------------------
        print("\n" + "="*50)
        print("FASE 1: Entrenando solo el decodificador (backbone congelado)")
        print("="*50)
        modelo.backbone.trainable = False
        modelo.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4))
        
        cbs1 = [
            ModelCheckpoint(CKPT_PHASE1, save_best_only=True, monitor=MONITOR_METRIC, mode='max', verbose=1),
            EarlyStopping(monitor=MONITOR_METRIC, mode='max', patience=10, restore_best_weights=True, verbose=1),
            TensorBoard(log_dir='./logs/phase1_decoder', update_freq='epoch')
        ]
        
        modelo.fit(train_X, train_y,
                   batch_size=BATCH_SIZE_PHASE1,
                   epochs=EPOCHS_PHASE1,
                   validation_data=(val_X, val_y),
                   callbacks=cbs1)
        
        print(f"Cargando los mejores pesos de la Fase 1 desde: {CKPT_PHASE1}")
        modelo.load_weights(CKPT_PHASE1)

        # ----------------- FASE 2: Ajuste fino (Fine-tuning) completo -----------------
        print("\n" + "="*50)
        print("FASE 2: Ajuste fino de todo el modelo")
        print("="*50)
        modelo.backbone.trainable = True
        modelo.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-5, weight_decay=5e-5)) # Tasa de aprendizaje mucho más baja

        cbs2 = [
            ModelCheckpoint(CKPT_PHASE2, save_best_only=True, monitor=MONITOR_METRIC, mode='max', verbose=1),
            EarlyStopping(monitor=MONITOR_METRIC, mode='max', patience=15, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1),
            TensorBoard(log_dir='./logs/phase2_finetune', update_freq='epoch')
        ]

        modelo.fit(train_X, train_y,
                   batch_size=BATCH_SIZE_PHASE2,
                   epochs=EPOCHS_PHASE2,
                   initial_epoch=0,  # Reiniciar contador de épocas
                   validation_data=(val_X, val_y),
                   callbacks=cbs2)

    print("\nEntrenamiento por fases completado.")