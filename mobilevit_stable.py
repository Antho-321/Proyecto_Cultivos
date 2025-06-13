# -*- coding: utf-8 -*-
"""
claude_mobilevit_deeplab_iou_optimized.py

Script final que implementa MobileViT-S como backbone. Esta versión RESTAURA
el decodificador U-Net completo con skip connections, aprovechando la naturaleza
híbrida de MobileViT para obtener lo mejor de las CNNs y los Transformers.
"""

# 1) Imports ------------------------------------------------------------------
import os
os.environ['MPLBACKEND'] = 'agg'

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
from tensorflow import keras
from tensorflow.keras.layers import (
    Input, Conv2D,
    BatchNormalization, Activation,
    Concatenate, UpSampling2D,
    Dense, Multiply, Reshape, GlobalAveragePooling2D
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
)
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

from lovasz_losses_tf import lovasz_softmax
### NUEVO ###
# Importa la función para construir el modelo MobileViT desde el archivo que creamos
from mobilevit_keras import mobile_vit_s

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

def conv_block(x, filters, kernel_size, strides=1, padding='same', dilation_rate=1):
    """Bloque convolucional simple con Conv2D, BatchNorm y Swish."""
    x = Conv2D(
        filters, 
        kernel_size, 
        strides=strides, 
        padding=padding,
        dilation_rate=dilation_rate,
        use_bias=False
    )(x)
    x = BatchNormalization()(x)
    x = Activation('swish')(x)
    return x

def ASPP_mejorado(x, dilation_rates=[6, 12, 18]):
    """Módulo ASPP (Atrous Spatial Pyramid Pooling)."""
    input_shape = x.shape
    input_filters = input_shape[-1]
    projected_filters = 256
    
    # Pooling a nivel de imagen
    image_pooling = GlobalAveragePooling2D()(x)
    image_pooling = Reshape((1, 1, input_filters))(image_pooling)
    image_pooling = conv_block(image_pooling, projected_filters, 1)
    image_pooling = UpSampling2D(size=(input_shape[1], input_shape[2]), interpolation='bilinear')(image_pooling)

    # Convolución 1x1
    conv_1x1 = conv_block(x, projected_filters, 1)
    
    # Convoluciones 'atrous'
    atrous_convs = [conv_block(x, projected_filters, 3, dilation_rate=rate) for rate in dilation_rates]
    
    # Concatenación y proyección final
    concatenated = Concatenate()([image_pooling, conv_1x1] + atrous_convs)
    projected = conv_block(concatenated, projected_filters, 1)
    return projected

### NUEVO ###
def build_mobilevit_model(shape=(256, 256, 3), num_classes_arg=None):
    """
    Construye el modelo de segmentación completo usando MobileViT-S como backbone
    y un decodificador U-Net con skip connections.
    """
    # --- Definición del backbone ---
    # `include_top=False` es implícito ya que no pedimos un clasificador.
    # El modelo `mobile_vit_s` está modificado para devolver las capas intermedias.
    backbone = mobile_vit_s(input_shape=shape)
    
    # --- Extracción de las skip connections y el bottleneck ---
    # El backbone devuelve una lista de tensores: [s1, s2, s3, s4, bottleneck]
    s1, s2, s3, s4, bottleneck = backbone.outputs
    inp = backbone.input

    print("Shapes de las Skip Connections:")
    print(f"  s1 (128x128): {s1.shape}")
    print(f"  s2 (64x64):   {s2.shape}")
    print(f"  s3 (32x32):   {s3.shape}")
    print(f"  s4 (16x16):   {s4.shape}")
    print(f"  Bottleneck (8x8): {bottleneck.shape}")

    # --- Cuello de botella con ASPP Mejorado ---
    # Tasas de dilatación más pequeñas, adecuadas para el mapa de características de 8x8
    x = ASPP_mejorado(bottleneck, dilation_rates=[2, 4, 6]) # 8x8 -> 8x8

    ### RESTAURADO ###
    # --- Rama del decodificador (Upsampling con lógica U-Net completa) ---
    # Usamos las skip connections del backbone MobileViT.

    # Bloque 1: De 8x8 a 16x16
    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    x = Concatenate()([x, s4]) # Concatenar con la skip connection de 16x16
    x = conv_block(x, filters=256, kernel_size=3)
    x = conv_block(x, filters=256, kernel_size=3)

    # Bloque 2: De 16x16 a 32x32
    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    x = Concatenate()([x, s3]) # Concatenar con la skip connection de 32x32
    x = conv_block(x, filters=128, kernel_size=3)
    x = conv_block(x, filters=128, kernel_size=3)

    # Bloque 3: De 32x32 a 64x64
    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    x = Concatenate()([x, s2]) # Concatenar con la skip connection de 64x64
    x = conv_block(x, filters=64, kernel_size=3)
    x = conv_block(x, filters=64, kernel_size=3)

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
    """ Tversky Loss para segmentación. """
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
    """ Pérdida de bordes basada en el Dice score (versión compatible con XLA). """
    y_true_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), depth=num_classes, axis=-1)
    y_pred_probs = tf.nn.softmax(y_pred, axis=-1)

    # Aproximar gradiente morfológico con max_pool2d (XLA-friendly)
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

def ultimate_iou_loss(y_true, y_pred, epoch=0):
    """ Pérdida combinada estática. """
    lovasz_weight = 0.5
    boundary_weight = 0.25
    tversky_weight = 0.25
    
    lov = lovasz_weight * lovasz_softmax(y_pred, tf.cast(y_true, tf.int32), per_image=True)
    abe_dice = boundary_weight * adaptive_boundary_enhanced_dice_loss_tf(y_true, y_pred, gamma=2.0)
    tver = tversky_weight * tversky_loss(y_true, y_pred, alpha=0.3, beta=0.7) # Penaliza más FN
    
    return lov + abe_dice + tver

class MeanIoU(tf.keras.metrics.Metric):
    """ Métrica Mean Intersection over Union. """
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

# 8) Modelo Personalizado con Bucle de Entrenamiento Optimizado para IoU -----
class IoUOptimizedModel(keras.Model):
    def __init__(self, shape, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        
        ### CAMBIO ###
        # Cambiamos la función que construye el modelo a la versión con MobileViT
        self.seg_model, self.backbone = build_mobilevit_model(shape, num_classes_arg=num_classes)
        
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

### CAMBIO ###
checkpoint_dir = './checkpoints_mobilevit'
os.makedirs(checkpoint_dir, exist_ok=True)
ckpt1_path = os.path.join(checkpoint_dir, 'phase1_mobilevit_model.keras')
ckpt2_path = os.path.join(checkpoint_dir, 'phase2_mobilevit_model.keras')

val_gen = (val_X, val_y)

with tf.device(device):
    iou_aware_model = IoUOptimizedModel(shape=train_X.shape[1:], num_classes=num_classes)
    iou_aware_model.build(input_shape=(None, *train_X.shape[1:]))
    
    model = iou_aware_model.seg_model
    backbone = iou_aware_model.backbone
    
    model.summary()
    # backbone.summary() # Descomentar para ver la estructura interna del backbone
    
    MONITOR_METRIC = 'val_enhanced_mean_iou'
    
    # --- Fase 1: entrenar cabeza ---
    print("\n--- Fase 1: Entrenando solo el decoder (backbone MobileViT congelado) ---")
    backbone.trainable = False
    
    iou_aware_model.compile(optimizer=tf.keras.optimizers.AdamW(1e-3, weight_decay=1e-4))
    
    cbs1 = [
        ModelCheckpoint(ckpt1_path, save_best_only=True, monitor=MONITOR_METRIC, mode='max', verbose=1),
        EarlyStopping(monitor=MONITOR_METRIC, mode='max', patience=15, restore_best_weights=True),
        TensorBoard(log_dir='./logs/mobilevit_phase1_head', update_freq='epoch')
    ]
    
    iou_aware_model.fit(train_X, train_y, batch_size=16, epochs=50, # Batch size más grande gracias a la eficiencia
                        validation_data=val_gen, callbacks=cbs1)
    
    print(f"Cargando mejores pesos de la fase 1 desde: {ckpt1_path}")
    iou_aware_model.load_weights(ckpt1_path)

    # --- Fase 2: fine-tuning de todo el modelo con LR bajo ---
    print("\n--- Fase 2: Fine-tuning de todo el modelo (backbone + decoder) ---")
    backbone.trainable = True
    
    iou_aware_model.compile(optimizer=tf.keras.optimizers.AdamW(2e-5, weight_decay=1e-5)) # LR bajo
    
    cbs2 = [
        ModelCheckpoint(ckpt2_path, save_best_only=True, monitor=MONITOR_METRIC, mode='max', verbose=1),
        EarlyStopping(monitor=MONITOR_METRIC, mode='max', patience=20, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-7, verbose=1),
        TensorBoard(log_dir='./logs/mobilevit_phase2_full', update_freq='epoch')
    ]

    iou_aware_model.fit(train_X, train_y, batch_size=8, epochs=80, # Batch size más pequeño para el fine-tuning completo
                        validation_data=val_gen, callbacks=cbs2, initial_epoch=0)
    
    print(f"Cargando mejores pesos finales desde: {ckpt2_path}")
    iou_aware_model.load_weights(ckpt2_path)
    
    eval_model = iou_aware_model.seg_model

# 10) Evaluación & visualización ----------------------------------------------
def apply_crf(image, prediction, num_classes):
    """Apply CRF post-processing"""
    try:
        import pydensecrf.densecrf as dcrf
        from pydensecrf.utils import unary_from_softmax
    except ImportError:
        print("pydensecrf no está instalado. Omitiendo CRF. Instálalo con 'pip install pydensecrf'.")
        return np.argmax(prediction, axis=-1)

    d = dcrf.DenseCRF2D(image.shape[1], image.shape[0], num_classes)
    # Transponer la predicción a (num_classes, height, width)
    unary = unary_from_softmax(prediction.transpose(2, 0, 1))
    d.setUnaryEnergy(unary)
    
    # Potenciales pairwise (suavizado y apariencia)
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=50, srgb=13, rgbim=image, compat=10)
    
    # Inferencia
    Q = d.inference(5)
    return np.argmax(Q, axis=0).reshape(image.shape[:2])

def evaluate_model(model_to_eval, X, y, num_classes, use_crf=False, batch_size=8):
    """Evalúa un modelo de segmentación."""
    
    print("Realizando predicciones estándar...")
    softmax_preds = model_to_eval.predict(X, batch_size=batch_size)
    
    if use_crf:
        print("Aplicando post-procesamiento CRF...")
        final_masks = []
        for i in tqdm(range(len(X)), desc="Aplicando CRF"):
            original_image = (X[i] * 255).astype(np.uint8)
            pred_probs = softmax_preds[i]
            crf_output = apply_crf(original_image, pred_probs, num_classes)
            final_masks.append(crf_output)
        final_masks = np.array(final_masks)
    else:
        final_masks = np.argmax(softmax_preds, axis=-1)
    
    print("Calculando métricas de evaluación...")
    ious = []
    if y.ndim == 4 and y.shape[-1] == 1:
        y = np.squeeze(y, axis=-1)
        
    for cls in range(num_classes):
        y_true_class = (y == cls).astype(int)
        y_pred_class = (final_masks == cls).astype(int)
        
        intersection = np.sum(y_true_class * y_pred_class)
        union = np.sum(y_true_class) + np.sum(y_pred_class) - intersection
        
        iou = intersection / union if union > 0 else 0.0
        ious.append(iou)
    
    mean_iou = np.mean(ious)
    pixel_accuracy = np.mean(final_masks == y)
    
    return {
        'mean_iou': mean_iou, 
        'class_ious': ious, 
        'pixel_accuracy': pixel_accuracy
    }

def visualize_predictions(model_to_eval,X,y,num_samples=5):
    """Visualiza algunas predicciones del modelo."""
    idxs = np.random.choice(len(X),num_samples,replace=False)
    preds = model_to_eval.predict(X[idxs])
    masks = np.argmax(preds,axis=-1)
    
    # Crear un mapa de colores discreto
    cmap = plt.cm.get_cmap('tab10', num_classes)
    
    plt.figure(figsize=(12, 4 * num_samples))
    for i,ix in enumerate(idxs):
        plt.subplot(num_samples, 3, 3*i + 1)
        plt.imshow(X[ix])
        plt.title('Image')
        plt.axis('off')
        
        plt.subplot(num_samples, 3, 3*i + 2)
        plt.imshow(y[ix], cmap=cmap, vmin=0, vmax=num_classes - 1)
        plt.title('GT Mask')
        plt.axis('off')
        
        plt.subplot(num_samples, 3, 3*i + 3)
        plt.imshow(masks[i], cmap=cmap, vmin=0, vmax=num_classes - 1)
        plt.title('Pred Mask')
        plt.axis('off')
        
    plt.tight_layout()
    ### CAMBIO ###
    plt.savefig('predictions_visualization_mobilevit.png')
    plt.close()
    print("\nVisualización de predicciones guardada en 'predictions_visualization_mobilevit.png'")

print("\n--- Iniciando evaluación del modelo final con MobileViT ---")

# Caso 1: Evaluación estándar
print("\n--- Configuración 1: Predicción Estándar ---")
metrics_base = evaluate_model(eval_model, val_X, val_y, num_classes=num_classes, batch_size=8)
print(f"  Mean IoU: {metrics_base['mean_iou']:.4f}")
print(f"  Pixel Accuracy: {metrics_base['pixel_accuracy']:.4f}")
for i, iou in enumerate(metrics_base['class_ious']): print(f"      Class {i}: {iou:.4f}")

# Caso 2: Evaluación con post-procesamiento CRF
print("\n--- Configuración 2: Con Post-procesamiento CRF ---")
metrics_crf = evaluate_model(eval_model, val_X, val_y, num_classes=num_classes, use_crf=True, batch_size=8)
print(f"  Mean IoU: {metrics_crf['mean_iou']:.4f}")
print(f"  Pixel Accuracy: {metrics_crf['pixel_accuracy']:.4f}")

# Visualizar predicciones finales
visualize_predictions(eval_model,val_X,val_y)
print("\nEntrenamiento y evaluación completados.")