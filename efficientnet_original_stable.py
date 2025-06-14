import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, UpSampling2D, BatchNormalization, Activation,
    Concatenate, GlobalAveragePooling2D, Lambda, Dropout
)
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from concurrent.futures import ThreadPoolExecutor, as_completed
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tqdm import tqdm
from tensorflow.keras import layers

# Capa de aumento de datos
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(height_factor=0.2, width_factor=0.2),
    layers.RandomContrast(0.2),
], name="data_augmentation")

# 1) Función de carga de datos optimizada
def load_dataset(image_directory, mask_directory, target_size=(256, 256)):
    images, masks = [], []
    files = [f for f in os.listdir(image_directory) if f.endswith(('.jpg', '.png'))]
    
    with ThreadPoolExecutor() as exe:
        futures = {
            exe.submit(
                lambda fn: (
                    img_to_array(load_img(os.path.join(image_directory, fn), target_size=target_size)),
                    img_to_array(load_img(
                        os.path.join(mask_directory, fn.rsplit('.', 1)[0] + '_mask.png'),
                        color_mode='grayscale',
                        target_size=target_size
                    ))
                ), fn
            ): fn for fn in files
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Loading data"):
            img_arr, mask_arr = future.result()
            images.append(img_arr)
            masks.append(mask_arr)
            
    X = np.array(images, dtype='float32')
    y = np.squeeze(np.array(masks, dtype='int32'), axis=-1)
    
    return X, y

# 2) Directorios del dataset
train_images_dir = 'Balanced/train/images'
train_masks_dir = 'Balanced/train/masks'
val_images_dir = 'Balanced/val/images'
val_masks_dir = 'Balanced/val/masks'

IMG_SIZE = (256, 256)
train_X, train_y = load_dataset(train_images_dir, train_masks_dir, target_size=IMG_SIZE)
val_X, val_y = load_dataset(val_images_dir, val_masks_dir, target_size=IMG_SIZE)

# Función de preprocesamiento para EfficientNetV2
def preprocess_fn(images):
    return tf.keras.applications.efficientnet_v2.preprocess_input(images)

train_X_processed = preprocess_fn(train_X)
val_X_processed = preprocess_fn(val_X)

# 3) Módulo ASPP
def ASPP(x, out_channels=256, rates=(6, 12, 18)):
    convs = [layers.Conv2D(out_channels, 1, padding="same", use_bias=False)(x)]
    for r in rates:
        convs.append(layers.Conv2D(out_channels, 3, dilation_rate=r, padding="same", use_bias=False)(x))
    pool = layers.GlobalAveragePooling2D(keepdims=True)(x)
    pool = layers.Conv2D(out_channels, 1, padding="same", use_bias=False)(pool)
    pool = layers.Lambda(
        lambda t: tf.image.resize(t[0], tf.shape(t[1])[1:3], method="bilinear")
    )([pool, x])
    convs.append(pool)
    y = layers.Concatenate()(convs)
    y = layers.Conv2D(out_channels, 1, padding="same", use_bias=False)(y)
    y = layers.BatchNormalization()(y)
    return layers.Activation("relu")(y)

# 4) Construcción del modelo con lógica dinámica para encontrar capas
def build_custom_deeplab(input_shape=(256, 256, 3), num_classes=6):
    """
    Construye el modelo DeepLabV3+ y devuelve tanto el modelo final como 
    una referencia al modelo backbone para facilitar el control de su entrenamiento.
    """
    inputs = Input(shape=input_shape)
    augmented_inputs = data_augmentation(inputs)

    backbone = EfficientNetV2S(
        include_top=False,
        weights='imagenet',
        input_tensor=augmented_inputs
    )
    
    # --- INICIO DE LA NUEVA LÓGICA ---
    # Encuentra dinámicamente las capas de extracción de características (skip connections)
    # en lugar de usar nombres fijos. Esto es más robusto.
    low_level_features = None
    
    # Buscamos una capa de bajo nivel con una resolución espacial de 64x64
    target_shape = (64, 64)

    for layer in reversed(backbone.layers):
        # Nos enfocamos en las capas de adición residual ('add') que son buenos puntos de extracción
        if 'add' in layer.name and layer.output.shape[1:3] == target_shape:
            low_level_features = layer.output
            print(f"Capa de características de bajo nivel encontrada: '{layer.name}' con shape {layer.output.shape}")
            break

    if low_level_features is None:
        raise ValueError(f"No se pudo encontrar una capa de características con la forma {target_shape}")

    # Las características de alto nivel son la salida final del backbone
    high_level_features = backbone.output
    print(f"Capa de características de alto nivel: Salida del backbone con shape {high_level_features.shape}")
    # --- FIN DE LA NUEVA LÓGICA ---

    x = ASPP(high_level_features, out_channels=256, rates=[6, 12, 18])
    
    upsampling_factor = low_level_features.shape[1] // high_level_features.shape[1]
    x = UpSampling2D(size=(upsampling_factor, upsampling_factor), interpolation='bilinear')(x)
    
    low = Conv2D(48, 1, padding='same', use_bias=False)(low_level_features)
    low = BatchNormalization()(low)
    low = Activation('relu')(low)
    
    x = Concatenate()([x, low])
    x = Conv2D(256, 3, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x); x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    x = Conv2D(256, 3, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x); x = Activation('relu')(x)
    
    x = Conv2D(num_classes, 1, padding='same')(x)
    x = UpSampling2D(size=(4, 4), interpolation='bilinear')(x)
    
    # Devolvemos el modelo y el backbone por separado
    final_model = Model(inputs, x)
    return final_model, backbone

# Funciones de pérdida y métricas (sin cambios)
def dice_loss(y_true, y_pred):
    y_true = tf.one_hot(tf.cast(y_true, tf.int32), num_classes)
    y_pred = tf.nn.softmax(y_pred, axis=-1)
    numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=-1)
    denominator = tf.reduce_sum(y_true + y_pred, axis=-1)
    return 1 - (numerator + 1e-6) / (denominator + 1e-6)

def combined_loss(y_true, y_pred):
    return dice_loss(y_true, y_pred) + tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)

def mean_iou(y_true, y_pred):
    y_pred = tf.argmax(y_pred, axis=-1)
    return tf.keras.metrics.MeanIoU(num_classes=num_classes)(y_true, y_pred)

# --- INSTANCIACIÓN Y ENTRENAMIENTO CON LA NUEVA LÓGICA ---
num_classes = int(np.max(train_y) + 1)

# Recibimos el modelo y el backbone por separado
model, backbone_layer = build_custom_deeplab(input_shape=(*IMG_SIZE, 3), num_classes=num_classes)
model.summary()

# 5) Proceso de entrenamiento en dos fases
# Fase 1: Entrenar el decodificador y las capas nuevas
print("\nCongelando el backbone para la Fase 1...")
backbone_layer.trainable = False # Usamos la referencia directa, evitando el error

model.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-5),
    loss=combined_loss,
    metrics=[mean_iou]
)

cbs1 = [
    EarlyStopping(monitor='val_mean_iou', mode='max', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_mean_iou', mode='max', factor=0.5, patience=5, min_lr=1e-6),
    ModelCheckpoint('phase1_custom_model.h5', monitor='val_mean_iou', mode='max', save_best_only=True)
]

print("\n--- Iniciando Fase 1: Entrenando el Decodificador ---")
model.fit(
    train_X_processed, train_y,
    validation_data=(val_X_processed, val_y),
    batch_size=16, epochs=50,
    callbacks=cbs1
)

# Fase 2: Ajuste fino del modelo completo
print("\nDescongelando el backbone para la Fase 2 (fine-tuning)...")
backbone_layer.trainable = True # Usamos la referencia directa nuevamente

model.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-6),
    loss=combined_loss,
    metrics=[mean_iou]
)

cbs2 = [
    EarlyStopping(monitor='val_mean_iou', mode='max', patience=15, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_mean_iou', mode='max', factor=0.2, patience=5, min_lr=1e-7),
    ModelCheckpoint('weed_detector_final_model.h5', monitor='val_mean_iou', mode='max', save_best_only=True)
]

print("\n--- Iniciando Fase 2: Ajuste Fino del Modelo Completo ---")
model.fit(
    train_X_processed, train_y,
    validation_data=(val_X_processed, val_y),
    batch_size=8,
    epochs=50,
    callbacks=cbs2
)

print("\nEntrenamiento completo. El modelo final se ha guardado como 'weed_detector_final_model.h5'")