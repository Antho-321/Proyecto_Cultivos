import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, UpSampling2D, BatchNormalization, Activation,
    Concatenate, GlobalAveragePooling2D, Lambda, Dropout, Layer
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from transformers import TFEfficientViTModel

# --- 1. Definir Parámetros ---

# Checkpoint del modelo EfficientViT-B3 de Hugging Face
MODEL_CHECKPOINT = "microsoft/efficientvit-b3"
NEW_MODEL_NAME = "efficientvit-b3-finetuned-custom"

# Directorios (asumiendo que existen)
train_images_dir = 'Balanced/train/images'
train_masks_dir = 'Balanced/train/masks'
val_images_dir = 'Balanced/val/images'
val_masks_dir = 'Balanced/val/masks'

IMG_SIZE = (256, 256)
NUM_CLASSES = 6 # Número de clases para tu tarea de segmentación

# --- Placeholders para funciones no definidas en el snippet original ---
# ¡IMPORTANTE! Reemplaza estas implementaciones con las tuyas.

def data_augmentation(x):
    # Por ahora, solo devolvemos la entrada.
    # Aquí puedes añadir tus capas de aumento de datos de Keras.
    return x

class MeanIoUFromLogits(tf.keras.metrics.MeanIoU):
    """Métrica MeanIoU que funciona con logits (salidas del modelo antes de softmax)."""
    def __init__(self, num_classes, name=None, dtype=None):
        super().__init__(num_classes=num_classes, name=name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convierte los logits a predicciones de clase
        y_pred_labels = tf.argmax(y_pred, axis=-1)
        return super().update_state(y_true, y_pred_labels, sample_weight)

def combined_loss(y_true, y_pred):
    # Placeholder para la función de pérdida. Usando SparseCategoricalCrossentropy.
    # Reemplaza esto con tu función de pérdida combinada si es diferente.
    y_true = tf.cast(y_true, tf.int32)
    return tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)


# --- 2. Función de Carga de Dataset (Sin cambios) ---

def load_dataset_from_dirs(image_directory, mask_directory, target_size=(256, 256)):
    images, masks = [], []
    # Asegúrate de que los directorios existan para evitar errores
    if not os.path.exists(image_directory) or not os.path.exists(mask_directory):
        print(f"Advertencia: El directorio {image_directory} o {mask_directory} no existe. Devolviendo arrays vacíos.")
        return np.array([]), np.array([])
        
    files = [f for f in os.listdir(image_directory) if f.endswith(('.jpg', '.png'))]
    
    for fn in files:
        img = load_img(os.path.join(image_directory, fn), target_size=target_size)
        img_arr = img_to_array(img)
        
        mask_fn = os.path.splitext(fn)[0] + '_mask.png' # Forma más robusta de obtener el nombre
        mask_path = os.path.join(mask_directory, mask_fn)
        
        if os.path.exists(mask_path):
            mask = load_img(mask_path, color_mode='grayscale', target_size=target_size)
            mask_arr = img_to_array(mask)
            images.append(img_arr)
            masks.append(mask_arr)
    
    X = np.array(images, dtype='float32')
    # Añadimos un squeeze al eje de la máscara por si tiene una dimensión extra
    y = np.squeeze(np.array(masks, dtype='int32'))
    return X, y

# Cargar dataset de entrenamiento y validación
print("Cargando datos...")
train_X, train_y = load_dataset_from_dirs(train_images_dir, train_masks_dir, target_size=IMG_SIZE)
val_X, val_y = load_dataset_from_dirs(val_images_dir, val_masks_dir, target_size=IMG_SIZE)

# Comprobación para ver si los datos se cargaron correctamente
if train_X.shape[0] == 0:
    raise ValueError("No se encontraron datos de entrenamiento. Verifica las rutas de los directorios.")

# --- 3. Preprocesamiento ---
# EfficientViT no requiere la misma función `preprocess_input`.
# Una simple normalización a [0, 1] es un buen punto de partida.
train_X_processed = train_X / 255.0
val_X_processed = val_X / 255.0

# --- 4. Modelo de Segmentación: DeepLabV3 con EfficientViT-B3 ---

def ASPP(x, out_channels=256, rates=(6, 12, 18)):
    """Módulo Atrous Spatial Pyramid Pooling."""
    inputs_shape = tf.shape(x)
    b, h, w = inputs_shape[0], inputs_shape[1], inputs_shape[2]
    
    convs = [Conv2D(out_channels, 1, padding="same", use_bias=False)(x)]
    for r in rates:
        convs.append(Conv2D(out_channels, 3, dilation_rate=r, padding="same", use_bias=False)(x))
        
    pool = GlobalAveragePooling2D(keepdims=True)(x)
    pool = Conv2D(out_channels, 1, use_bias=False)(pool)
    pool = tf.image.resize(pool, size=(h, w), method="bilinear")
    
    convs.append(pool)
    y = Concatenate()(convs)
    y = Conv2D(out_channels, 1, padding="same", use_bias=False)(y)
    y = BatchNormalization()(y)
    return Activation("relu")(y)

def build_deeplab_with_vit(input_shape=(256, 256, 3), num_classes=6, backbone_checkpoint=MODEL_CHECKPOINT):
    """Construye un modelo tipo DeepLabV3 usando un backbone de Vision Transformer."""
    inputs = Input(shape=input_shape)
    augmented_inputs = data_augmentation(inputs)

    # Cargar el backbone de EfficientViT desde Hugging Face
    # Lo configuramos para que devuelva los 'hidden_states' de todas sus capas
    backbone = TFEfficientViTModel.from_pretrained(
        backbone_checkpoint,
        output_hidden_states=True,
        include_top=False,
        input_shape=input_shape[1:]
    )
    
    # Obtenemos las salidas del backbone
    outputs = backbone(augmented_inputs)
    hidden_states = outputs.hidden_states

    # Los 'hidden_states' son mapas de características en diferentes profundidades.
    # - high_level_features: El último hidden state, con la información más semántica.
    # - low_level_features: Un hidden state intermedio, con más detalles espaciales.
    # La selección del índice (ej. hidden_states[2]) puede requerir experimentación.
    high_level_features = hidden_states[-1]
    low_level_features = hidden_states[2] # Un estado intermedio
    
    print(f"Forma de High-level features: {high_level_features.shape}")
    print(f"Forma de Low-level features: {low_level_features.shape}")

    # Decodificador DeepLabV3
    x = ASPP(high_level_features, out_channels=256)
    
    # Redimensionar el output de ASPP para que coincida con las low_level_features
    upsampling_factor = low_level_features.shape[1] // x.shape[1]
    x = UpSampling2D(size=(upsampling_factor, upsampling_factor), interpolation='bilinear')(x)
    
    low_level_proj = Conv2D(48, 1, padding='same', use_bias=False)(low_level_features)
    low_level_proj = BatchNormalization()(low_level_proj)
    low_level_proj = Activation('relu')(low_level_proj)
    
    x = Concatenate()([x, low_level_proj])
    x = Conv2D(256, 3, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x); x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    x = Conv2D(256, 3, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x); x = Activation('relu')(x)
    
    x = Conv2D(num_classes, 1, padding='same')(x) # Capa final de clasificación
    
    # Redimensionar a la salida final del tamaño de la imagen de entrada
    x = UpSampling2D(size=(4, 4), interpolation='bilinear')(x)
    
    final_model = Model(inputs, x)
    return final_model, backbone

# Construir el nuevo modelo
model, backbone_layer = build_deeplab_with_vit(input_shape=(*IMG_SIZE, 3), num_classes=NUM_CLASSES)
model.summary()

# --- 5. Métricas y Entrenamiento ---

iou_metric = MeanIoUFromLogits(num_classes=NUM_CLASSES)

# Phase 1: Entrenar el decodificador (congelando el backbone)
print("\n--- Fase 1: Entrenando el decodificador ---")
backbone_layer.trainable = False
model.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-5),
    loss=combined_loss,
    metrics=[iou_metric]
)

cbs1 = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
    ModelCheckpoint('phase1_vit_model.keras', monitor='val_loss', save_best_only=True)
]

# El entrenamiento se inicia solo si hay datos.
model.fit(
    train_X_processed, train_y,
    validation_data=(val_X_processed, val_y),
    batch_size=8, # Reducir batch size si hay problemas de memoria con ViT
    epochs=20,
    callbacks=cbs1
)

# Phase 2: Fine-tuning completo (descongelando el backbone)
print("\n--- Fase 2: Fine-tuning completo del modelo ---")
backbone_layer.trainable = True
model.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-5, weight_decay=1e-6), # Tasa de aprendizaje más baja
    loss=combined_loss,
    metrics=[iou_metric]
)

cbs2 = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-7),
    ModelCheckpoint('final_vit_model.keras', monitor='val_loss', save_best_only=True)
]

model.fit(
    train_X_processed, train_y,
    validation_data=(val_X_processed, val_y),
    batch_size=4, # Reducir aún más el batch size para fine-tuning
    epochs=20,
    callbacks=cbs2
)
