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
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import layers
from datasets import load_dataset
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer
)

# --- 1. Definir Parámetros ---

MODEL_CHECKPOINT = "microsoft/efficientvit-b0"
NEW_MODEL_NAME = "efficientvit-b0-finetuned-custom"

# Directorios de imágenes y máscaras
train_images_dir = 'Balanced/train/images'
train_masks_dir = 'Balanced/train/masks'
val_images_dir = 'Balanced/val/images'
val_masks_dir = 'Balanced/val/masks'

IMG_SIZE = (256, 256)  # Tamaño de las imágenes para preprocesar

# --- 2. Función de Carga de Dataset ---

def load_dataset_from_dirs(image_directory, mask_directory, target_size=(256, 256)):
    images, masks = [], []
    files = [f for f in os.listdir(image_directory) if f.endswith(('.jpg', '.png'))]
    
    for fn in files:
        img = load_img(os.path.join(image_directory, fn), target_size=target_size)
        img_arr = img_to_array(img)
        
        mask_fn = fn.rsplit('.', 1)[0] + '_mask.png'
        mask = load_img(os.path.join(mask_directory, mask_fn), color_mode='grayscale', target_size=target_size)
        mask_arr = img_to_array(mask)

        images.append(img_arr)
        masks.append(mask_arr)
    
    X = np.array(images, dtype='float32')
    y = np.squeeze(np.array(masks, dtype='int32'), axis=-1)
    return X, y

# Cargar dataset de entrenamiento y validación
train_X, train_y = load_dataset_from_dirs(train_images_dir, train_masks_dir, target_size=IMG_SIZE)
val_X, val_y = load_dataset_from_dirs(val_images_dir, val_masks_dir, target_size=IMG_SIZE)

# --- 3. Preprocesamiento para EfficientNetV2 ---

def preprocess_fn(images):
    return tf.keras.applications.efficientnet_v2.preprocess_input(images)

train_X_processed = preprocess_fn(train_X)
val_X_processed = preprocess_fn(val_X)

# --- 4. Modelo para Segmentación: DeepLabV3 con EfficientNetV2 ---

def ASPP(x, out_channels=256, rates=(6, 12, 18)):
    convs = [layers.Conv2D(out_channels, 1, padding="same", use_bias=False)(x)]
    for r in rates:
        convs.append(layers.Conv2D(out_channels, 3, dilation_rate=r, padding="same", use_bias=False)(x))
    pool = layers.GlobalAveragePooling2D(keepdims=True)(x)
    pool = layers.Conv2D(out_channels, 1, padding="same", use_bias=False)(pool)
    pool = layers.Lambda(lambda t: tf.image.resize(t[0], tf.shape(t[1])[1:3], method="bilinear"))([pool, x])
    convs.append(pool)
    y = layers.Concatenate()(convs)
    y = layers.Conv2D(out_channels, 1, padding="same", use_bias=False)(y)
    y = layers.BatchNormalization()(y)
    return layers.Activation("relu")(y)

def build_custom_deeplab(input_shape=(256, 256, 3), num_classes=6):
    inputs = Input(shape=input_shape)
    augmented_inputs = data_augmentation(inputs)
    backbone = EfficientNetV2S(include_top=False, weights='imagenet', input_tensor=augmented_inputs)
    
    low_level_features = None
    target_shape = (64, 64)

    for layer in reversed(backbone.layers):
        if 'add' in layer.name and layer.output.shape[1:3] == target_shape:
            low_level_features = layer.output
            break

    if low_level_features is None:
        raise ValueError(f"Could not find a feature layer with shape {target_shape}")

    high_level_features = backbone.output
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
    
    final_model = Model(inputs, x)
    return final_model, backbone

model, backbone_layer = build_custom_deeplab(input_shape=(*IMG_SIZE, 3), num_classes=6)

# --- 5. Métricas y Entrenamiento ---

iou_metric = MeanIoUFromLogits(num_classes=6)

# Phase 1: Entrenar el decodificador
backbone_layer.trainable = False
model.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-5),
    loss=combined_loss,
    metrics=[iou_metric]
)

cbs1 = [
    EarlyStopping(monitor='val_mean_iou', mode='max', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_mean_iou', mode='max', factor=0.5, patience=5, min_lr=1e-6),
    ModelCheckpoint('phase1_custom_model.keras', monitor='val_mean_iou', mode='max', save_best_only=True)
]

model.fit(
    train_X_processed, train_y,
    validation_data=(val_X_processed, val_y),
    batch_size=16, epochs=20,
    callbacks=cbs1
)

# Phase 2: Fine-tuning completo
backbone_layer.trainable = True
model.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-6),
    loss=combined_loss,
    metrics=[iou_metric]
)

cbs2 = [
    EarlyStopping(monitor='val_mean_iou', mode='max', patience=15, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_mean_iou', mode='max', factor=0.2, patience=5, min_lr=1e-7),
    ModelCheckpoint('weed_detector_final_model.keras', monitor='val_mean_iou', mode='max', save_best_only=True)
]

model.fit(
    train_X_processed, train_y,
    validation_data=(val_X_processed, val_y),
    batch_size=8,
    epochs=20
)
