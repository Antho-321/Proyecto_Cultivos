import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, UpSampling2D, BatchNormalization, Activation,
    Concatenate, GlobalAveragePooling2D
)
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from concurrent.futures import ThreadPoolExecutor, as_completed
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tqdm import tqdm
from tensorflow.keras import layers

# 1) Función de carga de datos (idéntica a tu u_net_prot_v3.py)
def load_dataset(image_directory, mask_directory, target_size=(128,128)):
    images, masks = [], []
    files = [f for f in os.listdir(image_directory) if f.endswith(('.jpg','.png'))]
    with ThreadPoolExecutor() as exe:
        futures = {
            exe.submit(
                lambda fn: (
                    img_to_array(load_img(os.path.join(image_directory, fn), target_size=target_size)),
                    img_to_array(load_img(
                        os.path.join(mask_directory, fn.rsplit('.',1)[0]+'_mask.png'),
                        color_mode='grayscale',
                        target_size=target_size
                    ))
                ), fn
            ): fn for fn in files
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Cargando datos"):
            img_arr, mask_arr = future.result()
            images.append(img_arr); masks.append(mask_arr)
    # Normalizamos imágenes a [0,1] y convertimos las máscaras a int32 sin la última dim
    X = np.array(images, dtype='float32') / 255.0
    y = np.squeeze(np.array(masks, dtype='int32'), axis=-1)
    return X, y

# 2) Directorios de tu dataset
train_images_dir = 'Balanced/train/images'
train_masks_dir  = 'Balanced/train/masks'
val_images_dir   = 'Balanced/val/images'
val_masks_dir    = 'Balanced/val/masks'

train_X, train_y = load_dataset(train_images_dir, train_masks_dir)
val_X,   val_y   = load_dataset(val_images_dir,   val_masks_dir)

# 3) Módulo ASPP
def ASPP(x, out_channels=256, rates=(6, 12, 18)):
    convs = [layers.Conv2D(out_channels, 1, padding="same", use_bias=False)(x)]
    for r in rates:
        convs.append(
            layers.Conv2D(out_channels, 3, dilation_rate=r,
                          padding="same", use_bias=False)(x)
        )

    # Image-level features branch
    pool = layers.GlobalAveragePooling2D(keepdims=True)(x)     # (B,1,1,C)
    pool = layers.Conv2D(out_channels, 1, padding="same", use_bias=False)(pool)

    # Resize to match feature map spatial dims
    pool = layers.Lambda(
        lambda t: tf.image.resize(t[0], tf.shape(t[1])[1:3],
                                  method="bilinear")
    )([pool, x])

    convs.append(pool)

    y = layers.Concatenate()(convs)
    y = layers.Conv2D(out_channels, 1, padding="same", use_bias=False)(y)
    y = layers.BatchNormalization()(y)
    return layers.Activation("relu")(y)

# 4) DeepLabV3+ con ResNet-50 backbone
def build_deeplabv3plus(input_shape=(128,128,3), num_classes=6, backbone_weights='resnet.hdf5'):
    inputs = Input(shape=input_shape)

    # Backbone
    backbone = ResNet50(include_top=False, weights=None, input_tensor=inputs)
    backbone.load_weights(backbone_weights, by_name=True, skip_mismatch=True)

    # Features
    low_level  = backbone.get_layer('conv2_block3_out').output   # 32×32
    high_level = backbone.get_layer('conv5_block3_out').output   # 4×4

    # ASPP y decoder
    x = ASPP(high_level, out_channels=256, rates=[6,12,18])

    # 4×4 → 32×32  (factor 8)
    x = UpSampling2D(size=(8,8), interpolation='bilinear')(x)

    low = Conv2D(48, 1, padding='same', use_bias=False)(low_level)
    low = BatchNormalization()(low)
    low = Activation('relu')(low)

    x = Concatenate()([x, low])        # shapes ahora coinciden: (None,32,32,*)

    x = Conv2D(256,3,padding='same',use_bias=False)(x)
    x = BatchNormalization()(x); x = Activation('relu')(x)
    x = Conv2D(256,3,padding='same',use_bias=False)(x)
    x = BatchNormalization()(x); x = Activation('relu')(x)
    x = Conv2D(num_classes, 1, padding='same')(x)

    # 32×32 → 128×128
    x = UpSampling2D(size=(4,4), interpolation='bilinear')(x)

    return Model(inputs, x)

# Instanciamos el modelo
num_classes = int(np.max(train_y) + 1)  # asumiendo etiquetas 0…C-1
model = build_deeplabv3plus(input_shape=train_X.shape[1:], num_classes=num_classes)

# 5) Fine-tuning en dos fases
# ---- Fase 1: entrenar solo ASPP+decoder ----
for layer in model.layers:
    if any(stage in layer.name for stage in ['conv1','conv2','conv3','conv4','conv5']):
        layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

cbs1 = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
    ModelCheckpoint('phase1_deeplabv3plus.h5', save_best_only=True)
]

model.fit(
    train_X, train_y,
    validation_data=(val_X, val_y),
    batch_size=16, epochs=50,
    callbacks=cbs1
)

# ---- Fase 2: descongelar todo y afinar ----
for layer in model.layers:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

cbs2 = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7),
    ModelCheckpoint('deeplabv3plus_finetuned.h5', save_best_only=True)
]

model.fit(
    train_X, train_y,
    validation_data=(val_X, val_y),
    batch_size=8, epochs=50,
    callbacks=cbs2
)

# El modelo final queda guardado en "deeplabv3plus_finetuned.h5"
# Puedes cargarlo más tarde con:
model = tf.keras.models.load_model('deeplabv3plus_finetuned.h5')