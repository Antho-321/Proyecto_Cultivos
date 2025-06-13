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
import tensorflow_addons as tfa # For AdamW optimizer

# Recommendation: Specific data augmentation for UAVs. 
# This layer applies random rotations, flips, and color adjustments.
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(height_factor=0.2, width_factor=0.2),
    layers.RandomContrast(0.2),
], name="data_augmentation")

# 1) Optimized data loading function
# The target size has been increased to 256x256 for better detail from UAV images.
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

# 2) Dataset directories
train_images_dir = 'Balanced/train/images'
train_masks_dir = 'Balanced/train/masks'
val_images_dir = 'Balanced/val/images'
val_masks_dir = 'Balanced/val/masks'

# Using a larger image size as recommended for UAV data. 
IMG_SIZE = (256, 256)
train_X, train_y = load_dataset(train_images_dir, train_masks_dir, target_size=IMG_SIZE)
val_X, val_y = load_dataset(val_images_dir, val_masks_dir, target_size=IMG_SIZE)

# Preprocessing function for EfficientNetV2
def preprocess_fn(images):
    return tf.keras.applications.efficientnet_v2.preprocess_input(images)

train_X_processed = preprocess_fn(train_X)
val_X_processed = preprocess_fn(val_X)

# 3) ASPP Module (Atrous Spatial Pyramid Pooling) 
# This module captures multi-scale context, which is crucial for detecting weeds of various sizes.
def ASPP(x, out_channels=256, rates=(6, 12, 18)):
    # Main branch
    convs = [layers.Conv2D(out_channels, 1, padding="same", use_bias=False)(x)]
    # Atrous convolutions
    for r in rates:
        convs.append(layers.Conv2D(out_channels, 3, dilation_rate=r, padding="same", use_bias=False)(x))

    # Image-level features branch
    pool = layers.GlobalAveragePooling2D(keepdims=True)(x)
    pool = layers.Conv2D(out_channels, 1, padding="same", use_bias=False)(pool)
    pool = layers.Lambda(
        lambda t: tf.image.resize(t[0], tf.shape(t[1])[1:3], method="bilinear")
    )([pool, x])
    convs.append(pool)
    
    # Concatenate and final convolution
    y = layers.Concatenate()(convs)
    y = layers.Conv2D(out_channels, 1, padding="same", use_bias=False)(y)
    y = layers.BatchNormalization()(y)
    return layers.Activation("relu")(y)

# 4) Building the custom model based on recommendations 
# This architecture uses EfficientNetV2-S as a backbone for its efficiency and performance.
def build_custom_deeplab(input_shape=(256, 256, 3), num_classes=6):
    inputs = Input(shape=input_shape)
    
    # Apply data augmentation
    augmented_inputs = data_augmentation(inputs)

    # Backbone: EfficientNetV2-S (lightweight and powerful) 
    backbone = EfficientNetV2S(include_top=False, weights='imagenet', input_tensor=augmented_inputs)
    
    # Feature extraction points
    # Using endpoints that provide good low-level and high-level features
    low_level = backbone.get_layer('block2d_add').output  # Shape: (None, 64, 64, 48)
    high_level = backbone.get_layer('block6h_add').output # Shape: (None, 8, 8, 160)
    
    # ASPP and Decoder 
    # Note: A Deformable Attention layer could be inserted here for higher precision. 
    x = ASPP(high_level, out_channels=256, rates=[6, 12, 18])
    
    # Upsample ASPP features to match low-level feature dimensions
    x = UpSampling2D(size=(high_level.shape[1] // low_level.shape[1]), interpolation='bilinear')(x)
    
    # Project low-level features
    low = Conv2D(48, 1, padding='same', use_bias=False)(low_level)
    low = BatchNormalization()(low)
    low = Activation('relu')(low)
    
    # Concatenate and refine
    x = Concatenate()([x, low])
    x = Conv2D(256, 3, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x); x = Activation('relu')(x)
    x = Dropout(0.5)(x) # Added dropout for regularization
    x = Conv2D(256, 3, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x); x = Activation('relu')(x)
    
    # Final output layer
    x = Conv2D(num_classes, 1, padding='same')(x)
    x = UpSampling2D(size=(4, 4), interpolation='bilinear')(x)
    
    return Model(inputs, x)

# Custom Loss Function: Combination of Dice and Categorical Cross-Entropy 
def dice_loss(y_true, y_pred):
    y_true = tf.one_hot(tf.cast(y_true, tf.int32), num_classes)
    y_pred = tf.nn.softmax(y_pred, axis=-1)
    numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=-1)
    denominator = tf.reduce_sum(y_true + y_pred, axis=-1)
    return 1 - (numerator + 1e-6) / (denominator + 1e-6)

def combined_loss(y_true, y_pred):
    return dice_loss(y_true, y_pred) + tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)

# Metric: Mean Intersection over Union (IoU) 
def mean_iou(y_true, y_pred):
    y_pred = tf.argmax(y_pred, axis=-1)
    return tf.keras.metrics.MeanIoU(num_classes=num_classes)(y_true, y_pred)

# Instantiate the model
num_classes = int(np.max(train_y) + 1)
model = build_custom_deeplab(input_shape=(*IMG_SIZE, 3), num_classes=num_classes)
model.summary()

# 5) Two-phase training process
# Phase 1: Train the decoder and newly added layers
for layer in model.get_layer('efficientnetv2-s').layers:
    layer.trainable = False

# Using AdamW optimizer as recommended for better weight decay 
model.compile(
    optimizer=tfa.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-5),
    loss=combined_loss,
    metrics=[mean_iou]
)

cbs1 = [
    EarlyStopping(monitor='val_mean_iou', mode='max', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_mean_iou', mode='max', factor=0.5, patience=5, min_lr=1e-6),
    ModelCheckpoint('phase1_custom_model.h5', monitor='val_mean_iou', mode='max', save_best_only=True)
]

print("\n--- Starting Phase 1: Training Decoder ---")
model.fit(
    train_X_processed, train_y,
    validation_data=(val_X_processed, val_y),
    batch_size=16, epochs=50,
    callbacks=cbs1
)

# Phase 2: Fine-tune the entire model
for layer in model.get_layer('efficientnetv2-s').layers:
    layer.trainable = True

model.compile(
    optimizer=tfa.optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-6), # Lower learning rate for fine-tuning
    loss=combined_loss,
    metrics=[mean_iou]
)

cbs2 = [
    EarlyStopping(monitor='val_mean_iou', mode='max', patience=15, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_mean_iou', mode='max', factor=0.2, patience=5, min_lr=1e-7),
    ModelCheckpoint('weed_detector_final_model.h5', monitor='val_mean_iou', mode='max', save_best_only=True)
]

print("\n--- Starting Phase 2: Fine-tuning Full Model ---")
model.fit(
    train_X_processed, train_y,
    validation_data=(val_X_processed, val_y),
    batch_size=8, # Smaller batch size for fine-tuning
    epochs=50,
    callbacks=cbs2
)

print("\nTraining complete. The final model is saved as 'weed_detector_final_model.h5'")
# You can load the model later with:
# model = tf.keras.models.load_model('weed_detector_final_model.h5', custom_objects={'combined_loss': combined_loss, 'mean_iou': mean_iou})