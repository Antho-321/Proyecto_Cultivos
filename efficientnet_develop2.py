# -*- coding: utf-8 -*-
"""
enhanced_iou_segmentation_model.py

Enhanced script with multiple improvements to maximize mean IoU:
1. Multi-scale training and testing
2. Improved loss functions with focal components
3. Advanced data augmentation with MixUp
4. Progressive image resolution training
5. Enhanced architecture with attention mechanisms
6. Class-balanced sampling
7. Test-time augmentation (TTA)
"""

# 1) Imports ------------------------------------------------------------------
import os
os.environ['MPLBACKEND'] = 'agg'

import tensorflow as tf

# GPU Configuration
print("TensorFlow Version:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Available GPUs: {[gpu.name for gpu in gpus]}")
        device = '/GPU:0'
    except RuntimeError as e:
        print(e)
        device = '/CPU:0'
else:
    print("No GPU found. Training will run on CPU.")
    device = '/CPU:0'

from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import cv2
from tqdm import tqdm
import albumentations as A
from tensorflow import keras
from tensorflow.keras import layers, backend as K
from tensorflow.keras.layers import (
    Input, Conv2D, SeparableConv2D, Conv2DTranspose,
    BatchNormalization, Activation, Dropout,
    Lambda, Concatenate, UpSampling2D, ReLU, Add, GlobalAveragePooling2D,
    Reshape, Dense, Multiply, GlobalMaxPooling2D, MultiHeadAttention,
    LayerNormalization, DepthwiseConv2D
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
)
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight

# Monkey-patch for SE module
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

# 2) Enhanced Augmentation with MixUp ----------------------------------------
def get_enhanced_training_augmentation():
    """Enhanced augmentation pipeline with more aggressive transformations"""
    return A.Compose([
        # Geometric transformations
        A.RandomScale(scale_limit=0.3, p=0.7),
        A.PadIfNeeded(min_height=256, min_width=256, border_mode=cv2.BORDER_REFLECT),
        A.RandomCrop(height=256, width=256),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.4),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5),
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
        
        # Color augmentations
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.RandomGamma(gamma_limit=(70, 130), p=0.3),
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=20, val_shift_limit=15, p=0.4),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
        
        # Noise and blur
        A.OneOf([
            A.GaussianBlur(blur_limit=5, p=1.0),
            A.MotionBlur(blur_limit=5, p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
        ], p=0.2),
        A.OneOf([
            A.GaussNoise(var_limit=(10, 50), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
        ], p=0.2),
        
        # Dropout augmentations
        A.CoarseDropout(max_holes=12, max_height=40, max_width=40, 
                        min_holes=1, min_height=8, min_width=8, p=0.3),
        A.GridDropout(ratio=0.3, p=0.2),
    ])

def mixup_data(x, y, alpha=0.4):
    """MixUp augmentation for better generalization"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.shape[0]
    index = np.random.permutation(batch_size)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    mixed_y = lam * y + (1 - lam) * y[index]
    
    return mixed_x, mixed_y

# 3) Multi-scale Data Loading ------------------------------------------------
def load_multiscale_dataset(img_dir, mask_dir, target_sizes=[(224,224), (256,256), (320,320)], augment=False):
    """Load dataset with multiple scales for progressive training"""
    all_data = {}
    
    for size in target_sizes:
        images, masks = [], []
        files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg','.png'))]
        aug = get_enhanced_training_augmentation() if augment else A.Compose([])
        
        with ThreadPoolExecutor() as exe:
            futures = {exe.submit(lambda fn: (
                img_to_array(load_img(os.path.join(img_dir,fn), target_size=size)),
                img_to_array(load_img(os.path.join(mask_dir,fn.rsplit('.',1)[0] + '_mask.png'),
                                      color_mode='grayscale', target_size=size))
            ), fn): fn for fn in files}
            
            for future in tqdm(as_completed(futures), total=len(futures), 
                               desc=f"Loading {size} {'train' if augment else 'val'} data"):
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
            
        all_data[size] = (X, y)
    
    return all_data

# 4) Enhanced Architecture with Self-Attention ------------------------------
def self_attention_block(x, num_heads=8, key_dim=64):
    """Self-attention mechanism for capturing long-range dependencies"""
    input_shape = tf.shape(x)
    batch_size, height, width, channels = input_shape[0], input_shape[1], input_shape[2], x.shape[-1]
    
    # Reshape for attention
    x_reshaped = tf.reshape(x, [batch_size, height * width, channels])
    
    # Multi-head self-attention
    attention_output = MultiHeadAttention(
        num_heads=num_heads, 
        key_dim=key_dim,
        dropout=0.1
    )(x_reshaped, x_reshaped)
    
    # Add residual connection and layer norm
    x_reshaped = LayerNormalization()(x_reshaped + attention_output)
    
    # Reshape back
    output = tf.reshape(x_reshaped, [batch_size, height, width, channels])
    
    return output

def enhanced_conv_block(x, filters, kernel_size, strides=1, padding='same', 
                        dilation_rate=1, use_attention=False, dropout_rate=0.1):
    """Enhanced convolutional block with optional attention and dropout"""
    # Depthwise separable convolution for efficiency
    x = SeparableConv2D(
        filters, 
        kernel_size, 
        strides=strides, 
        padding=padding, 
        dilation_rate=dilation_rate, 
        use_bias=False
    )(x)
    x = BatchNormalization()(x)
    x = Activation('swish')(x)
    
    if dropout_rate > 0:
        x = Dropout(dropout_rate)(x)
    
    if use_attention and filters >= 64:  # Only use attention for higher-dim features
        x = self_attention_block(x, num_heads=min(8, filters//8))
    
    return x

def advanced_ASPP(x, dilation_rates=[6, 12, 18, 24], use_attention=True):
    """Advanced ASPP with more dilation rates and attention"""
    input_shape = x.shape
    input_filters = input_shape[-1]
    
    # Global Average Pooling branch
    image_pooling = GlobalAveragePooling2D()(x)
    image_pooling = Reshape((1, 1, input_filters))(image_pooling)
    image_pooling = Conv2D(input_filters, 1, padding='same', use_bias=False)(image_pooling)
    image_pooling = BatchNormalization()(image_pooling)
    image_pooling = Activation('swish')(image_pooling)
    image_pooling = UpSampling2D(size=(input_shape[1], input_shape[2]), interpolation='bilinear')(image_pooling)

    # 1x1 convolution branch
    conv_1x1 = enhanced_conv_block(x, input_filters, 1)

    # Atrous convolution branches with different dilation rates
    atrous_convs = []
    for rate in dilation_rates:
        atrous_conv = enhanced_conv_block(x, input_filters, 3, dilation_rate=rate)
        atrous_convs.append(atrous_conv)

    # Concatenate all branches
    concatenated = Concatenate()([image_pooling, conv_1x1] + atrous_convs)
    
    # Projection with attention
    projected = enhanced_conv_block(concatenated, input_filters, 1, use_attention=use_attention)

    return projected

def build_enhanced_model(shape=(256, 256, 3), num_classes_arg=None):
    """Enhanced model architecture with better feature extraction and fusion"""
    # Backbone
    backbone = EfficientNetV2S(
        input_shape=shape,
        num_classes=0,
        pretrained='imagenet',
        include_preprocessing=False
    )

    # Skip connections extraction
    inp = backbone.input
    bottleneck = backbone.get_layer('post_swish').output

    # Define skip connection layer names based on the EfficientNetV2S architecture
    s1_name = 'add_1'  # Output shape: (128, 128, 32)
    s2_name = 'add_3'  # Output shape: (64, 64, 48)
    s3_name = 'add_7'  # Output shape: (32, 32, 64)
    s4_name = 'add_17' # Output shape: (16, 16, 160)

    s1 = backbone.get_layer(s1_name).output
    s2 = backbone.get_layer(s2_name).output
    s3 = backbone.get_layer(s3_name).output
    s4 = backbone.get_layer(s4_name).output
    
    # Enhanced bottleneck with advanced ASPP
    x = advanced_ASPP(bottleneck, dilation_rates=[6, 12, 18], use_attention=True)

    # Enhanced decoder with attention and better feature fusion
    # Block 1: 8x8 -> 16x16
    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    s4_aligned = enhanced_conv_block(s4, x.shape[-1], 1)
    x = Concatenate()([x, s4_aligned])
    x = enhanced_conv_block(x, 256, 3, use_attention=True)
    x = enhanced_conv_block(x, 256, 3)

    # Block 2: 16x16 -> 32x32
    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    s3_aligned = enhanced_conv_block(s3, x.shape[-1], 1)
    x = Concatenate()([x, s3_aligned])
    x = enhanced_conv_block(x, 128, 3, use_attention=True)
    x = enhanced_conv_block(x, 128, 3)

    # Block 3: 32x32 -> 64x64
    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    s2_aligned = enhanced_conv_block(s2, x.shape[-1], 1)
    x = Concatenate()([x, s2_aligned])
    x = enhanced_conv_block(x, 96, 3, use_attention=False) # Less attention in early layers
    x = enhanced_conv_block(x, 96, 3)

    # Block 4: 64x64 -> 128x128
    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    s1_aligned = enhanced_conv_block(s1, x.shape[-1], 1)
    x = Concatenate()([x, s1_aligned])
    x = enhanced_conv_block(x, 64, 3, use_attention=False)
    x = enhanced_conv_block(x, 64, 3)

    # Final upsampling
    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    x = enhanced_conv_block(x, 32, 3)
    
    # Output prediction
    main_out = Conv2D(num_classes_arg, 1, padding='same', activation='softmax', 
                      dtype='float32', name='main_output')(x)
    
    return Model(inp, main_out), backbone

# 5) Advanced Loss Functions -------------------------------------------------
def focal_tversky_loss(y_true, y_pred, alpha=0.4, beta=0.6, gamma=2.5, smooth=1e-7):
    """Focal Tversky Loss for handling class imbalance with hard examples focus"""
    y_true_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), depth=num_classes, axis=-1)
    y_pred_probs = tf.nn.softmax(y_pred, axis=-1)

    y_true_flat = tf.reshape(y_true_one_hot, [-1, num_classes])
    y_pred_flat = tf.reshape(y_pred_probs, [-1, num_classes])
    
    tp = tf.reduce_sum(y_true_flat * y_pred_flat, axis=0)
    fn = tf.reduce_sum(y_true_flat * (1 - y_pred_flat), axis=0)
    fp = tf.reduce_sum((1 - y_true_flat) * y_pred_flat, axis=0)
    
    tversky_index = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    focal_tversky = tf.pow((1 - tversky_index), 1/gamma)
    
    return tf.reduce_mean(focal_tversky)

def edge_enhanced_loss(y_true, y_pred, edge_weight=2.0):
    """Enhanced edge loss using Sobel filters for better boundary detection"""
    y_true_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), depth=num_classes, axis=-1)
    y_pred_probs = tf.nn.softmax(y_pred, axis=-1)
    
    sobel_x = tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=tf.float32)
    sobel_y = tf.constant([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=tf.float32)
    sobel_x = tf.reshape(sobel_x, [3, 3, 1, 1])
    sobel_y = tf.reshape(sobel_y, [3, 3, 1, 1])
    sobel_x = tf.tile(sobel_x, [1, 1, 1, num_classes])
    sobel_y = tf.tile(sobel_y, [1, 1, 1, num_classes])
    
    true_edge_x = tf.nn.depthwise_conv2d(y_true_one_hot, sobel_x, strides=[1,1,1,1], padding='SAME')
    true_edge_y = tf.nn.depthwise_conv2d(y_true_one_hot, sobel_y, strides=[1,1,1,1], padding='SAME')
    true_edge = tf.sqrt(true_edge_x**2 + true_edge_y**2)
    
    pred_edge_x = tf.nn.depthwise_conv2d(y_pred_probs, sobel_x, strides=[1,1,1,1], padding='SAME')
    pred_edge_y = tf.nn.depthwise_conv2d(y_pred_probs, sobel_y, strides=[1,1,1,1], padding='SAME')
    pred_edge = tf.sqrt(pred_edge_x**2 + pred_edge_y**2)
    
    edge_diff = tf.abs(true_edge - pred_edge)
    return edge_weight * tf.reduce_mean(edge_diff)

def ultimate_enhanced_loss(y_true, y_pred, class_weights=None):
    """Ultimate loss combining multiple loss functions with class weighting"""
    lovasz = 0.4 * lovasz_softmax(y_pred, tf.cast(y_true, tf.int32), per_image=True)
    focal = 0.4 * focal_tversky_loss(y_true, y_pred)
    edge = 0.2 * edge_enhanced_loss(y_true, y_pred)
    
    total_loss = lovasz + focal + edge

    if class_weights is not None:
        y_true_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), depth=num_classes, axis=-1)
        weights = tf.constant(class_weights, dtype=tf.float32)
        weight_map = tf.multiply(y_true_one_hot, weights)
        weight_map = tf.reduce_sum(weight_map, axis=-1)
        
        ce_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
        weighted_ce = tf.reduce_mean(ce_loss * weight_map)
        total_loss += 0.1 * weighted_ce

    return total_loss

# 6) Enhanced Metrics ---------------------------------------------------------
class EnhancedMeanIoU(tf.keras.metrics.Metric):
    def __init__(self, num_classes, name='enhanced_mean_iou', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.total_cm = self.add_weight(
            shape=(num_classes, num_classes),
            name='total_confusion_matrix',
            initializer='zeros'
        )
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        preds = tf.argmax(y_pred, axis=-1)
        if len(y_true.shape) == 4 and y_true.shape[-1] == 1:
            y_true = tf.squeeze(y_true, axis=-1)
            
        y_t = tf.reshape(tf.cast(y_true, tf.int32), [-1])
        y_p = tf.reshape(tf.cast(preds, tf.int32), [-1])
        
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

# 7) Enhanced Model Class -----------------------------------------------------
class EnhancedIoUModel(keras.Model):
    def __init__(self, shape, num_classes, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.class_weights = class_weights
        self.seg_model, self.backbone = build_enhanced_model(shape, num_classes_arg=num_classes)
        
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.iou_metric = EnhancedMeanIoU(num_classes=num_classes, name='enhanced_mean_iou')

    def call(self, inputs, training=False):
        return self.seg_model(inputs, training=training)

    @property
    def metrics(self):
        return [self.loss_tracker, self.iou_metric]

    def train_step(self, data):
        if len(data) == 3:
             x, y_true, _ = data
        else:
             x, y_true = data

        with tf.GradientTape() as tape:
            y_pred = self.seg_model(x, training=True)
            loss = ultimate_enhanced_loss(y_true, y_pred, self.class_weights)

        trainable_vars = self.seg_model.trainable_variables
        grads = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(grads, trainable_vars))

        self.loss_tracker.update_state(loss)
        self.iou_metric.update_state(y_true, y_pred)
        
        return {m.name: m.result() for m in self.metrics}
        
    def test_step(self, data):
        x, y_true = data
        y_pred = self.seg_model(x, training=False)
        
        loss = ultimate_enhanced_loss(y_true, y_pred, self.class_weights)
        
        self.loss_tracker.update_state(loss)
        self.iou_metric.update_state(y_true, y_pred)
        
        return {m.name: m.result() for m in self.metrics}

# 8) Test Time Augmentation --------------------------------------------------
def predict_with_tta(model, x, num_tta=8):
    """Test Time Augmentation for better inference results"""
    predictions = []
    
    # Original prediction
    pred = model.predict(x, verbose=0)
    predictions.append(pred)
    
    # Augmented predictions
    for _ in range(num_tta - 1):
        aug = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.5),
        ])
        
        x_aug = np.array([aug(image=(img * 255).astype(np.uint8))['image'] for img in x])
        x_aug = x_aug.astype(np.float32) / 255.0
        
        pred_aug = model.predict(x_aug, verbose=0)
        predictions.append(pred_aug)
    
    # Average predictions
    return np.mean(predictions, axis=0)

# 9) Main Training Pipeline --------------------------------------------------
def main():
    # --- Configuración Inicial ---
    resolutions = [(224, 224), (256, 256), (320, 320)]
    batch_sizes = { (224,224): 10, (256,256): 8, (320,320): 4 }
    
    # Load multi-scale data
    print("⏳ Loading multi-scale datasets...")
    train_data = load_multiscale_dataset('Balanced/train/images', 'Balanced/train/masks', 
                                         target_sizes=resolutions, augment=True)
    val_data = load_multiscale_dataset('Balanced/val/images', 'Balanced/val/masks', 
                                       target_sizes=resolutions, augment=False)
    
    # Compute class weights
    global num_classes
    _, y_sample = train_data[resolutions[0]]
    num_classes = int(np.max(y_sample) + 1)
    print(f"✅ Number of classes detected: {num_classes}")
    
    class_weights_list = compute_class_weight('balanced', classes=np.arange(num_classes), 
                                          y=y_sample.flatten())
    print(f"✅ Class weights: {class_weights_list}")
    
    checkpoint_dir = './enhanced_checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    MONITOR_METRIC = 'val_enhanced_mean_iou'
    
    with tf.device(device):
        
        # --- Fase 1: Entrenar el decodificador ---
        print("\n--- Phase 1: Training decoder (frozen backbone) at 224x224 ---")
        current_size = resolutions[0]
        train_X, train_y = train_data[current_size]
        val_X, val_y = val_data[current_size]
        
        enhanced_model = EnhancedIoUModel(
            shape=train_X.shape[1:], 
            num_classes=num_classes,
            class_weights=class_weights_list
        )
        enhanced_model.backbone.trainable = False
        enhanced_model.compile(
            optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4)
        )

        callbacks_phase1 = [
            ModelCheckpoint(
                os.path.join(checkpoint_dir, 'enhanced_phase1.keras'),
                save_best_only=True, monitor=MONITOR_METRIC, mode='max', verbose=1
            ),
            EarlyStopping(monitor=MONITOR_METRIC, mode='max', patience=12, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1),
            TensorBoard(log_dir='./logs/enhanced_phase1', update_freq='epoch')
        ]
        
        train_X_mixed, train_y_mixed = mixup_data(train_X, train_y, alpha=0.2)
        
        enhanced_model.fit(
            train_X_mixed, train_y_mixed, 
            batch_size=batch_sizes[current_size], epochs=30,
            validation_data=(val_X, val_y), 
            callbacks=callbacks_phase1
        )
        
        # --- Fase 2: Fine-tuning parcial ---
        print("\n--- Phase 2: Partial backbone fine-tuning at 256x256 ---")
        current_size = resolutions[1]
        train_X, train_y = train_data[current_size]
        val_X, val_y = val_data[current_size]

        # Reconstruir modelo para nuevo tamaño y cargar pesos
        enhanced_model = EnhancedIoUModel(shape=train_X.shape[1:], num_classes=num_classes, class_weights=class_weights_list)
        enhanced_model.load_weights(os.path.join(checkpoint_dir, 'enhanced_phase1.keras')).expect_partial()
        
        enhanced_model.backbone.trainable = True
        fine_tune_at = int(len(enhanced_model.backbone.layers) * 0.6) # Unfreeze more layers
        for layer in enhanced_model.backbone.layers[:fine_tune_at]:
            layer.trainable = False

        enhanced_model.compile(
            optimizer=tf.keras.optimizers.AdamW(learning_rate=5e-5, weight_decay=5e-5)
        )
        
        callbacks_phase2 = [
            ModelCheckpoint(
                os.path.join(checkpoint_dir, 'enhanced_phase2.keras'),
                save_best_only=True, monitor=MONITOR_METRIC, mode='max', verbose=1
            ),
            EarlyStopping(monitor=MONITOR_METRIC, mode='max', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, min_lr=1e-7, verbose=1),
            TensorBoard(log_dir='./logs/enhanced_phase2')
        ]

        train_X_mixed, train_y_mixed = mixup_data(train_X, train_y, alpha=0.3)

        enhanced_model.fit(
            train_X_mixed, train_y_mixed,
            batch_size=batch_sizes[current_size], epochs=40,
            validation_data=(val_X, val_y),
            callbacks=callbacks_phase2
        )
        
        # --- Fase 3: Fine-tuning completo en alta resolución ---
        print("\n--- Phase 3: Full fine-tuning at 320x320 ---")
        current_size = resolutions[2]
        train_X, train_y = train_data[current_size]
        val_X, val_y = val_data[current_size]

        enhanced_model = EnhancedIoUModel(shape=train_X.shape[1:], num_classes=num_classes, class_weights=class_weights_list)
        enhanced_model.load_weights(os.path.join(checkpoint_dir, 'enhanced_phase2.keras')).expect_partial()

        enhanced_model.backbone.trainable = True # Unfreeze all layers
        enhanced_model.compile(
            optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-5, weight_decay=2e-5)
        )

        callbacks_phase3 = [
            ModelCheckpoint(
                os.path.join(checkpoint_dir, 'enhanced_final_best.keras'),
                save_best_only=True, monitor=MONITOR_METRIC, mode='max', verbose=1
            ),
            EarlyStopping(monitor=MONITOR_METRIC, mode='max', patience=20, restore_best_weights=True),
            TensorBoard(log_dir='./logs/enhanced_phase3')
        ]

        train_X_mixed, train_y_mixed = mixup_data(train_X, train_y, alpha=0.3)

        enhanced_model.fit(
            train_X_mixed, train_y_mixed,
            batch_size=batch_sizes[current_size], epochs=50,
            validation_data=(val_X, val_y),
            callbacks=callbacks_phase3
        )
        
        # --- Evaluación Final con TTA ---
        print("\n--- 🚀 Final Evaluation with Test-Time Augmentation (TTA) ---")
        best_model = EnhancedIoUModel(shape=val_X.shape[1:], num_classes=num_classes)
        best_model.load_weights(os.path.join(checkpoint_dir, 'enhanced_final_best.keras'))
        
        print("Evaluating on validation set without TTA...")
        results_no_tta = best_model.evaluate(val_X, val_y, batch_size=batch_sizes[current_size])
        print(f"Results without TTA -> Loss: {results_no_tta[0]:.4f}, Mean IoU: {results_no_tta[1]:.4f}")
        
        print("\nEvaluating on validation set with TTA (this may take a while)...")
        tta_predictions = predict_with_tta(best_model, val_X, num_tta=4)
        
        # Manually calculate IoU for TTA predictions
        tta_iou_metric = EnhancedMeanIoU(num_classes=num_classes)
        tta_iou_metric.update_state(val_y, tta_predictions)
        final_iou = tta_iou_metric.result().numpy()
        
        print(f"\n🏆 Final Enhanced Mean IoU with TTA: {final_iou:.4f}")

# Punto de entrada para la ejecución del script
if __name__ == '__main__':
    # Nota: Asegúrate de tener los directorios 'Balanced/train/images', 'Balanced/train/masks', etc.
    # con tus datos antes de ejecutar este script.
    main()