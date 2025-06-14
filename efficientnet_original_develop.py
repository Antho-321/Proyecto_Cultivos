import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, UpSampling2D, BatchNormalization, Activation,
    Concatenate, GlobalAveragePooling2D, Lambda, Dropout, LayerNormalization, Dense,
    Reshape
)
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from concurrent.futures import ThreadPoolExecutor, as_completed
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tqdm import tqdm
from tensorflow.keras import layers
import math

# --- Data Loading and Preprocessing (Unchanged) ---

# Data augmentation layer
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(height_factor=0.2, width_factor=0.2),
    layers.RandomContrast(0.2),
], name="data_augmentation")

# Optimized data loading function
def load_dataset(image_directory, mask_directory, target_size=(256, 256)):
    images, masks = [], []
    files = [f for f in os.listdir(image_directory) if f.endswith(('.jpg', '.png'))]
    with ThreadPoolExecutor() as exe:
        futures = {exe.submit(lambda fn: (img_to_array(load_img(os.path.join(image_directory, fn), target_size=target_size)), img_to_array(load_img(os.path.join(mask_directory, fn.rsplit('.', 1)[0] + '_mask.png'), color_mode='grayscale', target_size=target_size))), fn): fn for fn in files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Loading data"):
            img_arr, mask_arr = future.result()
            images.append(img_arr)
            masks.append(mask_arr)
    X = np.array(images, dtype='float32')
    y = np.squeeze(np.array(masks, dtype='int32'), axis=-1)
    return X, y

os.makedirs('Balanced/train/images', exist_ok=True)
os.makedirs('Balanced/train/masks', exist_ok=True)
os.makedirs('Balanced/val/images', exist_ok=True)
os.makedirs('Balanced/val/masks', exist_ok=True)
train_images_dir = 'Balanced/train/images'
train_masks_dir = 'Balanced/train/masks'
val_images_dir = 'Balanced/val/images'
val_masks_dir = 'Balanced/val/masks'
IMG_SIZE = (256, 256)

if not os.listdir(train_images_dir):
    print("Creating dummy data for demonstration...")
    dummy_img = np.random.rand(256, 256, 3) * 255
    dummy_mask = np.random.randint(0, 6, (256, 256, 1), dtype=np.uint8)
    tf.keras.preprocessing.image.save_img(os.path.join(train_images_dir, 'dummy_train.png'), dummy_img)
    tf.keras.preprocessing.image.save_img(os.path.join(train_masks_dir, 'dummy_train_mask.png'), dummy_mask)
    tf.keras.preprocessing.image.save_img(os.path.join(val_images_dir, 'dummy_val.png'), dummy_img)
    tf.keras.preprocessing.image.save_img(os.path.join(val_masks_dir, 'dummy_val_mask.png'), dummy_mask)

train_X, train_y = load_dataset(train_images_dir, train_masks_dir, target_size=IMG_SIZE)
val_X, val_y = load_dataset(val_images_dir, val_masks_dir, target_size=IMG_SIZE)

def preprocess_fn(images):
    return tf.keras.applications.efficientnet_v2.preprocess_input(images)

train_X_processed = preprocess_fn(train_X)
val_X_processed = preprocess_fn(val_X)

# --- Model Architecture ---

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

class Mlp(tf.keras.layers.Layer):
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
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., **kwargs):
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

        num_relative_positions = (2 * window_size[0] - 1) * (2 * window_size[1] - 1)
        self.relative_position_bias_table = self.add_weight(shape=(num_relative_positions, num_heads), initializer=tf.initializers.TruncatedNormal(stddev=0.02), trainable=True, name="relative_position_bias_table")
        coords_h = tf.range(self.window_size[0])
        coords_w = tf.range(self.window_size[1])
        coords = tf.stack(tf.meshgrid(coords_h, coords_w, indexing='ij'))
        coords_flatten = tf.reshape(coords, [2, -1])
        relative_coords = coords_flatten[:, :, tf.newaxis] - coords_flatten[:, tf.newaxis, :]
        relative_coords = tf.transpose(relative_coords, (1, 2, 0))
        relative_coords = relative_coords + [self.window_size[0] - 1, self.window_size[1] - 1]
        relative_coords = relative_coords * [(2 * self.window_size[1] - 1), 1]
        relative_position_index = tf.reduce_sum(relative_coords, axis=-1)
        self.relative_position_index = tf.Variable(relative_position_index, trainable=False, name="relative_position_index", dtype=tf.int32)

    def call(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x)
        qkv = tf.reshape(qkv, (-1, N, 3, self.num_heads, C // self.num_heads))
        qkv = tf.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ tf.transpose(k, (0, 1, 3, 2)))
        relative_position_bias = tf.gather(self.relative_position_bias_table, tf.reshape(self.relative_position_index, [-1]))
        relative_position_bias = tf.reshape(relative_position_bias, (self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1))
        relative_position_bias = tf.transpose(relative_position_bias, (2, 0, 1))
        attn = attn + tf.expand_dims(relative_position_bias, axis=0)
        if mask is not None:
            nW = tf.shape(mask)[0]
            attn = tf.reshape(attn, (-1, nW, self.num_heads, N, N))
            attn = attn + tf.cast(tf.expand_dims(tf.expand_dims(mask, axis=1), axis=0), attn.dtype)
            attn = tf.reshape(attn, (-1, self.num_heads, N, N))
            attn = tf.nn.softmax(attn, axis=-1)
        else:
            attn = tf.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)
        x = tf.transpose((attn @ v), (0, 2, 1, 3))
        x = tf.reshape(x, (-1, N, C))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinTransformerBlock(tf.keras.layers.Layer):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0., norm_layer=LayerNormalization, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = (window_size, window_size) if isinstance(window_size, int) else window_size
        self.shift_size = (shift_size, shift_size) if isinstance(shift_size, int) else shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= min(self.window_size):
            self.shift_size = (0, 0)
            self.window_size = self.input_resolution
        self.norm1 = norm_layer()
        self.attn = WindowAttention(dim, window_size=self.window_size, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = tf.keras.layers.Dropout(drop_path, noise_shape=(None, 1, 1)) if drop_path > 0. else tf.identity
        self.norm2 = norm_layer()
        self.mlp = Mlp(hidden_features=int(dim * mlp_ratio), out_features=dim, drop=drop)
        if any(s > 0 for s in self.shift_size):
            H, W = self.input_resolution
            img_mask = np.zeros((1, H, W, 1))
            h_slices = (slice(0, H - self.window_size[0]), slice(H - self.window_size[0], H - self.shift_size[0]), slice(H - self.shift_size[0], H))
            w_slices = (slice(0, W - self.window_size[1]), slice(W - self.window_size[1], W - self.shift_size[1]), slice(W - self.shift_size[1], W))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    if h.start < h.stop and w.start < w.stop:
                        img_mask[:, h, w, :] = cnt
                        cnt += 1
            mask_windows = Lambda(lambda t: tf.reshape(tf.transpose(tf.reshape(t, (-1, H // self.window_size[0], self.window_size[0], W // self.window_size[1], self.window_size[1], 1)), (0, 1, 3, 2, 4, 5)), (-1, self.window_size[0], self.window_size[1], 1)))(tf.constant(img_mask, dtype=tf.float32))
            mask_windows = tf.reshape(mask_windows, [-1, self.window_size[0] * self.window_size[1]])
            attn_mask = tf.expand_dims(mask_windows, axis=1) - tf.expand_dims(mask_windows, axis=2)
            attn_mask = tf.where(attn_mask != 0, -100.0, 0.0)
            self.attn_mask = tf.Variable(attn_mask, trainable=False, dtype=tf.float32, name="attn_mask")
        else:
            self.attn_mask = None

    def call(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        shortcut = x
        x_norm = self.norm1(x)
        x_reshaped = Reshape((H, W, C))(x_norm)
        if any(s > 0 for s in self.shift_size):
            shifted_x = Lambda(lambda t: tf.roll(t, shift=[-s for s in self.shift_size], axis=[1, 2]))(x_reshaped)
        else:
            shifted_x = x_reshaped
        x_windows = Lambda(lambda t: tf.reshape(tf.transpose(tf.reshape(t, (-1, H // self.window_size[0], self.window_size[0], W // self.window_size[1], self.window_size[1], C)), (0, 1, 3, 2, 4, 5)), (-1, self.window_size[0], self.window_size[1], C)))(shifted_x)
        x_windows = Reshape((self.window_size[0] * self.window_size[1], C))(x_windows)
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        attn_windows = Reshape((self.window_size[0], self.window_size[1], C))(attn_windows)
        shifted_x = Lambda(lambda t: tf.reshape(tf.transpose(tf.reshape(t, (-1, H // self.window_size[0], W // self.window_size[1], self.window_size[0], self.window_size[1], C)), (0, 1, 3, 2, 4, 5)), (-1, H, W, C)))(attn_windows)
        if any(s > 0 for s in self.shift_size):
            x_rev_shift = Lambda(lambda t: tf.roll(t, shift=list(self.shift_size), axis=[1, 2]))(shifted_x)
        else:
            x_rev_shift = shifted_x
        x_unrolled = Reshape((L, C))(x_rev_shift)
        x = shortcut + self.drop_path(x_unrolled)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

def build_custom_deeplab_with_swin(input_shape=(256, 256, 3), num_classes=6, swin_dim=128, swin_num_heads=4, swin_window_size=8, num_swin_blocks=2):
    inputs = Input(shape=input_shape)
    augmented_inputs = data_augmentation(inputs)
    backbone = EfficientNetV2S(include_top=False, weights='imagenet', input_tensor=augmented_inputs)
    low_level_features = backbone.get_layer('block3a_expand_activation').output
    high_level_features = backbone.output
    
    B, H_feat, W_feat, C_feat = high_level_features.shape
    x_swin_input = Reshape((H_feat * W_feat, C_feat))(high_level_features)

    if C_feat != swin_dim:
        x_swin_input = Dense(swin_dim, name="swin_input_projection")(x_swin_input)
        C_feat = swin_dim

    swin_resolution = (H_feat, W_feat)
    
    for i in range(num_swin_blocks):
        x_swin_input = SwinTransformerBlock(
            dim=C_feat, input_resolution=swin_resolution, num_heads=swin_num_heads, window_size=swin_window_size,
            shift_size=0 if (i % 2 == 0) else swin_window_size // 2,
            mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0.1, attn_drop=0.1, drop_path=0.1,
            name=f'swin_block_{i}'
        )(x_swin_input)

    x_after_swin = Reshape((H_feat, W_feat, C_feat))(x_swin_input)
    
    x = ASPP(x_after_swin, out_channels=256, rates=[6, 12, 18])
    x = UpSampling2D(size=(low_level_features.shape[1] // x.shape[1], low_level_features.shape[2] // x.shape[2]), interpolation='bilinear')(x)
    
    low = Conv2D(48, 1, padding='same', use_bias=False)(low_level_features)
    low = BatchNormalization()(low)
    low = Activation('relu')(low)
    
    x = Concatenate()([x, low])
    x = Conv2D(256, 3, padding='same', use_bias=False)(x); x = BatchNormalization()(x); x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    x = Conv2D(256, 3, padding='same', use_bias=False)(x); x = BatchNormalization()(x); x = Activation('relu')(x)
    x = Conv2D(num_classes, 1, padding='same')(x)
    
    # --- CORRECTED FINAL UPSAMPLING ---
    # Dynamically calculate the upsampling factor to match the original input size
    upsample_factor_h = input_shape[0] // x.shape[1]
    upsample_factor_w = input_shape[1] // x.shape[2]
    x = UpSampling2D(size=(upsample_factor_h, upsample_factor_w), interpolation='bilinear')(x)
    
    final_model = Model(inputs, x, name="DeepLabV3_Plus_Swin")
    return final_model, backbone

# --- Training ---
num_classes = int(np.max(train_y) + 1)
if num_classes <= 1: num_classes = 6

def dice_loss(y_true, y_pred):
    y_true = tf.one_hot(tf.cast(y_true, tf.int32), num_classes)
    y_pred = tf.nn.softmax(y_pred, axis=-1)
    numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=-1)
    denominator = tf.reduce_sum(y_true + y_pred, axis=-1)
    return 1 - (numerator + 1e-6) / (denominator + 1e-6)

def combined_loss(y_true, y_pred):
    ce_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
    d_loss = dice_loss(y_true, y_pred)
    return ce_loss + d_loss

class MeanIoUFromLogits(tf.keras.metrics.Metric):
    def __init__(self, num_classes, name='mean_iou', dtype=None):
        super().__init__(name=name, dtype=dtype)
        self.num_classes = num_classes
        self.mean_iou = tf.keras.metrics.MeanIoU(num_classes=self.num_classes)
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_labels = tf.argmax(y_pred, axis=-1)
        y_true_labels = tf.cast(y_true, y_pred_labels.dtype)
        self.mean_iou.update_state(y_true_labels, y_pred_labels, sample_weight)
    def result(self):
        return self.mean_iou.result()
    def reset_state(self, *args, **kwargs):
        self.mean_iou.reset_state()
    def get_config(self):
        config = super().get_config(); config.update({'num_classes': self.num_classes}); return config

iou_metric = MeanIoUFromLogits(num_classes=num_classes)
model, backbone_layer = build_custom_deeplab_with_swin(input_shape=(*IMG_SIZE, 3), num_classes=num_classes)
model.summary()

print("\nFreezing the backbone for Phase 1...")
backbone_layer.trainable = False
model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-5), loss=combined_loss, metrics=[iou_metric])
cbs1 = [EarlyStopping(monitor='val_mean_iou', mode='max', patience=10, restore_best_weights=True), ReduceLROnPlateau(monitor='val_mean_iou', mode='max', factor=0.5, patience=5, min_lr=1e-6), ModelCheckpoint('phase1_custom_model_swin.keras', monitor='val_mean_iou', mode='max', save_best_only=True)]
print("\n--- Starting Phase 1: Training the Decoder ---")
model.fit(train_X_processed, train_y, validation_data=(val_X_processed, val_y), batch_size=4, epochs=20, callbacks=cbs1)

print("\nUnfreezing the backbone for Phase 2...")
backbone_layer.trainable = True
model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-6), loss=combined_loss, metrics=[iou_metric])
cbs2 = [EarlyStopping(monitor='val_mean_iou', mode='max', patience=15, restore_best_weights=True), ReduceLROnPlateau(monitor='val_mean_iou', mode='max', factor=0.2, patience=5, min_lr=1e-7), ModelCheckpoint('weed_detector_final_model_swin.keras', monitor='val_mean_iou', mode='max', save_best_only=True)]
print("\n--- Starting Phase 2: Fine-Tuning the Full Model ---")
model.fit(train_X_processed, train_y, validation_data=(val_X_processed, val_y), batch_size=2, epochs=20, callbacks=cbs2)

print("\nTraining complete.")
