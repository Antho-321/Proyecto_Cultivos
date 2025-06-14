import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, UpSampling2D, BatchNormalization, Activation,
    Concatenate, GlobalAveragePooling2D, Lambda, Dropout, LayerNormalization, Dense
)
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from concurrent.futures import ThreadPoolExecutor, as_completed
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tqdm import tqdm
from tensorflow.keras import layers
import math # Needed for sqrt in potential qk_scale calculation

# Data augmentation layer
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(height_factor=0.2, width_factor=0.2),
    layers.RandomContrast(0.2),
], name="data_augmentation")

# Optimized data loading function
def load_dataset(image_directory, mask_directory, target_size=(256, 256)):
    """
    Loads images and masks from specified directories using a ThreadPoolExecutor for efficiency.
    Args:
        image_directory (str): Path to the directory containing images.
        mask_directory (str): Path to the directory containing masks.
        target_size (tuple): Desired (height, width) for resizing images and masks.
    Returns:
        tuple: (images_array, masks_array) as numpy arrays.
    """
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
    y = np.squeeze(np.array(masks, dtype='int32'), axis=-1) # Remove single-channel dimension from masks
    
    return X, y

# Dataset directories
# Using placeholder directories for demonstration if they don't exist
os.makedirs('Balanced/train/images', exist_ok=True)
os.makedirs('Balanced/train/masks', exist_ok=True)
os.makedirs('Balanced/val/images', exist_ok=True)
os.makedirs('Balanced/val/masks', exist_ok=True)

train_images_dir = 'Balanced/train/images'
train_masks_dir = 'Balanced/train/masks'
val_images_dir = 'Balanced/val/images'
val_masks_dir = 'Balanced/val/masks'

IMG_SIZE = (256, 256)

# Create dummy data if directories are empty, for a runnable example
if not os.listdir(train_images_dir):
    print("Creating dummy data for demonstration...")
    dummy_img = np.random.rand(256, 256, 3) * 255
    # Dummy mask with multiple classes (0-5)
    dummy_mask = np.random.randint(0, 6, (256, 256, 1), dtype=np.uint8) 
    tf.keras.preprocessing.image.save_img(os.path.join(train_images_dir, 'dummy_train.png'), dummy_img)
    tf.keras.preprocessing.image.save_img(os.path.join(train_masks_dir, 'dummy_train_mask.png'), dummy_mask)
    tf.keras.preprocessing.image.save_img(os.path.join(val_images_dir, 'dummy_val.png'), dummy_img)
    tf.keras.preprocessing.image.save_img(os.path.join(val_masks_dir, 'dummy_val_mask.png'), dummy_mask)

train_X, train_y = load_dataset(train_images_dir, train_masks_dir, target_size=IMG_SIZE)
val_X, val_y = load_dataset(val_images_dir, val_masks_dir, target_size=IMG_SIZE)

# Preprocessing function for EfficientNetV2
def preprocess_fn(images):
    """
    Applies EfficientNetV2 specific preprocessing to image batch.
    Args:
        images (tf.Tensor): Batch of image tensors.
    Returns:
        tf.Tensor: Preprocessed image batch.
    """
    return tf.keras.applications.efficientnet_v2.preprocess_input(images)

train_X_processed = preprocess_fn(train_X)
val_X_processed = preprocess_fn(val_X)

# ASPP Module (existing from original code)
def ASPP(x, out_channels=256, rates=(6, 12, 18)):
    """
    Atrous Spatial Pyramid Pooling (ASPP) module.
    Aggregates multi-scale contextual information.
    Args:
        x (tf.Tensor): Input feature map.
        out_channels (int): Number of output channels for convolutional layers.
        rates (tuple): Dilation rates for atrous convolutions.
    Returns:
        tf.Tensor: Output tensor from ASPP.
    """
    # 1x1 convolution branch
    convs = [layers.Conv2D(out_channels, 1, padding="same", use_bias=False)(x)]
    
    # Atrous convolution branches
    for r in rates:
        convs.append(layers.Conv2D(out_channels, 3, dilation_rate=r, padding="same", use_bias=False)(x))
    
    # Image-level pooling branch
    pool = layers.GlobalAveragePooling2D(keepdims=True)(x)
    pool = layers.Conv2D(out_channels, 1, padding="same", use_bias=False)(pool)
    # Resize pooled features back to input spatial dimensions
    pool = layers.Lambda(
        lambda t: tf.image.resize(t[0], tf.shape(t[1])[1:3], method="bilinear")
    )([pool, x])
    convs.append(pool)
    
    # Concatenate all branches and apply final 1x1 convolution
    y = layers.Concatenate()(convs)
    y = layers.Conv2D(out_channels, 1, padding="same", use_bias=False)(y)
    y = layers.BatchNormalization()(y)
    return layers.Activation("relu")(y)

# --- START SW-MSA Implementation ---

def window_partition(x, window_size):
    """
    Partitions the input tensor into non-overlapping windows.
    Args:
        x (tf.Tensor): Input tensor of shape (B, H, W, C).
        window_size (tuple): Size of the attention window (Wh, Ww).
    Returns:
        tf.Tensor: Tensor of shape (B * num_windows, Wh, Ww, C).
    """
    B, H, W, C = x.shape
    Wh, Ww = window_size
    # Reshape into (B, num_windows_h, Wh, num_windows_w, Ww, C)
    x = tf.reshape(x, (-1, H // Wh, Wh, W // Ww, Ww, C))
    # Permute to (B, num_windows_h, num_windows_w, Wh, Ww, C)
    x = tf.transpose(x, (0, 1, 3, 2, 4, 5))
    # Reshape to (B * num_windows, Wh, Ww, C)
    windows = tf.reshape(x, (-1, Wh, Ww, C))
    return windows

def window_reverse(windows, window_size, H, W, C):
    """
    Reverses window partition to reconstruct the feature map.
    Args:
        windows (tf.Tensor): Input tensor of shape (B * num_windows, Wh, Ww, C).
        window_size (tuple): Size of the attention window (Wh, Ww).
        H (int): Original height.
        W (int): Original width.
        C (int): Original channels.
    Returns:
        tf.Tensor: Tensor of shape (B, H, W, C).
    """
    Wh, Ww = window_size
    # Reshape into (B, num_windows_h, num_windows_w, Wh, Ww, C)
    x = tf.reshape(windows, (-1, H // Wh, W // Ww, Wh, Ww, C))
    # Permute back to (B, num_windows_h, Wh, num_windows_w, Ww, C)
    x = tf.transpose(x, (0, 1, 3, 2, 4, 5))
    # Reshape to (B, H, W, C)
    x = tf.reshape(x, (-1, H, W, C))
    return x

class Mlp(tf.keras.layers.Layer):
    """
    Multi-layer Perceptron (Feed-forward network) block used in Transformer.
    """
    def __init__(self, hidden_features=None, out_features=None, drop=0., **kwargs):
        super().__init__(**kwargs)
        self.fc1 = Dense(hidden_features)
        self.act = Activation('gelu') # Gaussian Error Linear Unit activation
        self.fc2 = Dense(out_features)
        self.drop = Dropout(drop)

    def call(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "hidden_features": self.fc1.units,
            "out_features": self.fc2.units,
            "drop": self.drop.rate,
        })
        return config

class WindowAttention(tf.keras.layers.Layer):
    """
    Window-based Multi-head Self-Attention (W-MSA) module.
    """
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.window_size = window_size  # Wh, Ww as tuple
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5 # Scale factor for attention

        self.qkv = Dense(dim * 3, use_bias=qkv_bias, name="qkv_proj") # Linear projection for Q, K, V
        self.attn_drop = Dropout(attn_drop)
        self.proj = Dense(dim, name="attn_proj") # Output projection
        self.proj_drop = Dropout(proj_drop)

        # Define a trainable parameter table for relative position bias
        # This table stores biases for all possible relative positions within a window
        num_relative_positions = (2 * window_size[0] - 1) * (2 * window_size[1] - 1)
        self.relative_position_bias_table = self.add_weight(
            shape=(num_relative_positions, num_heads),
            initializer=tf.initializers.TruncatedNormal(stddev=0.02),
            trainable=True,
            name="relative_position_bias_table"
        )

        # Pre-compute the relative position index for each token pair in a window
        coords_h = tf.range(self.window_size[0])
        coords_w = tf.range(self.window_size[1])
        # Create a grid of coordinates
        coords = tf.stack(tf.meshgrid(coords_h, coords_w, indexing='ij')) # (2, Wh, Ww)
        coords_flatten = tf.reshape(coords, [2, -1]) # (2, Wh*Ww)
        
        # Calculate relative coordinates (pos1 - pos2)
        relative_coords = coords_flatten[:, :, tf.newaxis] - coords_flatten[:, tf.newaxis, :] # (2, Wh*Ww, Wh*Ww)
        relative_coords = tf.transpose(relative_coords, (1, 2, 0)) # (Wh*Ww, Wh*Ww, 2)
        
        # Shift coordinates to be non-negative for indexing purposes
        # The range of relative coordinates is [-(W-1), W-1]
        relative_coords = relative_coords + [self.window_size[0] - 1, self.window_size[1] - 1]
        
        # Scale for unique indexing: (h * (2W-1) + w)
        relative_coords = relative_coords * [(2 * self.window_size[1] - 1), 1]
        
        # Sum to get a single index for each relative position pair
        relative_position_index = tf.reduce_sum(relative_coords, axis=-1) # (Wh*Ww, Wh*Ww)
        
        # Store as a non-trainable variable
        self.relative_position_index = tf.Variable(
            relative_position_index, trainable=False, name="relative_position_index", dtype=tf.int32
        )

    def call(self, x, mask=None):
        B_, N, C = x.shape # B_ is batch_size * num_windows, N is num_patches_per_window, C is channels
        
        # Project input to Query, Key, Value
        qkv = self.qkv(x) # (B_, N, C*3)
        # Reshape to (3, B_, num_heads, N, head_dim)
        qkv = tf.reshape(qkv, (-1, N, 3, self.num_heads, C // self.num_heads))
        qkv = tf.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale # Scale Query
        attn = (q @ tf.transpose(k, (0, 1, 3, 2))) # Compute attention logits (B_, num_heads, N, N)

        # Retrieve and apply relative position bias
        relative_position_bias = tf.gather(
            self.relative_position_bias_table, 
            tf.reshape(self.relative_position_index, [-1])
        ) # (Wh*Ww * Wh*Ww, num_heads)
        relative_position_bias = tf.reshape(
            relative_position_bias, 
            (self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        ) # (N, N, num_heads)
        relative_position_bias = tf.transpose(relative_position_bias, (2, 0, 1)) # (num_heads, N, N)
        # Add relative bias to attention logits (broadcast across batch dimension)
        attn = attn + tf.expand_dims(relative_position_bias, axis=0) 

        if mask is not None:
            # Apply attention mask for shifted windows
            nW = tf.shape(mask)[0] # Number of windows in the batch (relevant if multiple image masks are concatenated)
            attn = tf.reshape(attn, (-1, nW, self.num_heads, N, N))
            # Expand mask dims for broadcasting to (1, nW, 1, N, N)
            attn = attn + tf.cast(tf.expand_dims(tf.expand_dims(mask, axis=1), axis=0), attn.dtype) 
            attn = tf.reshape(attn, (-1, self.num_heads, N, N))
            attn = tf.nn.softmax(attn, axis=-1)
        else:
            attn = tf.nn.softmax(attn, axis=-1)

        attn = self.attn_drop(attn) # Dropout on attention weights

        # Apply attention to Value and reshape
        x = tf.transpose((attn @ v), (0, 2, 1, 3)) # (B_, N, num_heads, head_dim)
        x = tf.reshape(x, (-1, N, C)) # Reshape back to original dimensions
        x = self.proj(x) # Final linear projection
        x = self.proj_drop(x) # Dropout on output
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "window_size": self.window_size,
            "num_heads": self.num_heads,
            "qkv_bias": True, 
            "qk_scale": self.scale if self.scale != (self.dim // self.num_heads)**-0.5 else None,
            "attn_drop": self.attn_drop.rate,
            "proj_drop": self.proj_drop.rate,
        })
        return config

class SwinTransformerBlock(tf.keras.layers.Layer):
    """
    Swin Transformer Block, supporting both Window MSA (W-MSA) and Shifted Window MSA (SW-MSA).
    """
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=LayerNormalization, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.input_resolution = input_resolution # (H, W) of the feature map
        self.num_heads = num_heads
        
        # Ensure window_size and shift_size are tuples
        if isinstance(window_size, int):
            self.window_size = (window_size, window_size)
        else:
            self.window_size = window_size

        if isinstance(shift_size, int):
            self.shift_size = (shift_size, shift_size)
        else:
            self.shift_size = shift_size

        self.mlp_ratio = mlp_ratio
        
        # Adjust window_size if input resolution is smaller (e.g., for very small feature maps)
        if min(self.input_resolution) <= min(self.window_size):
            self.shift_size = (0, 0) # No shifting if only one window covers the input
            self.window_size = self.input_resolution # Set window size to input resolution

        # Layer normalization before attention and MLP
        self.norm1 = norm_layer()
        self.attn = WindowAttention(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop
        )
        # DropPath (Stochastic Depth) for regularization
        self.drop_path = tf.keras.layers.Dropout(drop_path, noise_shape=(None, 1, 1)) if drop_path > 0. else tf.identity
        self.norm2 = norm_layer()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(hidden_features=mlp_hidden_dim, out_features=dim, drop=drop)

        # Calculate attention mask for Shifted Window MSA (SW-MSA)
        if self.shift_size[0] > 0 or self.shift_size[1] > 0:
            H, W = self.input_resolution
            # Create a mask to prevent attention between non-adjacent sub-windows
            img_mask = np.zeros((1, H, W, 1))  # (1, H, W, 1)
            
            # Define slices for regions that will be separated by the shift
            # These slices delineate different regions for masking
            h_slices = (slice(0, H - self.window_size[0]),
                        slice(H - self.window_size[0], H - self.shift_size[0]),
                        slice(H - self.shift_size[0], H))
            w_slices = (slice(0, W - self.window_size[1]),
                        slice(W - self.window_size[1], W - self.shift_size[1]),
                        slice(W - self.shift_size[1], W))
            
            cnt = 0
            # Assign unique IDs to each region
            for h in h_slices:
                for w in w_slices:
                    if h.start < h.stop and w.start < w.stop: # Ensure valid slices
                        img_mask[:, h, w, :] = cnt
                        cnt += 1

            # Partition the mask into windows
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = tf.reshape(mask_windows, [-1, self.window_size[0] * self.window_size[1]])
            
            # Compute the attention mask: 0 for same region, -100 for different regions
            attn_mask = tf.expand_dims(mask_windows, axis=1) - tf.expand_dims(mask_windows, axis=2)
            attn_mask = tf.where(attn_mask != 0, -100.0, 0.0) # Large negative value for masking
            self.attn_mask = tf.Variable(attn_mask, trainable=False, dtype=tf.float32, name="attn_mask")
        else:
            self.attn_mask = None # No mask needed for W-MSA

    def call(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape # B, H*W, C (flattened patches)

        # Reshape input to 4D tensor (B, H, W, C)
        x_reshaped = tf.reshape(x, (-1, H, W, C))

        # Cyclic shift (for SW-MSA)
        if self.shift_size[0] > 0 or self.shift_size[1] > 0:
            shifted_x = tf.roll(x_reshaped, shift=[-self.shift_size[0], -self.shift_size[1]], axis=[1, 2])
        else:
            shifted_x = x_reshaped
        
        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = tf.reshape(x_windows, (-1, self.window_size[0] * self.window_size[1], C))

        # W-MSA / SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # Merge windows
        attn_windows = tf.reshape(attn_windows, (-1, self.window_size[0], self.window_size[1], C))
        shifted_x = window_reverse(attn_windows, self.window_size, H, W, C)

        # Reverse cyclic shift
        if self.shift_size[0] > 0 or self.shift_size[1] > 0:
            x_unshifted = tf.roll(shifted_x, shift=[self.shift_size[0], self.shift_size[1]], axis=[1, 2])
        else:
            x_unshifted = shifted_x
        
        # Flatten back to (B, H*W, C)
        x_unrolled = tf.reshape(x_unshifted, (-1, H * W, C))

        # Apply Layer Norm and MLP with residual connections
        x = x + self.drop_path(self.norm1(x_unrolled)) # Residual connection after MSA
        x = x + self.drop_path(self.norm2(self.mlp(x))) # Residual connection after MLP
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "input_resolution": self.input_resolution,
            "num_heads": self.num_heads,
            "window_size": self.window_size,
            "shift_size": self.shift_size,
            "mlp_ratio": self.mlp_ratio,
            "qkv_bias": True, 
            "qk_scale": None, 
            "drop": self.mlp.drop.rate, 
            "attn_drop": self.attn.attn_drop.rate,
            "drop_path": self.drop_path.rate if isinstance(self.drop_path, tf.keras.layers.Dropout) else 0.,
            "norm_layer": LayerNormalization, 
        })
        return config


def build_custom_deeplab_with_swin(input_shape=(256, 256, 3), num_classes=6,
                                  swin_dim=256, swin_num_heads=8, swin_window_size=8,
                                  num_swin_blocks=2):
    """
    Builds a custom DeepLabV3+ model with an EfficientNetV2S backbone, 
    enhanced with Swin Transformer blocks for high-level feature processing.
    Args:
        input_shape (tuple): Shape of the input images (height, width, channels).
        num_classes (int): Number of semantic classes for segmentation.
        swin_dim (int): Feature dimension for Swin Transformer blocks.
        swin_num_heads (int): Number of attention heads in Swin Transformer blocks.
        swin_window_size (int): Window size for Swin Transformer attention.
        num_swin_blocks (int): Number of Swin Transformer blocks to apply.
    Returns:
        tuple: (Model, backbone_layer)
    """
    inputs = Input(shape=input_shape)
    augmented_inputs = data_augmentation(inputs)

    # EfficientNetV2S as backbone, pre-trained on ImageNet
    backbone = EfficientNetV2S(
        include_top=False,
        weights='imagenet',
        input_tensor=augmented_inputs
    )
    
    # Identify low-level features from the backbone for the decoder skip connection
    # For EfficientNetV2S, 'block3a_project_bn' often provides good low-level features (1/4 resolution)
    low_level_features = None
    target_low_level_shape = (IMG_SIZE[0] // 4, IMG_SIZE[1] // 4) # e.g., (64, 64) for 256x256 input
    try:
        low_level_features = backbone.get_layer('block3a_project_bn').output 
        print(f"Low-level feature layer found by name: 'block3a_project_bn' with shape {low_level_features.shape}")
    except ValueError:
        print(f"Could not find 'block3a_project_bn'. Searching for an 'add' layer with shape {target_low_level_shape}.")
        for layer in reversed(backbone.layers):
            if 'add' in layer.name and layer.output.shape[1:3] == target_low_level_shape:
                low_level_features = layer.output
                print(f"Low-level feature layer found: '{layer.name}' with shape {low_level_features.shape}")
                break
        if low_level_features is None:
            raise ValueError(f"Could not find a suitable low-level feature layer with shape {target_low_level_shape} or by known name.")


    high_level_features = backbone.output # Output of the EfficientNetV2S backbone
    print(f"High-level feature layer: Backbone output with shape {high_level_features.shape}")

    # Reshape high_level_features from (B, H, W, C) to (B, H*W, C) for Swin Transformer Blocks
    B, H_feat, W_feat, C_feat = high_level_features.shape
    x_swin_input = tf.reshape(high_level_features, (-1, H_feat * W_feat, C_feat))

    # Add a projection layer if the backbone's output channels don't match swin_dim
    if C_feat != swin_dim:
        print(f"Warning: Swin input dim ({C_feat}) from backbone does not match swin_dim ({swin_dim}). Adding projection layer.")
        x_swin_input = Dense(swin_dim, name="swin_input_projection")(x_swin_input)
        C_feat = swin_dim # Update C_feat to match the projected dimension

    swin_resolution = (H_feat, W_feat) # Spatial resolution for Swin blocks
    
    # Apply multiple Swin Transformer Blocks
    for i in range(num_swin_blocks):
        x_swin_input = SwinTransformerBlock(
            dim=C_feat, 
            input_resolution=swin_resolution, 
            num_heads=swin_num_heads, 
            window_size=swin_window_size,
            # Alternate between W-MSA (shift_size=0) and SW-MSA (shift_size > 0)
            shift_size=swin_window_size // 2 if i % 2 == 1 else 0, 
            mlp_ratio=4., 
            qkv_bias=True, 
            drop=0.1, # Dropout rate for MLP layers
            attn_drop=0.1, # Dropout rate for attention weights
            drop_path=0.1, # Stochastic depth rate
            name=f"swin_block_{i}"
        )(x_swin_input)

    # Reshape back to 4D tensor (B, H, W, C) after Swin Transformer Blocks
    x_after_swin = tf.reshape(x_swin_input, (-1, H_feat, W_feat, C_feat))

    # Continue with the DeepLabV3+ decoder path
    # ASPP operates on the features enhanced by Swin Blocks
    x = ASPP(x_after_swin, out_channels=256, rates=[6, 12, 18]) 
    
    # Upsample ASPP output to match low-level features resolution
    upsampling_factor = low_level_features.shape[1] // x.shape[1]
    x = UpSampling2D(size=(upsampling_factor, upsampling_factor), interpolation='bilinear')(x)
    
    # Process low-level features
    low = Conv2D(48, 1, padding='same', use_bias=False)(low_level_features)
    low = BatchNormalization()(low)
    low = Activation('relu')(low)
    
    # Concatenate high-level (ASPP + Swin) and low-level features
    x = Concatenate()([x, low])
    
    # Decoder convolutions
    x = Conv2D(256, 3, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x); x = Activation('relu')(x)
    x = Dropout(0.5)(x) # Dropout for regularization
    x = Conv2D(256, 3, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x); x = Activation('relu')(x)
    
    # Final classification layer
    x = Conv2D(num_classes, 1, padding='same')(x)
    # Upsample to original image resolution
    x = UpSampling2D(size=(4, 4), interpolation='bilinear')(x)
    
    final_model = Model(inputs, x)
    return final_model, backbone

# --- INSTANTIATION AND TRAINING ---

# Determine the number of classes based on loaded masks
num_classes = int(np.max(train_y) + 1)
if num_classes <= 1: 
    num_classes = 6 # Default to 6 classes if data is uniform (e.g., all 0s)

# Loss functions 
def dice_loss(y_true, y_pred):
    """Calculates the Dice loss."""
    y_true = tf.one_hot(tf.cast(y_true, tf.int32), num_classes) # Convert true labels to one-hot
    y_pred = tf.nn.softmax(y_pred, axis=-1) # Apply softmax to logits
    numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=-1)
    denominator = tf.reduce_sum(y_true + y_pred, axis=-1)
    return 1 - (numerator + 1e-6) / (denominator + 1e-6) # Add epsilon for stability

def combined_loss(y_true, y_pred):
    """Combines Sparse Categorical Crossentropy and Dice loss."""
    ce_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
    d_loss = dice_loss(y_true, y_pred)
    return ce_loss + d_loss

class MeanIoUFromLogits(tf.keras.metrics.Metric):
    """
    Custom Keras Metric for Mean IoU that handles logits output from the model.
    """
    def __init__(self, num_classes, name='mean_iou', dtype=None):
        super(MeanIoUFromLogits, self).__init__(name=name, dtype=dtype)
        self.num_classes = num_classes
        self.mean_iou = tf.keras.metrics.MeanIoU(num_classes=self.num_classes)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert logits to predicted class labels by taking argmax
        y_pred_labels = tf.argmax(y_pred, axis=-1)
        # Ensure y_true has the same integer type as the predicted labels
        y_true_labels = tf.cast(y_true, y_pred_labels.dtype)
        # Update the internal MeanIoU metric's state
        self.mean_iou.update_state(y_true_labels, y_pred_labels, sample_weight)

    def result(self):
        # Return the final result from the internal metric
        return self.mean_iou.result()

    def reset_state(self, *args, **kwargs):
        # Reset the state of the internal metric at the start of each epoch
        self.mean_iou.reset_state()
        
    def get_config(self):
        config = super(MeanIoUFromLogits, self).get_config()
        config.update({'num_classes': self.num_classes})
        return config


# Instantiate the custom MeanIoU metric
iou_metric = MeanIoUFromLogits(num_classes=num_classes)

# Build the model using the updated function
model, backbone_layer = build_custom_deeplab_with_swin(input_shape=(*IMG_SIZE, 3), num_classes=num_classes)
model.summary()

# Phase 1: Train the decoder (backbone frozen)
print("\nFreezing the backbone for Phase 1: Training the Decoder with SW-MSA enhanced features...")
backbone_layer.trainable = False # Freeze backbone layers

model.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-5),
    loss=combined_loss,
    metrics=[iou_metric] 
)

cbs1 = [
    EarlyStopping(monitor='val_mean_iou', mode='max', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_mean_iou', mode='max', factor=0.5, patience=5, min_lr=1e-6),
    ModelCheckpoint('phase1_custom_model_swin.keras', monitor='val_mean_iou', mode='max', save_best_only=True)
]

print("\n--- Starting Phase 1: Training the Decoder ---")
model.fit(
    train_X_processed, train_y,
    validation_data=(val_X_processed, val_y),
    batch_size=16, epochs=3, # Reduced epochs for a quicker demonstration
    callbacks=cbs1
)

# Phase 2: Fine-tune the entire model (backbone unfrozen)
print("\nUnfreezing the backbone for Phase 2: Fine-Tuning the Full Model with SW-MSA enhanced features...")
backbone_layer.trainable = True # Unfreeze all backbone layers

# Re-compile the model with a lower learning rate for fine-tuning
model.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-6),
    loss=combined_loss,
    metrics=[iou_metric] 
)

cbs2 = [
    EarlyStopping(monitor='val_mean_iou', mode='max', patience=15, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_mean_iou', mode='max', factor=0.2, patience=5, min_lr=1e-7),
    ModelCheckpoint('weed_detector_final_model_swin.keras', monitor='val_mean_iou', mode='max', save_best_only=True)
]

print("\n--- Starting Phase 2: Fine-Tuning the Full Model ---")
model.fit(
    train_X_processed, train_y,
    validation_data=(val_X_processed, val_y),
    batch_size=8,
    epochs=3, # Reduced epochs for a quicker demonstration
    callbacks=cbs2
)

print("\nTraining complete. The final model has been saved as 'weed_detector_final_model_swin.keras'")
