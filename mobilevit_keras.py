# mobilevit_keras.py

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Activation

def conv_block(x, filters=16, kernel_size=3, strides=2):
    """Bloque convolucional estándar."""
    conv_layer = layers.Conv2D(
        filters, kernel_size, strides=strides, activation=tf.nn.swish, padding="same"
    )
    return conv_layer(x)

def inverted_residual_block(x, expanded_channels, output_channels):
    """Bloque residual invertido (como en MobileNetV2)."""
    m = layers.Conv2D(expanded_channels, 1, padding="same", use_bias=False)(x)
    m = layers.BatchNormalization()(m)
    m = Activation('swish')(m)

    m = layers.DepthwiseConv2D(
        3, strides=1, padding="same", use_bias=False
    )(m)
    m = layers.BatchNormalization()(m)
    m = Activation('swish')(m)

    m = layers.Conv2D(output_channels, 1, padding="same", use_bias=False)(m)
    m = layers.BatchNormalization()(m)

    return layers.Add()([m, x])

def mlp(x, hidden_units, dropout_rate):
    """MLP estándar para los bloques Transformer."""
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.swish)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

def transformer_block(x, projection_dim, num_heads=4, dropout=0.1):
    """Bloque Transformer estándar."""
    # Normalización y Atención Multi-Cabeza
    x1 = layers.LayerNormalization(epsilon=1e-6)(x)
    attention_output = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=projection_dim, dropout=dropout
    )(x1, x1)
    
    # Conexión residual
    x2 = layers.Add()([attention_output, x])

    # Normalización y MLP
    x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
    x3 = mlp(x3, hidden_units=[x.shape[-1] * 2, x.shape[-1]], dropout_rate=dropout)

    # Conexión residual final
    return layers.Add()([x3, x2])

def mobilevit_block(x, num_blocks, projection_dim, strides=1):
    """El bloque principal de MobileViT que combina convoluciones y Transformers."""
    # Convolución local
    local_features = layers.Conv2D(
        projection_dim, 3, strides=strides, padding="same", activation=tf.nn.swish
    )(x)
    local_features = layers.Conv2D(
        projection_dim, 1, strides=1, padding="same", activation=tf.nn.swish
    )(local_features)

    # Desplegar en parches y aplicar Transformers
    num_patches = local_features.shape[1] * local_features.shape[2]
    unfolded_features = layers.Reshape((num_patches, projection_dim))(local_features)
    
    for _ in range(num_blocks):
        unfolded_features = transformer_block(unfolded_features, projection_dim)

    # Plegar los parches de vuelta a la forma de imagen
    folded_features = layers.Reshape((*local_features.shape[1:-1], projection_dim))(
        unfolded_features
    )
    
    # Fusión con una convolución 1x1
    folded_features = layers.Conv2D(
        x.shape[-1], 1, strides=1, padding="same", activation=tf.nn.swish
    )(folded_features)
    
    # Conexión residual con la entrada original
    return layers.Add()([x, folded_features])


def mobile_vit_s(input_shape=(256, 256, 3)):
    """Construye el modelo MobileViT-S y devuelve el modelo y las salidas de las skip connections."""
    inputs = keras.Input(input_shape)

    # --- STEM ---
    # Etapa 1: De 256x256 a 128x128
    x = conv_block(inputs, filters=32, kernel_size=3, strides=2)
    s1 = x # Skip connection 1 (128x128)

    # --- BODY ---
    # Etapa 2: De 128x128 a 64x64
    x = inverted_residual_block(x, expanded_channels=64, output_channels=32)
    x = inverted_residual_block(x, expanded_channels=64, output_channels=32)
    x = conv_block(x, filters=64, kernel_size=3, strides=2) 
    s2 = x # Skip connection 2 (64x64)

    # Etapa 3: De 64x64 a 32x32
    x = mobilevit_block(x, num_blocks=2, projection_dim=96, strides=1)
    x = conv_block(x, filters=128, kernel_size=3, strides=2)
    s3 = x # Skip connection 3 (32x32)

    # Etapa 4: De 32x32 a 16x16
    x = mobilevit_block(x, num_blocks=4, projection_dim=128, strides=1)
    x = conv_block(x, filters=256, kernel_size=3, strides=2)
    s4 = x # Skip connection 4 (16x16)
    
    # Etapa 5 (Bottleneck): De 16x16 a 8x8
    x = mobilevit_block(x, num_blocks=3, projection_dim=160, strides=1)
    bottleneck = conv_block(x, filters=320, kernel_size=3, strides=2)

    # Crear el modelo backbone que expone las salidas intermedias
    backbone = keras.Model(
        inputs=inputs, 
        outputs=[s1, s2, s3, s4, bottleneck], 
        name="mobilevit_s_backbone"
    )
    return backbone