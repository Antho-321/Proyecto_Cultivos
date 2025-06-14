import tensorflow as tf
from tensorflow.keras import layers

def WASP(x,
         out_channels: int = 256,
         dilation_rates=(6, 12, 18, 24),
         use_global_pool=True,
         name: str = "WASP"):
    """
    Waterfall Atrous Spatial Pooling (WASP)
    Args
    ----
    x               : Tensor de entrada (CUALQUIER tamaño H×W×C)
    out_channels    : Nº de filtros en cada rama (típicamente 256)
    dilation_rates  : Secuencia de dilataciones crecientes (cascada)
    use_global_pool : Añade rama de pooling global (opcional)
    name            : Prefijo para las capas
    Returns
    -------
    Tensor fusionado listo para el decodificador.
    """
    convs = []

    # 0) 1×1 para mantener la rama “sin dilatación”
    y = layers.Conv2D(out_channels, 1, padding="same",
                      use_bias=False, name=f"{name}_conv1x1")(x)
    y = layers.BatchNormalization(name=f"{name}_bn1")(y)
    y = layers.Activation("relu", name=f"{name}_relu1")(y)
    convs.append(y)                  # se concatena al final
    prev = y                         # arranque de la cascada

    # 1) Cascada de atrous 3×3 con dilataciones crecientes
    for i, r in enumerate(dilation_rates, start=1):
        prev = layers.Conv2D(out_channels, 3,
                             dilation_rate=r,
                             padding="same",
                             use_bias=False,
                             name=f"{name}_conv_d{r}")(prev)
        prev = layers.BatchNormalization(name=f"{name}_bn_d{r}")(prev)
        prev = layers.Activation("relu", name=f"{name}_relu_d{r}")(prev)
        convs.append(prev)           # se guarda la salida de *cada* rama

    # 2) Rama de pooling global opcional (idéntica a DeepLab)
    if use_global_pool:
        gp = layers.GlobalAveragePooling2D(keepdims=True,
                                           name=f"{name}_gap")(x)
        gp = layers.Conv2D(out_channels, 1, padding="same",
                           use_bias=False, name=f"{name}_gp_conv")(gp)
        gp = layers.BatchNormalization(name=f"{name}_gp_bn")(gp)
        gp = layers.Activation("relu", name=f"{name}_gp_relu")(gp)
        # reescalar al tamaño espacial de las demás ramas
        gp = layers.Lambda(
            lambda t: tf.image.resize(
                t[0], tf.shape(t[1])[1:3], method="bilinear"),
            name=f"{name}_gp_resize")([gp, x])
        convs.append(gp)

    # 3) Fusión (concatenación + 1×1) → salida final
    y = layers.Concatenate(name=f"{name}_concat")(convs)
    y = layers.Conv2D(out_channels, 1, padding="same",
                      use_bias=False, name=f"{name}_fuse_conv")(y)
    y = layers.BatchNormalization(name=f"{name}_fuse_bn")(y)
    return layers.Activation("relu", name=f"{name}_fuse_relu")(y)

# Sustituye la línea
# x = ASPP(high_level_features, out_channels=256, rates=[6, 12, 18])
# por
# x = WASP(high_level_features, out_channels=256, dilation_rates=(6, 12, 18, 24))
