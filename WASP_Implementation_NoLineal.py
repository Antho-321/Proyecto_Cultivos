import tensorflow as tf
from tensorflow.keras import layers

def WASP(x,
         out_channels: int = 256,
         levels: int = 5,           # Nº de ramas atrous
         base_rate: int = 1,        # r0; la i-ésima rama usa  r = base_rate * 2**i
         use_global_pool: bool = True,
         anti_gridding: bool = True,
         name: str = "WASP"):
    """
    Waterfall Atrous Spatial Pooling con:
      • Dilataciones exponenciales   (1, 2, 4, 8, …)
      • Suavizado anti-gridding (SDC/HDC) con depth-wise 3×3 extra.
    """
    convs = []

    # 0) Rama 1×1 (sin dilatación)  — re-entrada a la cascada
    y = layers.Conv2D(out_channels, 1, padding="same", use_bias=False,
                      name=f"{name}_conv1x1")(x)
    y = layers.BatchNormalization(name=f"{name}_bn1")(y)
    y = layers.Activation("relu", name=f"{name}_relu1")(y)
    convs.append(y)
    prev = y

    # 1) Cascada atrous exponencial  (rate = base_rate * 2**i)
    for i in range(levels):
        r = base_rate * (2 ** i)           # 1, 2, 4, 8, …
        branch = layers.Conv2D(out_channels, 3, padding="same",
                               dilation_rate=r, use_bias=False,
                               name=f"{name}_conv_d{r}")(prev)
        branch = layers.BatchNormalization(name=f"{name}_bn_d{r}")(branch)
        branch = layers.Activation("relu", name=f"{name}_relu_d{r}")(branch)

        # --- Anti-gridding ---------------------------------------------------
        if anti_gridding and r > 1:
            branch = layers.DepthwiseConv2D(3, padding="same", use_bias=False,
                                            name=f"{name}_ag_dw{r}")(branch)
            branch = layers.BatchNormalization(name=f"{name}_ag_bn{r}")(branch)
            branch = layers.Activation("relu", name=f"{name}_ag_relu{r}")(branch)
        # --------------------------------------------------------------------

        convs.append(branch)
        prev = branch                     # “waterfall” → la siguiente toma la salida anterior

    # 2) Rama de pooling global (opcional, igual que DeepLab)
    if use_global_pool:
        gp = layers.GlobalAveragePooling2D(keepdims=True, name=f"{name}_gap")(x)
        gp = layers.Conv2D(out_channels, 1, padding="same", use_bias=False,
                           name=f"{name}_gp_conv")(gp)
        gp = layers.BatchNormalization(name=f"{name}_gp_bn")(gp)
        gp = layers.Activation("relu", name=f"{name}_gp_relu")(gp)
        gp = layers.Lambda(lambda t: tf.image.resize(t[0],
                                                     tf.shape(t[1])[1:3],
                                                     method="bilinear"),
                            name=f"{name}_gp_resize")([gp, x])
        convs.append(gp)

    # 3) Fusión final
    y = layers.Concatenate(name=f"{name}_concat")(convs)
    y = layers.Conv2D(out_channels, 1, padding="same", use_bias=False,
                      name=f"{name}_fuse_conv")(y)
    y = layers.BatchNormalization(name=f"{name}_fuse_bn")(y)
    return layers.Activation("relu", name=f"{name}_fuse_relu")(y)
