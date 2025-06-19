import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization, Activation, Concatenate
from tensorflow.keras.models import Model

def create_unet_with_imagenet21k(input_shape=(224, 224, 3)):
    """
    Crea un modelo con arquitectura U-Net utilizando EfficientNetV2-B0 como
    codificador y los pesos pre-entrenados de ImageNet-21k.

    IMPORTANTE: Este script requiere una versión actualizada de TensorFlow
    (ej. 2.10 o superior) que soporte el argumento 'imagenet21k'.

    Args:
        input_shape (tuple): Las dimensiones de la imagen de entrada.

    Returns:
        tf.keras.Model: El modelo Keras compilado.
    """
    # Capa de Entrada: Recibe una imagen RGB de 224x224
    inputs = Input(shape=input_shape)

    # --- CODIFICADOR (ENCODER) ---
    # Se utiliza EfficientNetV2-B0, que soporta los pesos de ImageNet-21k
    # en versiones actualizadas de TensorFlow.
    encoder = tf.keras.applications.EfficientNetV2B0(
        include_top=False,
        weights='imagenet21k',  # Usando los pesos de 21k
        input_tensor=inputs,
        input_shape=input_shape
    )

    # Se identifican las salidas de los bloques del codificador para las conexiones de salto.
    # Estos nombres de capa son estándar en EfficientNetV2B0.
    skip_connection_names = [
        'block1a_project_activation', # 112x112
        'block2c_add',              # 56x56
        'block4a_expand_activation',# 28x28
        'block6a_expand_activation',# 14x14
    ]
    encoder_outputs = [encoder.get_layer(name).output for name in skip_connection_names]

    # La salida final del codificador es la salida del último bloque
    encoder_output = encoder.output # 7x7

    # --- DECODIFICADOR (DECODER) ---
    # El decodificador reconstruye la imagen a partir de las características extraídas.

    # Bloque de muestreo ascendente 1 (de 7x7 a 14x14)
    up1 = Conv2DTranspose(512, (2, 2), strides=2, padding='same')(encoder_output)
    up1 = Concatenate()([up1, encoder_outputs[3]]) # Concatena con skip de 14x14
    up1 = Conv2D(512, (3, 3), padding='same', activation='relu')(up1)
    up1 = BatchNormalization()(up1)

    # Bloque de muestreo ascendente 2 (de 14x14 a 28x28)
    up2 = Conv2DTranspose(256, (2, 2), strides=2, padding='same')(up1)
    up2 = Concatenate()([up2, encoder_outputs[2]]) # Concatena con skip de 28x28
    up2 = Conv2D(256, (3, 3), padding='same', activation='relu')(up2)
    up2 = BatchNormalization()(up2)

    # Bloque de muestreo ascendente 3 (de 28x28 a 56x56)
    up3 = Conv2DTranspose(128, (2, 2), strides=2, padding='same')(up2)
    up3 = Concatenate()([up3, encoder_outputs[1]]) # Concatena con skip de 56x56
    up3 = Conv2D(128, (3, 3), padding='same', activation='relu')(up3)
    up3 = BatchNormalization()(up3)

    # Bloque de muestreo ascendente 4 (de 56x56 a 112x112)
    up4 = Conv2DTranspose(64, (2, 2), strides=2, padding='same')(up3)
    up4 = Concatenate()([up4, encoder_outputs[0]]) # Concatena con skip de 112x112
    up4 = Conv2D(64, (3, 3), padding='same', activation='relu')(up4)
    up4 = BatchNormalization()(up4)
    
    # Bloque de muestreo ascendente 5 (de 112x112 a 224x224)
    # Este último bloque solo sube la resolución para llegar al tamaño original.
    up5 = Conv2DTranspose(32, (2, 2), strides=2, padding='same')(up4)
    up5 = Conv2D(32, (3, 3), padding='same', activation='relu')(up5)
    up5 = BatchNormalization()(up5)


    # --- CAPA DE SALIDA ---
    # Convolución final para obtener el mapa de segmentación binario.
    outputs = Conv2D(1, (1, 1), padding='same', activation='sigmoid', name='final_output')(up5)

    # Creación del modelo final
    model = Model(inputs=inputs, outputs=outputs, name='Unet_EfficientNetV2B0_21k')

    return model

if __name__ == '__main__':
    # Crear una instancia del modelo.
    # Tras actualizar TensorFlow, esto descargará los pesos de ImageNet-21k.
    print("Creando el modelo con EfficientNetV2-B0 y pesos de 'imagenet21k'...")
    try:
        model = create_unet_with_imagenet21k()
        print("\n¡Modelo creado exitosamente!")
        # Imprimir el resumen de la arquitectura.
        print("\nResumen de la Arquitectura del Modelo:")
        model.summary()
    except Exception as e:
        print(f"\nOcurrió un error al crear el modelo: {e}")
        print("\nPor favor, asegúrate de haber actualizado TensorFlow a una versión reciente:")
        print("pip install --upgrade tensorflow")