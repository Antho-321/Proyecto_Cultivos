import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization, Activation, Concatenate
from tensorflow.keras.models import Model

def create_proposed_model(input_shape=(224, 224, 3)):
    """
    Crea el modelo de arquitectura propuesto (Unet con EfficientNet-B0 como codificador).

    Esta función se basa en la descripción detallada proporcionada en el documento PDF,
    incluyendo la capa de tallo (Stem), los bloques del codificador, los bloques de muestreo
    ascendente del decodificador y la capa de salida.

    Args:
        input_shape (tuple): Las dimensiones de la imagen de entrada.

    Returns:
        tf.keras.Model: El modelo Keras compilado de la arquitectura propuesta.
    """
    # Capa de Entrada: Recibe una imagen RGB de 224x224 
    inputs = Input(shape=input_shape)

    # --- Capa de Tallo (Stem Layer) ---
    # Aplica una convolución 3x3 con stride de 2, seguida de Batch Norm y activación swish 
    stem = Conv2D(32, (3, 3), strides=2, padding='same', name='stem_conv')(inputs)
    stem = BatchNormalization(name='stem_bn')(stem)
    stem = Activation('swish', name='stem_activation')(stem)
    # Aplica max pooling de 2x2 para reducir dimensiones espaciales 
    stem = MaxPooling2D((2, 2), name='stem_pool')(stem)

    # --- CODIFICADOR (ENCODER) ---
    # Se utiliza EfficientNet-B0 como el codificador para extraer características 
    # Se carga sin la capa de clasificación (include_top=False)
    encoder = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights='imagenet', # Se usan pesos pre-entrenados como punto de partida estándar
        input_tensor=stem
    )

    # Se identifican las salidas de los bloques del codificador para las conexiones de salto (skip connections) 
    # Estos nombres de capa son estándar en la implementación Keras de EfficientNet
    skip_connection_names = [
        'block2a_expand_activation',  # Después del Bloque 2
        'block3a_expand_activation',  # Después del Bloque 3
        'block4a_expand_activation',  # Después del Bloque 4
        'block5a_expand_activation',  # Después del Bloque 5 (último antes de la salida del codificador)
    ]
    encoder_outputs = [encoder.get_layer(name).output for name in skip_connection_names]
    
    # La salida final del codificador es la salida del último bloque
    encoder_output = encoder.output

    # --- DECODIFICADOR (DECODER) ---
    # Utiliza bloques de muestreo ascendente para reconstruir el mapa de segmentación 
    
    # Bloque de muestreo ascendente 1
    # Convolución transpuesta 2x2 para upsampling 
    up1 = Conv2DTranspose(512, (2, 2), strides=2, padding='same')(encoder_output)
    # Concatenación con la salida del bloque 5 del codificador (skip connection) 
    up1 = Concatenate()([up1, encoder_outputs[3]])
    # Convolución 3x3, Batch Norm y activación ReLU 
    up1 = Conv2D(512, (3, 3), padding='same', activation='relu')(up1)
    up1 = BatchNormalization()(up1)


    # Bloque de muestreo ascendente 2
    up2 = Conv2DTranspose(256, (2, 2), strides=2, padding='same')(up1)
    # Concatenación con la salida del bloque 4 del codificador 
    up2 = Concatenate()([up2, encoder_outputs[2]])
    up2 = Conv2D(256, (3, 3), padding='same', activation='relu')(up2)
    up2 = BatchNormalization()(up2)

    # Bloque de muestreo ascendente 3
    up3 = Conv2DTranspose(128, (2, 2), strides=2, padding='same')(up2)
    # Concatenación con la salida del bloque 3 del codificador 
    up3 = Concatenate()([up3, encoder_outputs[1]])
    up3 = Conv2D(128, (3, 3), padding='same', activation='relu')(up3)
    up3 = BatchNormalization()(up3)

    # Bloque de muestreo ascendente 4
    up4 = Conv2DTranspose(64, (2, 2), strides=2, padding='same')(up3)
    # Concatenación con la salida del bloque 2 del codificador 
    up4 = Concatenate()([up4, encoder_outputs[0]])
    up4 = Conv2D(64, (3, 3), padding='same', activation='relu')(up4)
    up4 = BatchNormalization()(up4)

    # --- CAPA DE SALIDA ---
    # Aplica una convolución 1x1 para reducir canales al número de clases 
    # Aplica una activación sigmoide para segmentación binaria 
    # La salida es una imagen segmentada de 224x224x1 
    outputs = Conv2D(1, (1, 1), padding='same', activation='sigmoid', name='final_output')(up4)

    # Creación del modelo final
    model = Model(inputs=inputs, outputs=outputs, name='Unet_EfficientNetB0')

    return model

if __name__ == '__main__':
    # Crear una instancia del modelo
    proposed_model = create_proposed_model()
    
    # Imprimir el resumen de la arquitectura para verificar su estructura
    print("Resumen de la Arquitectura del Modelo Propuesto:")
    proposed_model.summary()