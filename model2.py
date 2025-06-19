import tensorflow as tf
from tensorflow.keras.layers import (
    Input, 
    Conv2D, 
    MaxPooling2D, 
    Conv2DTranspose,
    BatchNormalization, 
    Activation, 
    Concatenate
)
from tensorflow.keras.models import Model

def create_exact_pdf_architecture(input_shape=(224, 224, 3)):
    """
    Crea una réplica exacta del modelo de arquitectura propuesto en el PDF.
    
    Esta implementación incluye la 'Stem Layer' personalizada descrita en el
    documento, que consiste en Conv2D -> BatchNormalization -> Swish -> MaxPooling2D,
    antes de pasar la salida a la red troncal EfficientNet-B0.
    
    Argumentos:
        input_shape (tuple): Las dimensiones de la imagen de entrada.
        
    Retorna:
        model (tf.keras.Model): El modelo Keras compilado.
    """
    
    # Capa de Entrada: Recibe una imagen RGB de 224x224.
    inputs = Input(shape=input_shape)

    # --- STEM LAYER PERSONALIZADA (Según el PDF) ---
    # La descripción en el PDF especifica la siguiente secuencia.
    # El número de filtros (32) se asume para que coincida con la entrada del
    # primer bloque de EfficientNetB0.
    
    # 1. Aplicar una convolución 3x3 con un stride de 2.
    stem = Conv2D(32, (3, 3), strides=2, padding='same', name='stem_conv')(inputs)
    
    # 2. Seguido de batch normalization y activación swish.
    stem = BatchNormalization(name='stem_bn')(stem)
    stem = Activation('swish', name='stem_activation')(stem)
    
    # 3. Aplicar max pooling 2x2 para reducir dimensiones espaciales.
    stem_output = MaxPooling2D(pool_size=(2, 2), name='stem_max_pool')(stem)

    # --- CODIFICADOR (ENCODER) ---
    # Se utiliza EfficientNet-B0 como codificador.
    # Se inicializa sin su capa superior (clasificador) ni su propia capa 'stem',
    # ya que hemos creado una personalizada.
    encoder = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_tensor=stem_output,
        input_shape=(56, 56, 32) # La forma de salida de nuestra stem layer
    )

    # Nombres de las capas de donde se extraerán las conexiones de salto (skip connections)
    # Estas capas corresponden a diferentes niveles de extracción de características.
    skip_connection_names = [
        'block2a_expand_activation', # Resolución 56x56
        'block3a_expand_activation', # Resolución 28x28
        'block4a_expand_activation', # Resolución 14x14
        'block5c_add',               # Resolución 7x7 (antes de los últimos bloques)
    ]
    
    # Salidas del codificador para las conexiones de salto.
    encoder_outputs = [encoder.get_layer(name).output for name in skip_connection_names]
    
    # Salida final del codificador.
    encoder_output = encoder.output

    # --- DECODIFICADOR (DECODER) / BLOQUES DE MUESTREO ASCENDENTE ---
    # La descripción del PDF especifica: 2x2 Transpose Conv -> Concatenate -> 3x3 Conv -> BN -> ReLU.
    
    # Bloque de muestreo ascendente 1
    up1 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(encoder_output)
    up1 = Concatenate()([up1, encoder_outputs[3]])
    up1 = Conv2D(256, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1) # Activación ReLU como se especifica en el PDF.

    # Bloque de muestreo ascendente 2
    up2 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(up1)
    up2 = Concatenate()([up2, encoder_outputs[2]])
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)

    # Bloque de muestreo ascendente 3
    up3 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(up2)
    up3 = Concatenate()([up3, encoder_outputs[1]])
    up3 = Conv2D(64, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    
    # Bloque de muestreo ascendente 4
    up4 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(up3)
    up4 = Concatenate()([up4, encoder_outputs[0]])
    up4 = Conv2D(32, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)

    # --- CAPA DE SALIDA ---
    # La descripción del PDF especifica: Convolución 1x1 -> Activación Sigmoid.
    outputs = Conv2D(
        1, 
        (1, 1), 
        padding='same', 
        activation='sigmoid', 
        name='final_output'
    )(up4)

    # Creación del modelo final
    model = Model(inputs=inputs, outputs=outputs, name='Exact_PDF_Unet_EfficientNetB0')

    return model

if __name__ == '__main__':
    # Crear el modelo según la arquitectura exacta del PDF
    exact_model = create_exact_pdf_architecture()
    
    # Imprimir el resumen del modelo para verificar su estructura
    print("Resumen de la Réplica Exacta de la Arquitectura del PDF:")
    exact_model.summary()