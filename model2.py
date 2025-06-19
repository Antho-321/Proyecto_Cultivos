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

def create_functional_pdf_replica(input_shape=(224, 224, 3)):
    """
    Crea una réplica funcional del modelo del PDF, corrigiendo la
    inconsistencia arquitectónica para que el modelo sea construible.

    La corrección principal es omitir la capa MaxPooling2D de la 'Stem Layer'
    descrita en el PDF, ya que sus dimensiones de salida (56x56) son
    incompatibles con la entrada que espera el primer bloque de EfficientNetB0 (112x112).
    """
    
    # --- ENTRADA Y CODIFICADOR BASE ---
    # Usaremos el modelo EfficientNetB0 pre-entrenado como base.
    # Se carga sin la capa superior y con la forma de entrada estándar.
    base_encoder = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )

    # --- CONEXIONES DE SALTO (SKIP CONNECTIONS) ---
    # Nombres de las capas del codificador base que usaremos para las conexiones de salto.
    skip_connection_names = [
        'stem_activation',           # Para la primera conexión después del stem
        'block2c_add',               # Salida de Bloque 2
        'block3c_add',               # Salida de Bloque 3
        'block4d_add',               # Salida de Bloque 4
        'block6a_expand_activation'  # Salida de Bloque 6 (antes del final)
    ]
    
    encoder_outputs = [base_encoder.get_layer(name).output for name in skip_connection_names]
    
    # Salida final del codificador base para el inicio del decodificador.
    encoder_output = base_encoder.output

    # --- DECODIFICADOR (DECODER) / BLOQUES DE MUESTREO ASCENDENTE ---
    # Se construye la ruta del decodificador (U-Net) sobre las salidas del codificador.
    
    # Bloque de muestreo ascendente 1
    up1 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(encoder_output)
    up1 = Concatenate()([up1, encoder_outputs[4]])
    up1 = Conv2D(512, (3, 3), padding='same', activation='relu')(up1)
    up1 = BatchNormalization()(up1)

    # Bloque de muestreo ascendente 2
    up2 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(up1)
    up2 = Concatenate()([up2, encoder_outputs[3]])
    up2 = Conv2D(256, (3, 3), padding='same', activation='relu')(up2)
    up2 = BatchNormalization()(up2)

    # Bloque de muestreo ascendente 3
    up3 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(up2)
    up3 = Concatenate()([up3, encoder_outputs[2]])
    up3 = Conv2D(128, (3, 3), padding='same', activation='relu')(up3)
    up3 = BatchNormalization()(up3)

    # Bloque de muestreo ascendente 4
    up4 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(up3)
    up4 = Concatenate()([up4, encoder_outputs[1]])
    up4 = Conv2D(64, (3, 3), padding='same', activation='relu')(up4)
    up4 = BatchNormalization()(up4)
    
    # Bloque de muestreo ascendente 5
    up5 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(up4)
    up5 = Concatenate()([up5, encoder_outputs[0]])
    up5 = Conv2D(32, (3, 3), padding='same', activation='relu')(up5)
    up5 = BatchNormalization()(up5)

    # --- CAPA DE SALIDA ---
    outputs = Conv2D(
        1, 
        (1, 1), 
        padding='same', 
        activation='sigmoid', 
        name='final_output'
    )(up5)

    # Creación del modelo final
    model = Model(inputs=base_encoder.input, outputs=outputs, name='Functional_Unet_EfficientNetB0')

    return model

if __name__ == '__main__':
    # Crear el modelo funcional
    functional_model = create_functional_pdf_replica()
    
    # Imprimir el resumen del modelo para verificar su estructura
    print("Resumen de la Réplica Funcional de la Arquitectura del PDF:")
    functional_model.summary()