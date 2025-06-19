# model.py

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
    Crea una réplica funcional del modelo del PDF, utilizando los nombres de
    capa correctos para las conexiones de salto (skip connections) según se
    requiere en el entorno de ejecución.
    """
    
    # --- ENTRADA Y CODIFICADOR BASE ---
    # Se carga el modelo EfficientNetB0 pre-entrenado como base.
    base_encoder = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )

    # --- CONEXIONES DE SALTO (SKIP CONNECTIONS) ---
    # Nombres de las capas del codificador corregidos según el traceback.
    # Estos son los nombres que existen en tu versión de Keras/TensorFlow.
    skip_connection_names = [
        'stem_activation',           # Salida del Stem (Resolución 112x112)
        'block2b_add',               # Salida del Bloque 2 (Resolución 56x56)
        'block3b_add',               # Salida del Bloque 3 (Resolución 28x28)
        'block4c_add',               # Salida del Bloque 4 (Resolución 14x14)
        'block6a_expand_activation'  # Salida del Bloque 6 (Resolución 7x7)
    ]
    
    encoder_outputs = [base_encoder.get_layer(name).output for name in skip_connection_names]
    
    # Salida final del codificador para el inicio del decodificador.
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
    up2_resized = Conv2DTranspose(80, (2, 2), strides=(2, 2), padding='same')(encoder_outputs[3])  # Resize here
    up2 = Concatenate()([up2, up2_resized])  # Concatenate resized encoder output
    up2 = Conv2D(256, (3, 3), padding='same', activation='relu')(up2)
    up2 = BatchNormalization()(up2)

    # Bloque de muestreo ascendente 3
    up3 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(up2)
    # Resize encoder_outputs[2] to match the dimensions of up3
    encoder_output_resized = Conv2DTranspose(40, (2, 2), strides=(2, 2), padding='same')(encoder_outputs[2])  
    up3 = Concatenate()([up3, encoder_output_resized])  # Concatenate resized encoder output
    up3 = Conv2D(128, (3, 3), padding='same', activation='relu')(up3)
    up3 = BatchNormalization()(up3)

    # Bloque de muestreo ascendente 4
    up4 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(up3)
    # Resize encoder_outputs[1] to match the dimensions of up4
    encoder_output_resized = Conv2DTranspose(24, (2, 2), strides=(2, 2), padding='same')(encoder_outputs[1])  
    up4 = Concatenate()([up4, encoder_output_resized])  # Concatenate resized encoder output
    up4 = Conv2D(64, (3, 3), padding='same', activation='relu')(up4)
    up4 = BatchNormalization()(up4)
    
    # Bloque de muestreo ascendente 5
    up5 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(up4)
    # Resize encoder_outputs[0] to match the dimensions of up5
    encoder_output_resized = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(encoder_outputs[0])  
    up5 = Concatenate()([up5, encoder_output_resized])  # Concatenate resized encoder output
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

# if __name__ == '__main__':
#     # Crear el modelo funcional
#     functional_model = create_functional_pdf_replica()
    
#     # Imprimir el resumen del modelo para verificar su estructura
#     print("Resumen de la Réplica Funcional y Corregida de la Arquitectura del PDF:")
#     functional_model.summary()