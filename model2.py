import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization, Activation, Concatenate
from tensorflow.keras.models import Model

def create_proposed_model(input_shape=(224, 224, 3)):
    """
    Crea el modelo de arquitectura propuesto (Unet con EfficientNet-B0 como codificador).
    ...
    """
    # Capa de Entrada: Recibe una imagen RGB de 224x224 
    inputs = Input(shape=input_shape)

    # --- ELIMINA TODO EL BLOQUE DE LA CAPA "STEM" DE AQUÍ ---
    # stem = Conv2D(...)
    # stem = BatchNormalization(...)
    # stem = Activation(...)
    # stem = MaxPooling2D(...)

    # --- CODIFICADOR (ENCODER) ---
    # Pasa la entrada ('inputs') directamente a EfficientNetB0.
    # Keras se encargará de crear la capa 'stem' interna correctamente.
    encoder = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_tensor=inputs, # Cambiado de 'stem' a 'inputs'
        input_shape=input_shape
    )

    # El resto del código permanece igual...
    skip_connection_names = [
        'block2a_expand_activation',
        'block3a_expand_activation',
        'block4a_expand_activation',
        'block5a_expand_activation',
    ]
    encoder_outputs = [encoder.get_layer(name).output for name in skip_connection_names]
    
    encoder_output = encoder.output

    # --- DECODIFICADOR (DECODER) ---
    # ... (el resto del código no necesita cambios)
    up1 = Conv2DTranspose(512, (2, 2), strides=2, padding='same')(encoder_output)
    up1 = Concatenate()([up1, encoder_outputs[3]])
    up1 = Conv2D(512, (3, 3), padding='same', activation='relu')(up1)
    up1 = BatchNormalization()(up1)

    up2 = Conv2DTranspose(256, (2, 2), strides=2, padding='same')(up1)
    up2 = Concatenate()([up2, encoder_outputs[2]])
    up2 = Conv2D(256, (3, 3), padding='same', activation='relu')(up2)
    up2 = BatchNormalization()(up2)

    up3 = Conv2DTranspose(128, (2, 2), strides=2, padding='same')(up2)
    up3 = Concatenate()([up3, encoder_outputs[1]])
    up3 = Conv2D(128, (3, 3), padding='same', activation='relu')(up3)
    up3 = BatchNormalization()(up3)

    up4 = Conv2DTranspose(64, (2, 2), strides=2, padding='same')(up3)
    # ¡CUIDADO! La primera skip connection de EfficientNetB0 tiene una resolución
    # de 112x112, pero tu 'stem' personalizado la reducía a 56x56.
    # Al usar el stem nativo, la salida del 'block1a_project_bn' es la primera
    # que podrías usar para un skip, pero tus skip connections empiezan desde el bloque 2,
    # lo cual está bien. Solo asegúrate de que las dimensiones coincidan.
    # La salida de 'block2a_expand_activation' es 56x56, que coincide con tu 'up4'
    up4 = Concatenate()([up4, encoder_outputs[0]])
    up4 = Conv2D(64, (3, 3), padding='same', activation='relu')(up4)
    up4 = BatchNormalization()(up4)

    # --- CAPA DE SALIDA ---
    outputs = Conv2D(1, (1, 1), padding='same', activation='sigmoid', name='final_output')(up4)

    model = Model(inputs=inputs, outputs=outputs, name='Unet_EfficientNetB0')

    return model

if __name__ == '__main__':
    proposed_model = create_proposed_model()
    print("Resumen de la Arquitectura del Modelo Propuesto:")
    proposed_model.summary()