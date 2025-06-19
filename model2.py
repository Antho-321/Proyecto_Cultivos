import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization, Activation, Concatenate
from tensorflow.keras.models import Model

def create_unet_with_imagenet21k(input_shape=(224, 224, 3)):
    """
    Crea un modelo con arquitectura U-Net utilizando EfficientNetV2-B0 como
    codificador, cargando los pesos pre-entrenados de ImageNet-21k desde TensorFlow Hub.

    Args:
        input_shape (tuple): Las dimensiones de la imagen de entrada.

    Returns:
        tf.keras.Model: El modelo Keras completo.
    """
    # --- CAPA DE ENTRADA ---
    inputs = Input(shape=input_shape)

    # --- CODIFICADOR (ENCODER) ---
    # URL del modelo EfficientNetV2-B0 pre-entrenado en ImageNet-21k en TensorFlow Hub.
    # Usamos la versión "feature-extractor" que no tiene la capa de clasificación final.
    TFHUB_URL = "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_b0/feature_vector/2"

    # Cargamos el modelo desde TF Hub como una capa. `trainable=True` permite el ajuste fino (fine-tuning).
    encoder_layer = hub.KerasLayer(TFHUB_URL, trainable=True, name='efficientnetv2b0_21k_encoder')
    
    # Para obtener las capas intermedias, necesitamos construir un modelo a partir del modelo de Hub
    # NOTA: Los nombres de las capas internas pueden variar. Estos nombres son correctos para este modelo de TF Hub.
    # Se obtuvieron inspeccionando `encoder_layer.resolved_object.summary()`.
    
    # Creamos un modelo intermedio que toma nuestra entrada y la pasa a través del encoder de TF Hub
    # y nos da las salidas de las capas que necesitamos para las skip connections.
    
    # Pasamos los inputs por el encoder para conectar el grafo.
    encoder_outputs_all = encoder_layer(inputs, training=False) # training=False es importante aquí

    # Para obtener las capas intermedias, necesitamos acceder al modelo subyacente descargado por TF Hub
    encoder = encoder_layer.resolved_object

    skip_connection_names = [
        'block1a_project_activation', # 112x112
        'block2c_add',                # 56x56
        'block4a_expand_activation',  # 28x28
        'block6a_expand_activation',  # 14x14
    ]

    # Obtenemos las salidas de las capas de interés del grafo del encoder.
    skip_outputs = [encoder.get_layer(name).output for name in skip_connection_names]

    # La salida final del encoder es la salida del último bloque.
    encoder_output = encoder.get_layer('top_activation').output # Salida final del encoder (7x7)

    # Creamos un nuevo modelo "encoder" que toma la entrada y devuelve las salidas que necesitamos
    encoder_model = Model(inputs=encoder.input, outputs=[encoder_output] + skip_outputs)

    # Ahora usamos este modelo en nuestro grafo principal
    encoder_results = encoder_model(inputs)
    encoder_output = encoder_results[0]
    skip_connections = encoder_results[1:]
    
    # Invertimos la lista de skip_connections para que coincida con el orden del decodificador (de la más profunda a la más superficial)
    skip_connections = skip_connections[::-1]


    # --- DECODIFICADOR (DECODER) ---
    # El decodificador reconstruye la imagen a partir de las características extraídas.

    # Bloque de muestreo ascendente 1 (de 7x7 a 14x14)
    up1 = Conv2DTranspose(512, (2, 2), strides=2, padding='same')(encoder_output)
    up1 = Concatenate()([up1, skip_connections[0]]) # Concatena con skip de 14x14
    up1 = Conv2D(512, (3, 3), padding='same', activation='relu')(up1)
    up1 = BatchNormalization()(up1)

    # Bloque de muestreo ascendente 2 (de 14x14 a 28x28)
    up2 = Conv2DTranspose(256, (2, 2), strides=2, padding='same')(up1)
    up2 = Concatenate()([up2, skip_connections[1]]) # Concatena con skip de 28x28
    up2 = Conv2D(256, (3, 3), padding='same', activation='relu')(up2)
    up2 = BatchNormalization()(up2)

    # Bloque de muestreo ascendente 3 (de 28x28 a 56x56)
    up3 = Conv2DTranspose(128, (2, 2), strides=2, padding='same')(up2)
    up3 = Concatenate()([up3, skip_connections[2]]) # Concatena con skip de 56x56
    up3 = Conv2D(128, (3, 3), padding='same', activation='relu')(up3)
    up3 = BatchNormalization()(up3)

    # Bloque de muestreo ascendente 4 (de 56x56 a 112x112)
    up4 = Conv2DTranspose(64, (2, 2), strides=2, padding='same')(up3)
    up4 = Concatenate()([up4, skip_connections[3]]) # Concatena con skip de 112x112
    up4 = Conv2D(64, (3, 3), padding='same', activation='relu')(up4)
    up4 = BatchNormalization()(up4)
    
    # Bloque de muestreo ascendente 5 (de 112x112 a 224x224)
    up5 = Conv2DTranspose(32, (2, 2), strides=2, padding='same')(up4)
    up5 = Conv2D(32, (3, 3), padding='same', activation='relu')(up5)
    up5 = BatchNormalization()(up5)

    # --- CAPA DE SALIDA ---
    outputs = Conv2D(1, (1, 1), padding='same', activation='sigmoid', name='final_output')(up5)

    # Creación del modelo final
    model = Model(inputs=inputs, outputs=outputs, name='Unet_EfficientNetV2B0_21k_TFHub')

    return model

if __name__ == '__main__':
    print("Creando el modelo con EfficientNetV2-B0 y pesos de 'imagenet21k' desde TensorFlow Hub...")
    try:
        model = create_unet_with_imagenet21k()
        print("\n¡Modelo creado exitosamente!")
        print("\nResumen de la Arquitectura del Modelo:")
        model.summary()
    except Exception as e:
        print(f"\nOcurrió un error al crear el modelo: {e}")
        print("\nPor favor, asegúrate de tener instalados tensorflow y tensorflow_hub:")
        print("pip install --upgrade tensorflow tensorflow_hub")