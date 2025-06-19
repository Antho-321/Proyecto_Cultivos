import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization, Activation, Concatenate
from tensorflow.keras.models import Model

def create_unet_with_efficientnetv2_encoder(input_shape=(224, 224, 3)):
    """
    Crea un modelo de codificador multi-salida utilizando EfficientNetV2-B0 
    pre-entrenado con ImageNet-21k desde TensorFlow Hub.

    Este modelo está diseñado para ser la columna vertebral (encoder) de una 
    arquitectura tipo U-Net.

    Args:
        input_shape (tuple): Las dimensiones de la imagen de entrada.

    Returns:
        tf.keras.Model: Un modelo Keras que toma una imagen de entrada y devuelve
                        la salida del cuello de botella y las 4 salidas para
                        las skip connections.
    """
    # 1. SELECCIONAR LA URL CORRECTA DEL MODELO DE TF HUB
    # Usamos la versión de "clasificación" que contiene todo el cuerpo del modelo,
    # lo que nos permite acceder a las capas intermedias.
    TFHUB_URL = "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_b0/classification/2"

    # 2. CREAR EL CODIFICADOR COMO UN MODELO FUNCIONAL
    # Esto evita los errores de grafo desconectado.
    
    # Definimos la capa de entrada para el codificador.
    encoder_input = Input(shape=input_shape, name='encoder_input')

    # Cargamos el modelo desde TF Hub como una capa. `trainable=True` permite el ajuste fino (fine-tuning).
    hub_layer = hub.KerasLayer(TFHUB_URL, trainable=True, name='efficientnetv2b0_encoder')
    
    # Conectamos la capa de Hub a nuestra entrada. Esto construye el grafo interno del modelo de Hub.
    # El resultado 'outputs' aquí es la salida de clasificación final, que ignoraremos.
    outputs = hub_layer(encoder_input)

    # 3. EXTRAER LAS SALIDAS INTERMEDIAS (SKIP CONNECTIONS)
    # Los nombres de las capas se obtienen inspeccionando `hub_layer.resolved_object.summary()`.
    # Estos nombres corresponden a las salidas de los bloques principales de EfficientNetV2
    # y son adecuados para las skip connections en una U-Net.
    skip_connection_names = [
        'block1b_add',  # Tamaño: 112x112
        'block2d_add',  # Tamaño: 56x56
        'block4e_add',  # Tamaño: 28x28
        'block6h_add',  # Tamaño: 14x14
    ]

    # La salida final del codificador (cuello de botella) antes de la capa de pooling global.
    encoder_output_layer_name = 'block7b_add' # Tamaño: 7x7

    # Obtenemos los tensores de salida de las capas de interés por su nombre.
    # Accedemos al modelo subyacente a través de `hub_layer`.
    skip_outputs = [hub_layer.resolved_object.get_layer(name).output for name in skip_connection_names]
    encoder_output = hub_layer.resolved_object.get_layer(encoder_output_layer_name).output
    
    # Creamos un nuevo modelo Keras que encapsula esta lógica.
    # Este es nuestro codificador final y reutilizable.
    return Model(inputs=encoder_input, outputs=[encoder_output] + skip_outputs, name='efficientnetv2_encoder')


def create_unet_with_imagenet21k(input_shape=(224, 224, 3), num_classes=1):
    """
    Crea un modelo completo con arquitectura U-Net utilizando el codificador 
    EfficientNetV2-B0 pre-entrenado.

    Args:
        input_shape (tuple): Las dimensiones de la imagen de entrada.
        num_classes (int): El número de clases para la segmentación. Para 
                           segmentación binaria, usa 1.

    Returns:
        tf.keras.Model: El modelo Keras completo.
    """
    # --- CAPA DE ENTRADA ---
    inputs = Input(shape=input_shape)

    # --- CODIFICADOR (ENCODER) ---
    # Obtenemos el modelo codificador pre-construido.
    encoder = create_unet_with_efficientnetv2_encoder(input_shape)
    
    # Pasamos la entrada principal a través del codificador para obtener todas las salidas.
    encoder_results = encoder(inputs)
    
    # Desempaquetamos los resultados.
    encoder_output = encoder_results[0]      # Salida del cuello de botella (7x7)
    skip_connections = encoder_results[1:]   # Lista de skip connections
    
    # Invertimos la lista para que coincida con el orden del decodificador (de profundo a superficial).
    skip_connections = skip_connections[::-1]

    # --- DECODIFICADOR (DECODER) ---
    # El decodificador reconstruye la máscara de segmentación a partir de los mapas de características.
    # Usamos el patrón: Upsample -> Concatenate -> ConvBlock(Conv -> BN -> ReLU)
    
    decoder_filters = [256, 128, 64, 32] # Filtros para los bloques del decodificador

    # Bloque de muestreo ascendente 1 (de 7x7 a 14x14)
    x = Conv2DTranspose(decoder_filters[0], (2, 2), strides=2, padding='same')(encoder_output)
    x = Concatenate()([x, skip_connections[0]])
    x = Conv2D(decoder_filters[0], 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Bloque de muestreo ascendente 2 (de 14x14 a 28x28)
    x = Conv2DTranspose(decoder_filters[1], (2, 2), strides=2, padding='same')(x)
    x = Concatenate()([x, skip_connections[1]])
    x = Conv2D(decoder_filters[1], 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Bloque de muestreo ascendente 3 (de 28x28 a 56x56)
    x = Conv2DTranspose(decoder_filters[2], (2, 2), strides=2, padding='same')(x)
    x = Concatenate()([x, skip_connections[2]])
    x = Conv2D(decoder_filters[2], 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Bloque de muestreo ascendente 4 (de 56x56 a 112x112)
    x = Conv2DTranspose(decoder_filters[3], (2, 2), strides=2, padding='same')(x)
    x = Concatenate()([x, skip_connections[3]])
    x = Conv2D(decoder_filters[3], 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Bloque de muestreo ascendente 5 (de 112x112 a 224x224)
    # Este último bloque no tiene skip connection.
    x = Conv2DTranspose(16, (2, 2), strides=2, padding='same')(x)
    x = Conv2D(16, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # --- CAPA DE SALIDA ---
    # La activación final depende del número de clases.
    if num_classes == 1:
        activation = 'sigmoid' # Para segmentación binaria
    else:
        activation = 'softmax' # Para segmentación multiclase

    outputs = Conv2D(num_classes, (1, 1), padding='same', activation=activation, name='final_output')(x)

    # Creación del modelo final
    model = Model(inputs=inputs, outputs=outputs, name='Unet_EfficientNetV2B0_21k')

    return model

if __name__ == '__main__':
    print("Creando el modelo U-Net con EfficientNetV2-B0 (pesos 'imagenet21k') como codificador...")
    try:
        model = create_unet_with_imagenet21k()
        print("\n¡Modelo creado exitosamente!")
        print("\nResumen de la Arquitectura del Modelo:")
        # El summary es muy largo, así que lo imprimimos con un límite de líneas
        model.summary(line_length=120)
    except Exception as e:
        print(f"\nError al crear el modelo: {e}")
        print("Esto puede ocurrir si TensorFlow o TensorFlow Hub no están instalados correctamente,")
        print("o si hay un problema de conexión para descargar el modelo de Hub.")