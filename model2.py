import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model
import traceback

def create_unet_with_efficientnetv2_encoder(input_shape=(224, 224, 3)):
    """
    Crea un modelo de codificador multi-salida utilizando EfficientNetV2-B0
    pre-entrenado con ImageNet-21k desde TensorFlow Hub.

    Esta versión utiliza hub.load() para máxima compatibilidad.
    """
    # 1. DEFINE LA URL DEL MODELO Y CÁRGALO USANDO hub.load()
    # hub.load() es más robusto para modelos que no se comportan como capas Keras estándar.
    TFHUB_URL = "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_b0/classification/2"
    
    # Usamos hub.load() para obtener el modelo como un objeto callable de bajo nivel.
    # Establecemos las etiquetas de 'train' para cargar la versión completa del modelo.
    loaded_model = hub.load(TFHUB_URL, tags={'train'})

    # 2. CONSTRUYE UN MODELO KERAS ALREDEDOR DEL MODELO CARGADO
    # --- ESTA ES LA PARTE CORREGIDA ---

    # Paso 2.1: Define una entrada explícita.
    encoder_input = Input(shape=input_shape, name='encoder_input')
    
    # Paso 2.2: Envuelve el modelo cargado en una capa Lambda.
    # La capa Lambda le dice a Keras cómo usar el objeto 'loaded_model' como si fuera una capa.
    # El modelo de Hub espera un diccionario, por lo que lo construimos dentro de la lambda.
    # NOTA: La salida de este modelo de clasificación es un diccionario con logits. Tomamos la salida 'default'.
    model_output = Lambda(lambda x: loaded_model(x, training=False), name='efficientnetv2_lambda')(encoder_input)

    # Paso 2.3: Crea un modelo Keras temporal para poder inspeccionar las capas.
    base_model = Model(inputs=encoder_input, outputs=model_output)

    # 3. IDENTIFICA Y OBTÉN LAS SALIDAS PARA LAS SKIP CONNECTIONS
    # ¡IMPORTANTE! Como no usamos KerasLayer, los nombres de las capas ya no están anidados.
    # Accedemos a ellos directamente desde base_model.
    skip_connection_names = [
        'block1b_add',  # Size: 112x112
        'block2d_add',  # Size: 56x56
        'block4e_add',  # Size: 28x28
        'block6h_add',  # Size: 14x14
    ]
    encoder_output_layer_name = 'block7b_add' # Bottleneck size: 7x7

    # Obtenemos las capas directamente del 'base_model'.
    skip_outputs = [base_model.get_layer(name).output for name in skip_connection_names]
    encoder_output = base_model.get_layer(encoder_output_layer_name).output
    
    # 4. CREA EL MODELO CODIFICADOR FINAL
    encoder = Model(inputs=encoder_input,
                      outputs=[encoder_output] + skip_outputs,
                      name='efficientnetv2_unet_encoder')

    return encoder

# El resto del script permanece igual...
if __name__ == '__main__':
    print("Creando el modelo U-Net con EfficientNetV2-B0 (pesos 'imagenet21k') como codificador...")
    try:
        model = create_unet_with_efficientnetv2_encoder()
        print("\n¡Modelo creado exitosamente!")
        print("\nResumen de la Arquitectura del Modelo:")
        # Es una buena práctica imprimir el resumen para verificar los nombres de las capas.
        model.summary(line_length=120)

    except Exception as e:
        print(f"\n--- ERROR DETALLADO AL CREAR EL MODELO ---")
        print(f"Tipo de Excepción: {type(e).__name__}")
        print(f"Mensaje de Error: {e}")
        print("\nA continuación se muestra el 'traceback' completo del error para facilitar la depuración:")
        print("vvv-------------------------------------------------------------------vvv")
        traceback.print_exc()
        print("^^^-------------------------------------------------------------------^^^")