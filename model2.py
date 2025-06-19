import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import traceback

def create_unet_with_efficientnetv2_encoder(input_shape=(224, 224, 3)):
    """
    Crea un modelo de codificador multi-salida utilizando EfficientNetV2-B0
    pre-entrenado con ImageNet-21k desde TensorFlow Hub.

    Este modelo está diseñado para ser la columna vertebral (encoder) de una
    arquitectura tipo U-Net.
    """
    # 1. DEFINE LA URL DEL MODELO Y CREA LA CAPA DE HUB
    TFHUB_URL = "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_b0/classification/2"
    hub_layer = hub.KerasLayer(TFHUB_URL, trainable=True, name='efficientnetv2b0_encoder')

    # 2. CONSTRUYE UN MODELO ALREDEDOR DE LA CAPA DE HUB PARA PODER INSPECCIONARLO
    # --- ESTA ES LA PARTE CORREGIDA ---

    # Paso 2.1: Define una entrada explícita para nuestro nuevo modelo.
    # No intentamos obtener la entrada DESDE hub_layer, sino que creamos la nuestra.
    encoder_input = Input(shape=input_shape, name='encoder_input')
    
    # Paso 2.2: Pasa la entrada a través de la capa de Hub para obtener la salida final.
    # Esto construye el grafo del modelo.
    model_output = hub_layer(encoder_input)

    # Paso 2.3: Crea un modelo Keras temporal. Este modelo SÍ es un tf.keras.Model
    # estándar y podemos inspeccionar sus capas internas.
    base_model = Model(inputs=encoder_input, outputs=model_output)

    # 3. IDENTIFICA Y OBTÉN LAS SALIDAS PARA LAS SKIP CONNECTIONS
    # Ahora que 'base_model' es un modelo Keras real, podemos usar .get_layer()
    # para acceder a las capas internas que la capa de Hub ha creado.
    skip_connection_names = [
        'block1b_add',  # Size: 112x112
        'block2d_add',  # Size: 56x56
        'block4e_add',  # Size: 28x28
        'block6h_add',  # Size: 14x14
    ]
    encoder_output_layer_name = 'block7b_add' # Bottleneck size: 7x7

    # Obtenemos las capas del grafo del 'base_model' que acabamos de crear.
    # El nombre de la capa de hub ('efficientnetv2b0_encoder') actúa como un prefijo.
    skip_outputs = [base_model.get_layer('efficientnetv2b0_encoder').get_layer(name).output for name in skip_connection_names]
    encoder_output = base_model.get_layer('efficientnetv2b0_encoder').get_layer(encoder_output_layer_name).output

    # 4. CREA EL MODELO CODIFICADOR FINAL
    # Este modelo toma nuestra entrada original y produce la salida del cuello de botella
    # y las salidas de las skip connections.
    encoder = Model(inputs=encoder_input,
                      outputs=[encoder_output] + skip_outputs,
                      name='efficientnetv2_unet_encoder')

    return encoder

# El resto del script permanece igual
if __name__ == '__main__':
    print("Creando el modelo U-Net con EfficientNetV2-B0 (pesos 'imagenet21k') como codificador...")
    try:
        model = create_unet_with_efficientnetv2_encoder()
        print("\n¡Modelo creado exitosamente!")
        print("\nResumen de la Arquitectura del Modelo:")
        model.summary(line_length=120)

    except Exception as e:
        print(f"\n--- ERROR DETALLADO AL CREAR EL MODELO ---")
        print(f"Tipo de Excepción: {type(e).__name__}")
        print(f"Mensaje de Error: {e}")
        print("\nA continuación se muestra el 'traceback' completo del error para facilitar la depuración:")
        print("vvv-------------------------------------------------------------------vvv")
        traceback.print_exc()
        print("^^^-------------------------------------------------------------------^^^")
        
        print("\nPosibles Causas Comunes y Soluciones:")
        print("- Conexión a Internet: Verifica tu conexión. El modelo necesita ser descargado desde TensorFlow Hub la primera vez.")
        print("- Nombres de Capas: Asegúrate de que los nombres en 'skip_connection_names' y 'encoder_output_layer_name' son correctos para la versión del modelo de Hub que estás usando.")
        print("- Dependencias: Confirma que 'tensorflow' y 'tensorflow_hub' están instalados correctamente (`pip install tensorflow tensorflow_hub`).")
        print("- Incompatibilidad de Versiones: Podría haber un conflicto entre las versiones de TensorFlow, Keras y el modelo cargado desde Hub.")