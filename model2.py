import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model
import traceback

def create_unet_with_efficientnetv2_encoder(input_shape=(224, 224, 3)):
    """
    Crea un modelo de codificador multi-salida utilizando EfficientNetV2-B0
    pre-entrenado con ImageNet-21k desde TensorFlow Hub.

    Esta versión utiliza hub.load() y carga el grafo 'serve' por defecto.
    """
    # 1. DEFINE LA URL DEL MODELO Y CÁRGALO USANDO hub.load()
    TFHUB_URL = "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_b0/classification/2"
    
    # --- ESTA ES LA LÍNEA CORREGIDA ---
    # Omitimos el argumento 'tags' para cargar el único grafo disponible ('serve').
    loaded_model = hub.load(TFHUB_URL)

    # 2. CONSTRUYE UN MODELO KERAS ALREDEDOR DEL MODELO CARGADO
    encoder_input = Input(shape=input_shape, name='encoder_input')
    
    # Envolvemos el modelo cargado en una capa Lambda.
    # El grafo 'serve' ya espera estar en modo inferencia, así que no es necesario pasar training=False
    model_output = Lambda(lambda x: loaded_model(x), name='efficientnetv2_lambda')(encoder_input)

    # Creamos un modelo Keras temporal para poder inspeccionar las capas.
    base_model = Model(inputs=encoder_input, outputs=model_output)

    # 3. IDENTIFICA Y OBTÉN LAS SALIDAS PARA LAS SKIP CONNECTIONS
    skip_connection_names = [
        'block1b_add',
        'block2d_add',
        'block4e_add',
        'block6h_add',
    ]
    encoder_output_layer_name = 'block7b_add'

    skip_outputs = [base_model.get_layer(name).output for name in skip_connection_names]
    encoder_output = base_model.get_layer(encoder_output_layer_name).output
    
    # 4. CREA EL MODELO CODIFICADOR FINAL
    encoder = Model(inputs=encoder_input,
                      outputs=[encoder_output] + skip_outputs,
                      name='efficientnetv2_unet_encoder')

    # El modelo EfficientNetV2 original tiene trainable=True por defecto. 
    # Para hacer fine-tuning, lo mantenemos así. Si quisieras congelar el encoder, harías:
    # encoder.trainable = False
    
    return encoder

# El resto de tu script para probar el modelo...
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