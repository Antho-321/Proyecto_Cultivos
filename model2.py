import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import traceback

def create_unet_with_efficientnetv2_encoder(input_shape=(224, 224, 3)):
    """
    Crea un modelo de codificador multi-salida utilizando EfficientNetV2-B0
    cargado como una capa de Keras para permitir el acceso a las capas intermedias.
    """
    TFHUB_URL = "https://tfhub.dev/google/efficientnetv2-b0/feature-vector/2" # Usamos el feature-vector para más flexibilidad

    # 1. DEFINE LA ENTRADA
    encoder_input = Input(shape=input_shape, name='encoder_input')

    # --- ESTA ES LA PARTE CORREGIDA ---
    # 2. CARGA EL MODELO USANDO hub.KerasLayer
    # Esto integra el modelo del Hub como una capa nativa de Keras,
    # permitiendo el acceso a su estructura interna.
    # Lo ponemos como entrenable para permitir el fine-tuning.
    hub_layer = hub.KerasLayer(TFHUB_URL, trainable=True, name='efficientnetv2_basemodel')
    
    # El modelo del Hub se llama directamente sobre la entrada
    model_output = hub_layer(encoder_input)

    # 3. CONSTRUYE EL MODELO BASE PARA PODER INSPECCIONAR LAS CAPAS
    # Ahora 'base_model' contiene todas las capas internas de EfficientNetV2
    base_model = Model(inputs=encoder_input, outputs=model_output)
    
    # DESCOMENTA LA SIGUIENTE LÍNEA SI QUIERES VER TODAS LAS CAPAS DISPONIBLES
    # print([layer.name for layer in base_model.layers[-1].layers])

    # 4. IDENTIFICA Y OBTÉN LAS SALIDAS PARA LAS SKIP CONNECTIONS
    # Los nombres de las capas están anidados dentro de la KerasLayer
    skip_connection_names = [
        'block1b_add', # -> Después del bloque 1
        'block2d_add', # -> Después del bloque 2
        'block4e_add', # -> Después del bloque 4
        'block6h_add', # -> Después del bloque 6
    ]
    # La salida final del encoder antes de la cabeza de clasificación original
    encoder_output_layer_name = 'block7b_add'

    # Obtenemos las capas desde el objeto 'hub_layer' anidado
    skip_outputs = [base_model.get_layer('efficientnetv2_basemodel').get_layer(name).output for name in skip_connection_names]
    encoder_output = base_model.get_layer('efficientnetv2_basemodel').get_layer(encoder_output_layer_name).output
    
    # 5. CREA EL MODELO CODIFICADOR FINAL
    encoder = Model(inputs=encoder_input,
                    outputs=[encoder_output] + skip_outputs,
                    name='efficientnetv2_unet_encoder')
    
    return encoder

# El resto de tu script para probar el modelo...
if __name__ == '__main__':
    print("Creando el modelo U-Net con EfficientNetV2-B0 como codificador...")
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