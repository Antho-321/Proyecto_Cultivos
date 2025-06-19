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
    # Para acceder a las capas internas, necesitamos construir un modelo temporal
    # que envuelva la KerasLayer.
    pre_model = tf.keras.Model(encoder_input, hub_layer(encoder_input, training=False))

    # 3. IDENTIFICA Y OBTÉN LAS SALIDAS PARA LAS SKIP CONNECTIONS
    # Los nombres de las capas están anidados dentro de la KerasLayer
    try:
        # Accede al modelo concreto cargado por KerasLayer.
        # En TensorFlow > 2.5, se puede acceder a través de `_func` o un método similar.
        # Este es el punto más frágil, ya que depende de la implementación interna de KerasLayer.
        # Una forma más robusta es construir el modelo y luego buscar las capas.
        internal_model = hub_layer.resolved_object
    
        skip_connection_names = [
            'block1b_add', # -> Después del bloque 1
            'block2d_add', # -> Después del bloque 2
            'block4e_add', # -> Después del bloque 4
            'block6h_add', # -> Después del bloque 6
        ]
        encoder_output_layer_name = 'block7b_add'

        skip_outputs = [internal_model.get_layer(name).output for name in skip_connection_names]
        encoder_output = internal_model.get_layer(encoder_output_layer_name).output

        # Reconstruimos el modelo final con las salidas correctas
        outputs = [encoder_output] + skip_outputs
        
        # Creamos un nuevo modelo que sí tiene múltiples salidas
        encoder = Model(inputs=encoder_input, outputs=outputs, name='efficientnetv2_unet_encoder')

    except AttributeError:
        # Fallback para cuando `resolved_object` no está disponible o la estructura es diferente.
        # Este es un enfoque más general: construir el modelo y luego buscar las capas.
        print("Advertencia: No se pudo acceder directamente a `resolved_object`. Usando un método alternativo para obtener las capas.")
        
        # Creamos un modelo completo para poder buscar las capas por su nombre
        # El truco es que KerasLayer anida el modelo. El nombre de la capa KerasLayer es 'efficientnetv2_basemodel'
        # Y las capas de adentro tienen ese prefijo.
        full_model = Model(inputs=encoder_input, outputs=hub_layer(encoder_input))
        
        skip_connection_names = [
            'block1b_add', 'block2d_add', 'block4e_add', 'block6h_add'
        ]
        encoder_output_layer_name = 'block7b_add'
        
        # Para obtener la capa interna, necesitas referenciar el modelo que contiene la KerasLayer
        try:
            skip_outputs = [full_model.get_layer('efficientnetv2_basemodel').get_layer(name).output for name in skip_connection_names]
            encoder_output = full_model.get_layer('efficientnetv2_basemodel').get_layer(encoder_output_layer_name).output
        except ValueError as e:
            print(f"Error al obtener las capas: {e}")
            print("Nombres de capas disponibles dentro de 'efficientnetv2_basemodel':")
            # Lista todas las capas para facilitar la depuración
            for layer in full_model.get_layer('efficientnetv2_basemodel').layers:
                print(layer.name)
            raise e # relanza la excepción después de imprimir la ayuda

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
        # --- BLOQUE DE EXCEPCIÓN MEJORADO ---
        
        # 1. Captura el traceback completo como una cadena de texto para un control total
        traceback_str = traceback.format_exc()

        print(f"\n--- ERROR DETALLADO AL CREAR EL MODELO ---")
        print(f"Tipo de Excepción: {type(e).__name__}")
        print(f"Mensaje de Error: {e}")
        
        # 2. Imprime los argumentos de la excepción, que pueden contener más detalles
        print(f"Argumentos de la Excepción: {e.args}")
        
        print("\nA continuación se muestra el 'traceback' completo del error para facilitar la depuración:")
        print("vvv-------------------------------------------------------------------vvv")
        # 3. Imprime la cadena del traceback que capturamos
        print(traceback_str)
        print("^^^-------------------------------------------------------------------^^^")
        print("\nCONSEJO: Revisa las últimas líneas del traceback para identificar la línea exacta del código que causó el problema.")