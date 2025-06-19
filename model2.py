import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.layers import Input
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
    # 1. DEFINE THE MODEL URL AND LOAD THE LAYER
    # Using the feature vector version is often more direct for feature extraction.
    # However, the classification model contains the full architecture, which is what we need
    # to access intermediate layers.
    TFHUB_URL = "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_b0/classification/2"

    # Load the model from TF Hub as a layer, but DO NOT call it yet.
    # We set trainable=True to allow for fine-tuning.
    hub_layer = hub.KerasLayer(TFHUB_URL, trainable=True, name='efficientnetv2b0_encoder')
    
    # 2. WRAP THE HUB LAYER IN A KERAS MODEL TO EXTRACT INTERMEDIATE LAYERS
    # This is the correct way to build a new model from a pre-trained one.
    
    # Get the underlying model loaded by the KerasLayer.
    # It's a standard Keras model, and we can inspect it.
    base_model = hub_layer.resolved_object

    # Set the input of our new encoder to be the same as the base model's input.
    encoder_input = base_model.input

    # 3. IDENTIFY AND GET THE OUTPUTS FOR SKIP CONNECTIONS
    # The layer names are found by inspecting `base_model.summary()`.
    skip_connection_names = [
        'block1b_add',  # Size: 112x112
        'block2d_add',  # Size: 56x56
        'block4e_add',  # Size: 28x28
        'block6h_add',  # Size: 14x14
    ]
    encoder_output_layer_name = 'block7b_add' # Bottleneck size: 7x7

    # Get the actual output tensors from the base_model graph.
    skip_outputs = [base_model.get_layer(name).output for name in skip_connection_names]
    encoder_output = base_model.get_layer(encoder_output_layer_name).output
    
    # 4. CREATE THE NEW ENCODER MODEL
    # This model is a new graph that starts at the original input and ends at our
    # desired intermediate and final outputs.
    # NOTE: The input shape of this model is fixed by the TF Hub model (e.g., 224x224).
    # We will pass our desired input_shape when creating the final U-Net.
    encoder = Model(inputs=encoder_input, 
                    outputs=[encoder_output] + skip_outputs, 
                    name='efficientnetv2_encoder')

    # Now, we need to wrap our new encoder so it can accept a custom input shape
    final_input = Input(shape=input_shape)
    final_output = encoder(final_input)

    return Model(inputs=final_input, outputs=final_output, name='custom_input_encoder')

if __name__ == '__main__':
    print("Creando el modelo U-Net con EfficientNetV2-B0 (pesos 'imagenet21k') como codificador...")
    try:
        model = create_unet_with_efficientnetv2_encoder()
        print("\n¡Modelo creado exitosamente!")
        print("\nResumen de la Arquitectura del Modelo:")
        # El summary es muy largo, así que lo imprimimos con un límite de líneas
        model.summary(line_length=120)

    # ----- BLOQUE DE EXCEPCIÓN MODIFICADO -----
    except Exception as e:
        print(f"\n--- ERROR DETALLADO AL CREAR EL MODELO ---")
        print(f"Tipo de Excepción: {type(e).__name__}")
        print(f"Mensaje de Error: {e}")
        print("\nA continuación se muestra el 'traceback' completo del error para facilitar la depuración:")
        print("vvv-------------------------------------------------------------------vvv")
        # Imprime la pila de llamadas completa para identificar el origen del error.
        traceback.print_exc()
        print("^^^-------------------------------------------------------------------^^^")
        
        print("\nPosibles Causas Comunes y Soluciones:")
        print("- Conexión a Internet: Verifica tu conexión. El modelo necesita ser descargado desde TensorFlow Hub la primera vez.")
        print("- Nombres de Capas: Asegúrate de que los nombres en 'skip_connection_names' y 'encoder_output_layer_name' son correctos para la versión del modelo de Hub que estás usando.")
        print("- Dependencias: Confirma que 'tensorflow' y 'tensorflow_hub' están instalados correctamente (`pip install tensorflow tensorflow_hub`).")
        print("- Incompatibilidad de Versiones: Podría haber un conflicto entre las versiones de TensorFlow, Keras y el modelo cargado desde Hub.")