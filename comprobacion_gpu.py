import tensorflow as tf

# --- Verificación de GPU (ahora usando el tf ya importado) ---
print("Versión de TensorFlow:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPUs disponibles: {[gpu.name for gpu in gpus]}")
    except RuntimeError as e:
        print(e)
else:
    print("No se encontró ninguna GPU. El entrenamiento se ejecutará en la CPU.")