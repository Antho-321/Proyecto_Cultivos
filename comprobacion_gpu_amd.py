import tensorflow as tf

# Lista dispositivos GPU reconocidos
gpus = tf.config.list_physical_devices('GPU')
print("GPUs reconocidas por DirectML:", gpus)

for gpu in gpus:
    print(f"  • {gpu.name} ({gpu.device_type})")