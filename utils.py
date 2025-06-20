import numpy as np
from tensorflow.keras.utils import load_img, img_to_array
import os

def verificar_mascara_desde_archivo(ruta_archivo_mascara):
  """
  Verifica si un archivo de imagen de máscara (.png) contiene únicamente
  los valores de píxel 0 y 4.

  Args:
    ruta_archivo_mascara (str): La ruta al archivo de imagen de la máscara.

  Returns:
    bool: True si la máscara contiene solo los valores 0 y 4, False en caso contrario.
           También devuelve False si el archivo no se encuentra.
  """
  # 1. Verificar si el archivo existe para evitar errores
  if not os.path.exists(ruta_archivo_mascara):
      print(f"Error: El archivo no se encontró en la ruta: {ruta_archivo_mascara}")
      return False

  try:
    # 2. Cargar la imagen de la máscara en modo de escala de grises
    # Esto es crucial para que los valores de los píxeles sean números únicos (0, 1, 2...)
    # en lugar de tuplas RGB como (0,0,0).
    mascara_img = load_img(ruta_archivo_mascara, color_mode='grayscale')

    # 3. Convertir la imagen cargada a un arreglo de NumPy
    mascara_array = img_to_array(mascara_img)

    # 4. Obtener los valores únicos del arreglo
    # El arreglo tendrá valores de 0 a 255.
    valores_unicos = np.unique(mascara_array.astype(int))

    # 5. Comparar con el conjunto de valores deseado {0, 4}
    valores_objetivo = {0, 4}
    
    return set(valores_unicos) == valores_objetivo

  except Exception as e:
    print(f"Ocurrió un error al procesar el archivo {ruta_archivo_mascara}: {e}")
    return False