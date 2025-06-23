import torch
import timm
from collections import OrderedDict # Importa OrderedDict

# ======================= ZONA DE CONFIGURACIÓN =======================
# --- 1. Importa la clase de tu modelo desde el archivo model.py ---
from model import CloudDeepLabV3Plus

# --- 2. Define las rutas de tus archivos ---
RUTA_CHECKPOINT_ENTRADA = r'C:\Users\Administrador\Documents\INGENIERIA_EN_SOFTWARE\BIG_DATA\CÓDIGO\Proyecto_Cultivos\checkpoints\0.8410.pth.tar' 
RUTA_MODELO_SALIDA_PT = 'cultivos_deeplab_final.pt'

# --- 3. Define los parámetros de tu modelo ---
NUM_CLASES = 6
FORMA_INPUT_EJEMPLO = (1, 3, 256, 256) 
# ===================================================================

def convertir():
    """
    Función principal para cargar el checkpoint y convertirlo a TorchScript (.pt).
    """
    print("Iniciando el proceso de conversión...")

    # --- Paso 1: Instanciar la arquitectura del modelo ---
    print(f"1. Instanciando la arquitectura 'CloudDeepLabV3Plus' con {NUM_CLASES} clase(s)...")
    modelo = CloudDeepLabV3Plus(num_classes=NUM_CLASES)
    print("   Arquitectura creada.")

    # --- Paso 2: Cargar los pesos desde el archivo .pth.tar (VERSIÓN CORREGIDA) ---
    try:
        print(f"2. Cargando pesos desde el checkpoint '{RUTA_CHECKPOINT_ENTRADA}'...")
        checkpoint = torch.load(RUTA_CHECKPOINT_ENTRADA, map_location=torch.device('cpu'))
        
        # Accedemos al diccionario de estado original
        state_dict_cargado = checkpoint['state_dict']
        
        # ¡AQUÍ ESTÁ LA MAGIA!
        # Creamos un nuevo diccionario para almacenar las claves corregidas.
        # El checkpoint fue guardado desde un modelo compilado (`torch.compile`), 
        # por lo que todas las claves tienen el prefijo '_orig_mod.'. Lo eliminamos.
        print("   Detectado prefijo '_orig_mod.'. Limpiando las claves del state_dict...")
        new_state_dict = OrderedDict()
        for k, v in state_dict_cargado.items():
            if k.startswith('_orig_mod.'):
                # Elimina el prefijo '_orig_mod.' (que tiene 10 caracteres)
                name = k[10:] 
                new_state_dict[name] = v
            else:
                # Si por alguna razón una clave no tiene el prefijo, la dejamos como está
                new_state_dict[k] = v

        # Cargamos el diccionario de estado ya corregido en el modelo
        modelo.load_state_dict(new_state_dict)
        print("   Pesos cargados exitosamente en la arquitectura.")

    except FileNotFoundError:
        print(f"[ERROR] No se encontró el archivo de checkpoint: '{RUTA_CHECKPOINT_ENTRADA}'")
        return
    except KeyError:
        print("[ERROR] No se encontró la clave 'state_dict' en el checkpoint.")
        print("         Claves disponibles en tu archivo:", checkpoint.keys())
        return
    except Exception as e:
        print(f"[ERROR] Ocurrió un error inesperado al cargar el checkpoint: {e}")
        return

    # --- Paso 3: Poner el modelo en modo de evaluación ---
    modelo.eval()
    print("3. Modelo puesto en modo de evaluación (model.eval()).")

    # --- Paso 4: Crear una entrada de ejemplo y trazar el modelo ---
    print(f"4. Creando una entrada de ejemplo (dummy input) con forma: {FORMA_INPUT_EJEMPLO}")
    dummy_input = torch.randn(FORMA_INPUT_EJEMPLO)
    
    print("   Trazando el modelo para convertirlo a TorchScript...")
    try:
        traced_model = torch.jit.trace(
        modelo, dummy_input,
        check_trace=True,
        check_inputs=[torch.randn(*FORMA_INPUT_EJEMPLO) for _ in range(3)]
    )
    except Exception as e:
        print(f"[ERROR] Ocurrió un error durante el trazado de TorchScript: {e}")
        return

    # --- Paso 5: Guardar el modelo trazado en el archivo .pt ---
    traced_model.save(RUTA_MODELO_SALIDA_PT)
    print("\n==========================================================")
    print(f"¡CONVERSIÓN EXITOSA!")
    print(f"El modelo listo para inferencia ha sido guardado en: '{RUTA_MODELO_SALIDA_PT}'")
    print("==========================================================")

if __name__ == '__main__':
    convertir()