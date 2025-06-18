import torch
import os
# from torchsummary import summary  <-- ELIMINADO
from torchinfo import summary      # <-- AÑADIDO: Se usa torchinfo en su lugar
import timm

# =================================================================================
# 1. IMPORTAR LA ARQUITECTURA DEL MODELO DESDE TU ARCHIVO
# =================================================================================
# Esta línea asume que la clase CloudDeepLabV3Plus está definida en model.py
try:
    from model import CloudDeepLabV3Plus
except ImportError:
    print("Error: No se pudo importar 'CloudDeepLabV3Plus' desde 'model.py'.")
    print("Asegúrate de que 'model.py' esté en la misma carpeta y no tenga errores de sintaxis.")
    exit()

# =================================================================================
# 2. FUNCIÓN PRINCIPAL PARA MOSTRAR EL RESUMEN
# =================================================================================
def mostrar_resumen_modelo(weights_path, input_size=(3, 512, 512), num_classes=1):
    """
    Carga un modelo CloudDeepLabV3Plus, aplica los pesos desde un archivo .tar
    y muestra un resumen de la arquitectura.

    Args:
        weights_path (str): Ruta al archivo de pesos (.tar).
        input_size (tuple): Dimensiones de la entrada (canales, altura, ancho).
        num_classes (int): Número de clases de salida para la segmentación.
    """
    # Forzar la descarga del modelo preentrenado si no está en caché
    # Esto es importante para que el backbone se cargue correctamente
    timm.create_model('tf_efficientnetv2_s.in21k', pretrained=True)

    # Verificar si el archivo de pesos existe
    if not os.path.exists(weights_path):
        print(f"Error: No se encontró el archivo de pesos en la ruta: {weights_path}")
        return

    # Usar CPU para asegurar la compatibilidad.
    device = torch.device('cpu')

    print("1. Instanciando el modelo CloudDeepLabV3Plus desde model.py...")
    # Instanciamos el modelo importado
    model = CloudDeepLabV3Plus(num_classes=num_classes).to(device)

    try:
        print(f"2. Cargando los pesos desde '{weights_path}'...")
        # Cargar el archivo .tar. map_location='cpu' asegura que se cargue en la CPU.
        checkpoint = torch.load(weights_path, map_location=device, weights_only=True)

        # Los checkpoints pueden guardar el state_dict directamente o dentro de una clave
        # Se ha modificado para ser más robusto y mostrar las claves encontradas si falla.
        state_dict = None
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                # Si no encuentra claves comunes, intenta cargarlo directamente.
                # Puede ser un state_dict guardado sin un diccionario contenedor.
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        if state_dict:
            # strict=False es más flexible si hay claves que no coinciden (ej. del optimizador)
            model.load_state_dict(state_dict, strict=False)
            print("   Pesos cargados exitosamente.")
        else:
            print(f"\nError: No se encontró un 'state_dict' válido en el archivo. Claves encontradas: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else 'Ninguna'}")
            return

    except Exception as e:
        print(f"\nError al cargar los pesos: {e}")
        print("Asegúrate de que el archivo .tar contenga un 'state_dict' de PyTorch válido.")
        return

    # --- MODIFICACIÓN CLAVE ---
    print("\n3. Generando el resumen del modelo...")
    print("=" * 80)
    # Reemplaza la llamada a torchsummary por torchinfo
    # Se añade el tamaño del lote (batch_size=1) al input_size
    summary(model, input_size=(1, *input_size), device=str(device),
            col_names=["input_size", "output_size", "num_params", "mult_adds"],
            verbose=1)
    print("=" * 80)
    # --- FIN DE LA MODIFICACIÓN ---

    # El resumen de torchinfo ya incluye los parámetros, pero los dejamos para consistencia
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,} (calculado manualmente)")
    print(f"Trainable params: {trainable_params:,} (calculado manualmente)")
    print(f"Input size (CHW): {input_size}")


if __name__ == '__main__':
    # --- PARÁMETROS ---
    # Ruta al archivo que contiene los pesos del modelo
    ruta_pesos = 'checkpoints/0.8143miou.tar'

    # Tamaño de la imagen de entrada (canales, altura, ancho)
    # Asegúrate de que coincida con cómo entrenaste el modelo
    tamaño_entrada = (3, 256, 256)

    # Número de clases de salida
    num_clases = 6

    # --- EJECUCIÓN ---
    # Asegúrate de instalar torchinfo primero: pip install torchinfo
    mostrar_resumen_modelo(
        weights_path=ruta_pesos,
        input_size=tamaño_entrada,
        num_classes=num_clases
    )
    