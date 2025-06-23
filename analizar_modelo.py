import torch
import os
from torchinfo import summary
import timm

# =================================================================================
# 1. IMPORTAR LA ARQUITECTURA DEL MODELO (sin cambios)
# =================================================================================
try:
    from model import CloudDeepLabV3Plus
except ImportError:
    print("Error: No se pudo importar 'CloudDeepLabV3Plus' desde 'model.py'.")
    print("Asegúrate de que 'model.py' esté en la misma carpeta y no tenga errores de sintaxis.")
    exit()

# =================================================================================
# 2. FUNCIÓN PRINCIPAL MODIFICADA
# =================================================================================
def mostrar_resumen_modelo(weights_path, input_size=(3, 512, 512), num_classes=1):
    """
    Carga un modelo, sus pesos, y calcula el tamaño de los parámetros del modelo
    y del optimizador desde un archivo .tar.

    Args:
        weights_path (str): Ruta al archivo de pesos (.tar).
        input_size (tuple): Dimensiones de la entrada (canales, altura, ancho).
        num_classes (int): Número de clases de salida para la segmentación.
    """
    timm.create_model('tf_efficientnetv2_s.in21k', pretrained=True)

    if not os.path.exists(weights_path):
        print(f"Error: No se encontró el archivo de pesos en la ruta: {weights_path}")
        return

    device = torch.device('cpu')
    print("1. Instanciando el modelo CloudDeepLabV3Plus...")
    model = CloudDeepLabV3Plus(num_classes=num_classes).to(device)

    # --- INICIO DE MODIFICACIÓN: Cálculo de parámetros y tamaño ---
    optimizer_params_count = 0
    optimizer_state_dict = None

    try:
        print(f"2. Cargando el checkpoint completo desde '{weights_path}'...")
        # Se carga el diccionario completo del checkpoint
        checkpoint = torch.load(weights_path, map_location=device)

        # --- Cargar pesos del MODELO ---
        model_state_dict = None
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                model_state_dict = checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint:
                model_state_dict = checkpoint['model_state_dict']
            else:
                model_state_dict = checkpoint
        else:
            model_state_dict = checkpoint
        
        if model_state_dict:
            model.load_state_dict(model_state_dict, strict=False)
            print("   Pesos del modelo cargados exitosamente.")
        else:
            print(f"Error: No se encontró un 'state_dict' de modelo válido.")
            return

        # --- Buscar y procesar el estado del OPTIMIZADOR ---
        if isinstance(checkpoint, dict):
            if 'optimizer_state_dict' in checkpoint:
                optimizer_state_dict = checkpoint['optimizer_state_dict']
            elif 'optimizer' in checkpoint:
                optimizer_state_dict = checkpoint['optimizer']

        if optimizer_state_dict and 'state' in optimizer_state_dict:
            print("   Estado del optimizador encontrado. Calculando su tamaño...")
            # Iterar sobre los estados de cada parámetro para sumar los elementos
            for param_state in optimizer_state_dict['state'].values():
                for state_tensor in param_state.values():
                    if torch.is_tensor(state_tensor):
                        optimizer_params_count += state_tensor.numel()
        else:
            print("   No se encontró el estado del optimizador en el checkpoint.")

    except Exception as e:
        print(f"\nError al cargar el checkpoint: {e}")
        return
    
    # --- Calcular parámetros y tamaño del MODELO ---
    trainable_params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Asumimos que cada parámetro es un float32 (4 bytes)
    bytes_per_param = 4
    model_size_mb = (trainable_params_count * bytes_per_param) / (1024**2)
    optimizer_size_mb = (optimizer_params_count * bytes_per_param) / (1024**2)
    total_size_mb = model_size_mb + optimizer_size_mb
    # --- FIN DE MODIFICACIÓN ---

    print("\n3. Resumen del Modelo (torchinfo)...")
    print("=" * 80)
    summary(model, input_size=(1, *input_size), device=str(device),
            col_names=["input_size", "output_size", "num_params", "mult_adds"],
            verbose=1)
    print("=" * 80)

    # --- Resumen final de tamaños, similar al del PDF ---
    print("\n4. Resumen de Tamaño del Checkpoint")
    print("-" * 40)
    print(f"Parámetros Entrenados (Modelo): {trainable_params_count:,}")
    print(f"   -> Tamaño Aprox. del Modelo: {model_size_mb:.2f} MB")
    print("-" * 40)
    print(f"Parámetros del Optimizador:   {optimizer_params_count:,}")
    print(f"   -> Tamaño Aprox. del Optimizador: {optimizer_size_mb:.2f} MB")
    print("-" * 40)
    print(f"Tamaño Total del Checkpoint:    {total_size_mb:.2f} MB")
    print("=" * 40)


if __name__ == '__main__':
    # --- PARÁMETROS (ajusta según tu archivo) ---
    ruta_pesos = 'checkpoints/0.8410.pth.tar' # Asegúrate de que esta ruta sea correcta
    tamaño_entrada = (3, 256, 256)
    num_clases = 6

    # --- EJECUCIÓN ---
    mostrar_resumen_modelo(
        weights_path=ruta_pesos,
        input_size=tamaño_entrada,
        num_classes=num_clases
    )