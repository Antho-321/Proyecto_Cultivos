import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from typing import Dict, List, Union, Optional
import os
# Asegúrate de tener tensorflow instalado para estas importaciones
from tensorflow.keras.preprocessing.image import load_img, img_to_array

#==============================================================================
# FUNCIÓN 1: CARGAR DATOS (Sin cambios)
#==============================================================================
def load_dataset(image_directory, mask_directory):
    images = []
    masks = []

    for image_filename in os.listdir(image_directory):
        if image_filename.lower().endswith(('.jpg', '.png')):
            # Cargar imagen
            img_path = os.path.join(image_directory, image_filename)
            img = load_img(img_path)
            images.append(img_to_array(img))

            # Cargar máscara asociada
            name_wo_ext = os.path.splitext(image_filename)[0]
            mask_filename = f"{name_wo_ext}_mask.png"
            mask_path = os.path.join(mask_directory, mask_filename)

            if os.path.exists(mask_path):
                mask = load_img(mask_path, color_mode='grayscale')
                m = img_to_array(mask).astype(np.int32)
                masks.append(np.squeeze(m, axis=-1))  # (H, W)
            else:
                print(f"No se encontró la máscara para: {image_filename}")

    images = np.array(images)
    masks = np.array(masks)
    print(f"Carga completa. Se encontraron {len(images)} imágenes y {len(masks)} máscaras.")
    return images, masks

# =============================================================================
# FUNCIÓN 2: ANÁLISIS DE DISTRIBUCIÓN  ── con Rich para salida enriquecida
# =============================================================================
def analizar_distribucion_mascaras(
    masks_list: np.ndarray,
    clases: List[int],
    mostrar_resultados: bool = True,
    k_vecino: float = 1.0,          # k en (μ − k·σ)
    percentil_bajo: float = 0.10    # p en Pp (0-1, ej. 0.10 = P10)
):
    """
    Calcula la distribución de píxeles por clase e imagen y muestra
    resultados con formato enriquecido si la librería 'rich' está instalada.
    -------------------------------------------------------------------------
    Retorna
    -------
    df_img : DataFrame  – conteos por imagen.
    stats  : dict       – totales, proporciones, Pp, umbral μ−k·σ, CV, etc.
    """
    # ------------------------------------------------------------------ #
    # 1) Conteo por imagen
    # ------------------------------------------------------------------ #
    filas = [
        {"idx_img": i, **{c: int((mask == c).sum()) for c in clases}}
        for i, mask in enumerate(masks_list)
    ]

    if not filas:
        print("La lista de máscaras está vacía.")
        return None, None

    df_img = pd.DataFrame(filas).set_index("idx_img")

    # ------------------------------------------------------------------ #
    # 2) Estadísticas globales
    # ------------------------------------------------------------------ #
    tot   = df_img.sum()
    prop  = tot / tot.sum() if tot.sum() else tot
    media = df_img.mean()
    desv  = df_img.std(ddof=0)                # población
    cv    = (desv / media).replace([np.inf, np.nan], 0)
    Pp    = df_img.quantile(percentil_bajo)
    um_mu = (media - k_vecino * desv).clip(lower=0).astype(int)  # μ−kσ

    # ------------------------------------------------------------------ #
    # 3) Evaluación de criterios
    # ------------------------------------------------------------------ #
    clases_fuera_Pp     = [c for c in clases if tot[c] < Pp[c]]
    clases_fuera_mu_k   = [c for c in clases if tot[c] < um_mu[c]]
    clases_fuera_cv_max = [c for c in clases if cv[c] > 0.5]

    stats = {
        "totales":            tot.astype(int),
        "proporciones":       prop,
        "percentil_bajo":     Pp.astype(int),
        "umbral_mu_menos_k":  um_mu,
        "media":              media.astype(int),
        "desviacion":         desv.astype(int),
        "coef_variacion":     cv.round(3),
        "clases_fuera_Pp":    clases_fuera_Pp,
        "clases_fuera_mu_k":  clases_fuera_mu_k,
        "clases_cv>0.5":      clases_fuera_cv_max
    }

    # ------------------------------------------------------------------ #
    # 4) Salida enriquecida (Rich) o clásica
    # ------------------------------------------------------------------ #
    if mostrar_resultados:
        try:
            from rich.console import Console
            from rich.table   import Table
            from rich import box

            console = Console()

            # --------- Tabla 1: conteo por imagen (primeras 5) ----------
            t1 = Table(title="Distribución de píxeles (primeras 5 máscaras)",
                       box=box.SIMPLE_HEAD)
            t1.add_column("idx_img", justify="right")
            for c in clases:
                t1.add_column(f"Clase {c}", justify="right")

            for idx, row in df_img.head().iterrows():
                t1.add_row(str(idx), *[f"{row[c]:,}" for c in clases])

            # --------- Tabla 2: estadísticas globales ----------
            t2 = Table(title="Estadísticas globales", box=box.SIMPLE_HEAD)
            t2.add_column("Clase", justify="right")
            t2.add_column("Total", justify="right")
            t2.add_column("%", justify="right")
            t2.add_column(f"P{int(percentil_bajo*100)}", justify="right")
            t2.add_column(f"μ-{k_vecino}σ", justify="right")
            t2.add_column("CV", justify="right")

            for c in clases:
                t2.add_row(
                    str(c),
                    f"{int(tot[c]):,}",
                    f"{(prop[c]*100):.2f}",
                    f"{int(Pp[c]):,}",
                    f"{int(um_mu[c]):,}",
                    f"{cv[c]:.2f}"
                )

            console.print(t1)
            console.print(t2)

            # --------- Listas de clases fuera de umbral ----------
            if clases_fuera_Pp:
                console.print(f"[bold yellow]Clases < P{int(percentil_bajo*100)}[/]: {clases_fuera_Pp}")
            if clases_fuera_mu_k:
                console.print(f"[bold yellow]Clases < μ−{k_vecino}σ[/]: {clases_fuera_mu_k}")
            if clases_fuera_cv_max:
                console.print("[bold yellow]Clases con CV > 0.5[/]:", clases_fuera_cv_max)

        except ImportError:
            # --- Fallback: impresión simple ---
            print("\n========== Distribución de píxeles por imagen ==========")
            print(df_img.head())
            print("\n========== Estadísticas globales ==========")
            print("Totales:\n", tot.astype(int))
            print("Proporción (%):\n", (prop*100).round(2))
            print(f"P{int(percentil_bajo*100)}:\n", Pp.astype(int))
            print(f"μ-{k_vecino}σ:\n", um_mu)
            print("CV:\n", cv.round(3))

    return df_img, stats

#==============================================================================
# FUNCIÓN 3: DIFERENCIA DE PÍXELES (Sin cambios)
#==============================================================================
def calculate_pixel_difference(masks: np.ndarray, class_names: Optional[Dict[int, str]] = None) -> Dict[int, int]:
    """
    Calcula la diferencia total de píxeles entre la clase 0 (fondo) y las demás clases.
    """
    flat_pixels = masks.flatten()
    unique_classes, counts = np.unique(flat_pixels, return_counts=True)
    pixel_counts = dict(zip(unique_classes, counts))
    background_pixel_count = pixel_counts.get(0, 0)
    
    if background_pixel_count == 0:
        print("Advertencia: La clase 0 (fondo) no se encontró. No se pueden calcular diferencias.")
        return {}

    differences = {}
    print("\n--- Diferencia de Píxeles vs. Fondo (Clase 0) ---")
    print(f"Píxeles de Fondo (Clase 0) para referencia: {background_pixel_count}")

    for cls_id, count in pixel_counts.items():
        if cls_id == 0:
            continue
        
        diff = background_pixel_count - count
        differences[int(cls_id)] = diff
        name_str = f" ({class_names[cls_id]})" if class_names and cls_id in class_names else ""
        print(f"Clase {int(cls_id)}{name_str}: {background_pixel_count} - {count} = {diff}")

    return differences


#==============================================================================
# BLOQUE PRINCIPAL DE EJECUCIÓN
#==============================================================================
if __name__ == '__main__':
    # --- 1. CONFIGURACIÓN ---
    # Define tus rutas y clases aquí
    IMAGE_DIR = "Balanced/train/images"
    MASK_DIR = "Balanced/train/masks"
    
    CLASES_A_ANALIZAR = [0, 1, 2, 3, 4, 5]
    CLASS_NAMES = {
        0: 'Fondo', 1: 'Lengua de vaca', 2: 'Diente de león', 3: 'Kikuyo', 4: 'Otras', 5: 'Papa'
    }
    
    # --- 2. CARGA DE DATOS (UNA SOLA VEZ) ---
    # Comprueba si los directorios existen antes de continuar
    if not (os.path.isdir(IMAGE_DIR) and os.path.isdir(MASK_DIR)):
         print(f"Error: No se encontraron los directorios '{IMAGE_DIR}' o '{MASK_DIR}'.")
         print("Asegúrate de que las rutas son correctas y de que el script se ejecuta desde la ubicación adecuada.")
    else:
        images_list, masks_list = load_dataset(IMAGE_DIR, MASK_DIR)

        # --- 3. EJECUCIÓN DE ANÁLISIS ---
        # Solo procede si la carga de datos fue exitosa
        if masks_list.size > 0:
            # Llama a la primera función de análisis (distribución por máscara)
            df_distribucion, stats_distribucion = analizar_distribucion_mascaras(
                masks_list=masks_list, 
                clases=CLASES_A_ANALIZAR
            )

            # Llama a la segunda función de análisis (diferencia total vs fondo)
            pixel_diff_dict = calculate_pixel_difference(
                masks=masks_list, 
                class_names=CLASS_NAMES
            )

            # --- 4. USAR LOS RESULTADOS ---
            # Ahora puedes trabajar con los resultados devueltos por las funciones
            if df_distribucion is not None:
                print("\n\n--- Ejemplo de uso de resultados ---")
                # Por ejemplo, encontrar la máscara con más píxeles de la clase 1
                if 1 in df_distribucion.columns:
                    mascara_con_mas_clase_1 = df_distribucion[1].idxmax()
                    num_pixeles = df_distribucion[1].max()
                    print(f"La máscara con más píxeles de la Clase 1 es la número {mascara_con_mas_clase_1} (con {num_pixeles} píxeles).")