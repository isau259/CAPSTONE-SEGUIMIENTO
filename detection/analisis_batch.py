import funciones_2 as fn2
import pandas as pd
from pathlib import Path
import re
import sys
import warnings

# ==================================================================
# --- PARÁMETROS GLOBALES DEL ANÁLISIS ---
# (Ajustados a tu última configuración)

WIN_SIZE_SEC = 150       # 150s para CSI
STEP_SEC = 5             # Step de 5s
ROLLING_WINDOW_SEC = 300 # Baseline de 5 min
PERSISTENCE_SEC = 10     # Persistencia de 10s
MIN_VOTES_ALERTA = 3     # 3 de 6 criterios

# --- PARÁMETROS DE ESTE SCRIPT ---
SEGUNDOS_ANTES_DE_CRISIS = 200 # Tu ventana de "Predicción"
# ==================================================================

def analizar_archivo(sub, ses, run, seg):
    """
    Ejecuta el pipeline de análisis completo para un solo archivo
    y devuelve un desglose de Predicciones, Detecciones y Falsas Alarmas.
    """
    
    # 1. Cargar y 2. Procesar (sin cambios)
    datos = fn2.crear_variable(sub, ses, run, seg)
    df = fn2.procesamiento(datos, 
                          win_size=WIN_SIZE_SEC, 
                          step=STEP_SEC, 
                          smooth_n=5, 
                          overview_ds=10)
    
    # 3. Cálculo de alertas (sin cambios)
    df_alertas = fn2.calcular_baseline_y_alertas(df, 
                                                window_rolling=ROLLING_WINDOW_SEC, 
                                                persistence_sec=PERSISTENCE_SEC, 
                                                min_votes=MIN_VOTES_ALERTA)

    # 4. Extraer resultados y filtrar
    onset_abs = df_alertas.attrs.get("onset_abs")
    offset_abs = df_alertas.attrs.get("offset_abs")
    
    # Obtener todas las alertas persistentes que se activaron
    df_alertas_persistentes = df_alertas[df_alertas['ALERTA_PERSISTENTE'] == True]

    # --- NUEVA LÓGICA: Separar en 3 Categorías ---
    
    segundos_prediccion = 0
    segundos_deteccion_durante_crisis = 0
    segundos_falsa_alarma = 0

    # CASO 1: Es un archivo de CRISIS (tiene onset)
    if onset_abs and onset_abs > 0.0:
        
        # --- Definir las 3 ventanas de tiempo ---
        
        # 1. Ventana de PREDICCIÓN (0-200s ANTES)
        poi_pred_start = onset_abs - SEGUNDOS_ANTES_DE_CRISIS
        poi_pred_end = onset_abs # Termina justo cuando empieza el onset
        
        # 2. Ventana DURANTE CRISIS (onset a offset)
        poi_during_start = onset_abs
        poi_during_end = offset_abs
        
        # --- Asignar cada alerta a una categoría ---
        
        # Máscara para Predicciones
        mask_pred = (df_alertas_persistentes['start_s'] >= poi_pred_start) & \
                    (df_alertas_persistentes['start_s'] < poi_pred_end)
        
        # Máscara para Detección Durante Crisis
        mask_during = (df_alertas_persistentes['start_s'] >= poi_during_start) & \
                      (df_alertas_persistentes['start_s'] <= poi_during_end)
        
        # Máscara para Falsas Alarmas (ni predicción, ni durante)
        mask_fp = (~mask_pred) & (~mask_during)
        
        # --- Contar los segundos de cada categoría ---
        segundos_prediccion = df_alertas_persistentes[mask_pred].shape[0] * STEP_SEC
        segundos_deteccion_durante_crisis = df_alertas_persistentes[mask_during].shape[0] * STEP_SEC
        segundos_falsa_alarma = df_alertas_persistentes[mask_fp].shape[0] * STEP_SEC
        
        print(f"-> ¡Éxito (Crisis)! Pred: {segundos_prediccion}s, Durante: {segundos_deteccion_durante_crisis}s, FP: {segundos_falsa_alarma}s")
        
        return {
            "sub": sub, "ses": ses, "run": run, "seg": seg,
            "tipo_archivo": "crisis",
            "onset_abs": onset_abs,
            "segundos_prediccion": segundos_prediccion,
            "segundos_deteccion_durante_crisis": segundos_deteccion_durante_crisis,
            "segundos_falsa_alarma": segundos_falsa_alarma
        }

    # CASO 2: Es un archivo de CONTROL (no tiene onset)
    else:
        # CUALQUIER alerta es una Falsa Alarma
        segundos_falsa_alarma = df_alertas_persistentes.shape[0] * STEP_SEC
        
        print(f"-> Éxito (Control). FP: {segundos_falsa_alarma}s")

        return {
            "sub": sub, "ses": ses, "run": run, "seg": seg,
            "tipo_archivo": "control",
            "onset_abs": None,
            "segundos_prediccion": 0,
            "segundos_deteccion_durante_crisis": 0,
            "segundos_falsa_alarma": segundos_falsa_alarma
        }

# --- Función Principal (main) ---
def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    
    file_pattern = re.compile(r"sub-(\d+)_ses-(\d+)_task-szMonitoring_run-(\d+)_seg-(\d+)\.json")
    
    data_root = Path("../derived_ecg_dataset")
    
    if not data_root.exists():
        print(f"Error: No se encontró la carpeta de datos en: {data_root.resolve()}")
        sys.exit(1)

    print("Iniciando análisis por lotes (con desglose de 3 categorías)...")
    json_files = list(data_root.rglob("sub-*.json"))
    total_archivos_a_procesar = len(json_files)
    print(f"Se encontraron {total_archivos_a_procesar} archivos de segmento para analizar.")
    print(f"Config: win={WIN_SIZE_SEC}s, step={STEP_SEC}s, persist={PERSISTENCE_SEC}s, min_votes={MIN_VOTES_ALERTA}")

    # Lista para el CSV final
    resultados_totales = []
    
    # --- NUEVOS CONTADORES GLOBALES ---
    total_archivos_crisis = 0
    total_archivos_control = 0
    
    archivos_con_prediccion = 0  # Archivos de crisis con al menos 1 predicción
    archivos_con_deteccion = 0 # Archivos de crisis con al menos 1 detección "durante"
    archivos_con_fp = 0        # Archivos (cualquier tipo) con al menos 1 FP
    
    total_segundos_prediccion = 0
    total_segundos_deteccion_durante_crisis = 0
    total_segundos_falsa_alarma = 0
    # ------------------------------------

    for i, json_path in enumerate(json_files):
        
        match = file_pattern.search(json_path.name)
        if not match: continue
            
        sub, ses, run, seg = [int(g) for g in match.groups()]
        
        print(f"\n--- Procesando [{i+1}/{total_archivos_a_procesar}]: sub={sub} ses={ses} run={run} seg={seg} ---")

        try:
            resultado = analizar_archivo(sub, ses, run, seg)
            
            if not resultado: continue 
            
            resultados_totales.append(resultado)
            
            # --- Actualizar contadores globales ---
            total_segundos_prediccion += resultado["segundos_prediccion"]
            total_segundos_deteccion_durante_crisis += resultado["segundos_deteccion_durante_crisis"]
            total_segundos_falsa_alarma += resultado["segundos_falsa_alarma"]

            if resultado["segundos_falsa_alarma"] > 0:
                archivos_con_fp += 1

            if resultado["tipo_archivo"] == "crisis":
                total_archivos_crisis += 1
                if resultado["segundos_prediccion"] > 0:
                    archivos_con_prediccion += 1
                if resultado["segundos_deteccion_durante_crisis"] > 0:
                    archivos_con_deteccion += 1
            
            elif resultado["tipo_archivo"] == "control":
                total_archivos_control += 1
            
            # Imprimir resumen parcial
            print(f"-> Resumen Parcial:")
            print(f"   Archivos de Crisis con Predicción: {archivos_con_prediccion} de {total_archivos_crisis}")
            print(f"   Archivos de Crisis con Detección 'Durante': {archivos_con_deteccion} de {total_archivos_crisis}")
            print(f"   Archivos Totales con Falsas Alarmas: {archivos_con_fp} de {i+1}")

        except Exception as e:
            print(f"ERROR al procesar sub={sub} seg={seg}: {e}")
            resultados_totales.append({
                "sub": sub, "ses": ses, "run": run, "seg": seg,
                "tipo_archivo": "error", "segundos_prediccion": 0, "segundos_deteccion_durante_crisis": 0, "segundos_falsa_alarma": 0
            })

    # --- Reporte Final ---
    print("\n\n--- ANÁLISIS COMPLETO ---")
    
    if not resultados_totales:
        print("No se procesó ningún archivo con éxito.")
    else:
        df_final = pd.DataFrame(resultados_totales)
        csv_path = "resultados_batch_completo.csv"
        df_final.to_csv(csv_path, index=False)
        print(f"\nResultados detallados guardados en: {csv_path}")
        
        # --- NUEVO RESUMEN FINAL ---
        print("\n==================================================================")
        print(f"  RESULTADO FINAL (Nivel Archivo)")
        print(f"==================================================================")
        print(f"  Total Archivos de Crisis Procesados: {total_archivos_crisis}")
        print(f"  Total Archivos de Control Procesados: {total_archivos_control}")
        
        print(f"\n  CATEGORÍA: PREDICCIONES (0-{SEGUNDOS_ANTES_DE_CRISIS}s antes de onset)")
        print(f"    {archivos_con_prediccion} de {total_archivos_crisis} archivos de crisis tuvieron al menos 1 alerta de predicción.")
        
        print(f"\n  CATEGORÍA: DETECCIÓN DURANTE CRISIS (onset a offset)")
        print(f"    {archivos_con_deteccion} de {total_archivos_crisis} archivos de crisis tuvieron al menos 1 alerta 'durante'.")
        
        print(f"\n  CATEGORÍA: FALSAS ALARMAS (fuera de POI o en controles)")
        print(f"    {archivos_con_fp} de {total_archivos_a_procesar} archivos totales tuvieron al menos 1 Falsa Alarma.")
        
        print(f"\n==================================================================")
        print(f"  RESULTADO FINAL (Nivel Duración en Segundos)")
        print(f"==================================================================")
        print(f"  Total Segundos de 'Predicción':               {total_segundos_prediccion} s")
        print(f"  Total Segundos de 'Detección Durante Crisis': {total_segundos_deteccion_durante_crisis} s")
        print(f"  Total Segundos de 'Falsa Alarma':             {total_segundos_falsa_alarma} s")
        print(f"==================================================================")


if __name__ == "__main__":
    main()