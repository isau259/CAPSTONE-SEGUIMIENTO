import funciones_2 as fn2
import pandas as pd
from pathlib import Path
import re
import sys
import warnings

# ==================================================================
# --- PARÁMETROS GLOBALES DEL ANÁLISIS ---
# (Ajustados a tu última configuración)

WIN_SIZE_SEC = 150       # 150s para CSI/ModCSI
STEP_SEC = 5             # Step de 5s
ROLLING_WINDOW_SEC = 300 # Baseline de 5 min
PERSISTENCE_SEC = 10     # Persistencia de 10s
MIN_VOTES_ALERTA = 3     # 3 de 6 criterios

# --- PARÁMETROS DE ESTE SCRIPT ---
SEGUNDOS_ANTES_DE_CRISIS = 200 # Tu criterio "hasta 200 segundos antes"
# ==================================================================

def analizar_archivo(sub, ses, run, seg):
    """
    Ejecuta el pipeline de análisis completo para un solo archivo
    y devuelve un desglose de Verdaderos Positivos (TP) y Falsos Positivos (FP).
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

    # --- NUEVA LÓGICA: Separar Crisis de Control ---
    
    # CASO 1: Es un archivo de CRISIS (tiene onset)
    if onset_abs and onset_abs > 0.0:
        poi_start = onset_abs - SEGUNDOS_ANTES_DE_CRISIS
        poi_end = offset_abs
        
        # Verdaderos Positivos (TP): Alertas DENTRO del Período de Interés
        mask_tp = (df_alertas_persistentes['start_s'] >= poi_start) & (df_alertas_persistentes['start_s'] <= poi_end)
        df_tp = df_alertas_persistentes[mask_tp]
        segundos_tp = df_tp.shape[0] * STEP_SEC

        # Falsos Positivos (FP): Alertas FUERA del Período de Interés
        df_fp = df_alertas_persistentes[~mask_tp]
        segundos_fp = df_fp.shape[0] * STEP_SEC
        
        print(f"-> ¡Éxito (Crisis)! TP: {segundos_tp}s, FP: {segundos_fp}s")
        
        return {
            "sub": sub, "ses": ses, "run": run, "seg": seg,
            "tipo_archivo": "crisis",
            "onset_abs": onset_abs,
            "segundos_tp": segundos_tp,
            "segundos_fp": segundos_fp
        }

    # CASO 2: Es un archivo de CONTROL (no tiene onset)
    else:
        # Falsos Positivos (FP): CUALQUIER alerta es un FP
        segundos_fp = df_alertas_persistentes.shape[0] * STEP_SEC
        
        print(f"-> Éxito (Control). FP: {segundos_fp}s")

        return {
            "sub": sub, "ses": ses, "run": run, "seg": seg,
            "tipo_archivo": "control",
            "onset_abs": None,
            "segundos_tp": 0, # No puede tener TP
            "segundos_fp": segundos_fp
        }

# --- Función Principal (main) ---
def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    
    file_pattern = re.compile(r"sub-(\d+)_ses-(\d+)_task-szMonitoring_run-(\d+)_seg-(\d+)\.json")
    
    data_root = Path("../derived_ecg_dataset")
    
    if not data_root.exists():
        print(f"Error: No se encontró la carpeta de datos en: {data_root.resolve()}")
        sys.exit(1)

    print("Iniciando análisis por lotes (con conteo de Falsos Positivos)...")
    json_files = list(data_root.rglob("sub-*.json"))
    total_archivos_a_procesar = len(json_files)
    print(f"Se encontraron {total_archivos_a_procesar} archivos de segmento para analizar.")
    print(f"Config: win={WIN_SIZE_SEC}s, step={STEP_SEC}s, persist={PERSISTENCE_SEC}s, min_votes={MIN_VOTES_ALERTA}")

    # Lista para el CSV final
    resultados_totales = []
    
    # --- NUEVOS CONTADORES GLOBALES ---
    total_archivos_crisis = 0
    total_archivos_control = 0
    crisis_con_alerta_tp = 0  # Archivos de crisis con al menos 1 TP
    control_con_alerta_fp = 0 # Archivos de control con al menos 1 FP
    total_segundos_tp = 0
    total_segundos_fp = 0
    # ------------------------------------

    for i, json_path in enumerate(json_files):
        
        match = file_pattern.search(json_path.name)
        if not match: continue
            
        sub, ses, run, seg = [int(g) for g in match.groups()]
        
        print(f"\n--- Procesando [{i+1}/{total_archivos_a_procesar}]: sub={sub} ses={ses} run={run} seg={seg} ---")

        try:
            resultado = analizar_archivo(sub, ses, run, seg)
            
            if not resultado: continue # Si algo raro pasa
            
            # Guardar el resultado individual
            resultados_totales.append(resultado)
            
            # --- Actualizar contadores globales ---
            total_segundos_tp += resultado["segundos_tp"]
            total_segundos_fp += resultado["segundos_fp"]

            if resultado["tipo_archivo"] == "crisis":
                total_archivos_crisis += 1
                if resultado["segundos_tp"] > 0:
                    crisis_con_alerta_tp += 1
            
            elif resultado["tipo_archivo"] == "control":
                total_archivos_control += 1
                if resultado["segundos_fp"] > 0:
                    control_con_alerta_fp += 1
            
            # Imprimir resumen parcial
            print(f"-> Resumen Parcial:")
            print(f"   Crisis detectadas (TP>0): {crisis_con_alerta_tp} de {total_archivos_crisis}")
            print(f"   Controles con Falsas Alarmas (FP>0): {control_con_alerta_fp} de {total_archivos_control}")

        except Exception as e:
            print(f"ERROR al procesar sub={sub} seg={seg}: {e}")
            resultados_totales.append({
                "sub": sub, "ses": ses, "run": run, "seg": seg,
                "tipo_archivo": "error", "segundos_tp": 0, "segundos_fp": 0
            })

    # --- Reporte Final ---
    print("\n\n--- ANÁLISIS COMPLETO ---")
    
    if not resultados_totales:
        print("No se procesó ningún archivo con éxito.")
    else:
        df_final = pd.DataFrame(resultados_totales)
        csv_path = "resultados_batch_alertas_TP_FP.csv"
        df_final.to_csv(csv_path, index=False)
        print(f"\nResultados detallados guardados en: {csv_path}")
        
        # --- NUEVO RESUMEN FINAL ---
        print("\n==================================================================")
        print(f"  RESULTADO FINAL (Nivel Archivo)")
        print(f"==================================================================")
        print(f"  Sensibilidad (Crisis Detectadas):")
        print(f"    {crisis_con_alerta_tp} de {total_archivos_crisis} archivos de crisis tuvieron alertas correctas (TP > 0s).")
        
        print(f"\n  Falsas Alarmas (Controles):")
        print(f"    {control_con_alerta_fp} de {total_archivos_control} archivos de control tuvieron Falsos Positivos (FP > 0s).")
        
        print(f"\n==================================================================")
        print(f"  RESULTADO FINAL (Nivel Duración en Segundos)")
        print(f"==================================================================")
        print(f"  Total Segundos de Alerta Correcta (TP): {total_segundos_tp} s")
        print(f"  Total Segundos de Falsa Alarma (FP):    {total_segundos_fp} s")
        print(f"==================================================================")


if __name__ == "__main__":
    main()