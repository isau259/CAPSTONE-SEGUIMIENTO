import numpy as np
import pandas as pd
import mne, neurokit2 as nk
from scipy.stats import skew, kurtosis
import json

def listar_funciones(n_col=1):
    import inspect, math, sys, os
    modulo = sys.modules[__name__]
    try:
        ruta_modulo = os.path.abspath(inspect.getsourcefile(modulo))
    except TypeError:
        ruta_modulo = None

    funciones = []
    for name, obj in inspect.getmembers(modulo, inspect.isfunction):
        if name.startswith('_'):
            continue
        if getattr(obj, "__module__", None) != __name__:
            continue
        if ruta_modulo:
            origen = os.path.abspath(inspect.getsourcefile(obj))
            if origen != ruta_modulo:
                continue
        funciones.append(name)

    funciones.sort()
    ancho = math.ceil(len(funciones) / n_col)
    col1, col2 = funciones[:ancho], funciones[ancho:]
    for i in range(ancho):
        print(f"{(col1[i] if i < len(col1) else ''):<25} {(col2[i] if i < len(col2) else '')}")


def crear_variable(sub, ses, run, seg):
    sub_str, ses_str, run_str, seg_str = f"{sub:03d}", f"{ses:02d}", f"{run:02d}", f"{seg:02d}"
    json = f"../derived_ecg_dataset/sub-{sub_str}/ses-{ses_str}/run-{run_str}/sub-{sub_str}_ses-{ses_str}_task-szMonitoring_run-{run_str}_seg-{seg_str}.json"
    npz  = f"../derived_ecg_dataset/sub-{sub_str}/ses-{ses_str}/run-{run_str}/sub-{sub_str}_ses-{ses_str}_task-szMonitoring_run-{run_str}_seg-{seg_str}.npz"

    return npz, json, (sub,ses,run,seg)

# Se eliminan 'matplotlib.pyplot as plt', 'entropy' e 'iqr' si solo se usaban en funciones auxiliares obsoletas.
def procesamiento(datos, win_size=60, step=15, smooth_n=5, overview_ds=10):
    # =======================
    # Parámetros
    # =======================
    WIN_SIZE = int(win_size)
    STEP = int(step)
    SMOOTH_N = int(smooth_n)
    DOWNSAMPLE_OVERVIEW = int(overview_ds)
    
    npz_path, json_path, info = datos

    # =======================
    # Cargar metadata (JSON)
    # =======================
    with open(json_path, "r") as f:
        meta = json.load(f)
    fs = float(meta["fs_hz"])
    onset_abs = float(meta.get("onset_sec", 0.0))
    offset_abs = onset_abs + float(meta.get("duration_sec", 0.0))
    t_cut0 = float(meta.get("cut_start_sec", 0.0))  # inicio absoluto del segmento


    # Eventos -> listas de onsets/offsets
    ONSET_TIME = onset_abs
    OFFSET_TIME = offset_abs
    
    # =======================
    # Cargar ECG desde NPZ
    # =======================
    npz = np.load(npz_path)
    # Detectar vector 1D largo como ECG
    ecg = None
    for k in npz.files:
        arr = np.asarray(npz[k])
        if arr.ndim == 1 and arr.size > 100:
            ecg = arr.astype(float)
            break
    if ecg is None:
        raise RuntimeError(f"No encontré señal 1D en {npz_path}. Claves: {npz.files}")

    # Eje de tiempo (absoluto). Si el npz trae 't'/'time', lo uso y lo paso a absoluto.
    t = None
    for k in npz.files:
        if k.lower() in ("t", "time", "times"):
            t = np.asarray(npz[k], dtype=float)
            break
    if t is None:
        t_abs = np.arange(len(ecg))/fs + t_cut0
    else:
        # si es relativo (arranca en 0), igual desplazar no rompe si ya es absoluto
        t_abs = t + t_cut0

    dur_total = len(ecg) / fs
    print(f"fs={fs:.1f} Hz | duración={dur_total:.1f} s | muestras={len(ecg):,}")
    print(f"onset={onset_abs:.2f}s | offset={offset_abs:.2f}s | cut_start={t_cut0:.2f}s")


    _step = max(DOWNSAMPLE_OVERVIEW, 1)
    x_ds = ecg[::_step]  # o ecg_clean si prefieres
    t_ds = (t_abs if len(t_abs)==len(ecg) else (np.arange(len(ecg))/fs + t_cut0))[::_step]
    
    # =======================
    # Ventanas deslizantes
    # =======================
    def make_segments(signal, fs, win_s=30, step_s=15):
        dur = len(signal) / fs
        segments = []
        starts = np.arange(0, max(dur - win_s, 0) + 1e-9, step_s)
        for start in starts:
            s = int(start * fs); e = int((start + win_s) * fs)
            if e <= len(signal):
                segments.append((start, start + win_s, signal[s:e]))
        return segments

    segments = make_segments(ecg, fs, WIN_SIZE, STEP)
    print(f"Total de ventanas: {len(segments)} (win={WIN_SIZE}s, step={STEP}s)")

    # =======================
    # Extraer features por ventana
    # =======================
    rows = []
    for (start, end, segment) in segments:
        ecg_clean = nk.ecg_clean(segment, sampling_rate=fs)
        signals, info = nk.ecg_peaks(ecg_clean, sampling_rate=fs)
        rpeaks = info.get("ECG_R_Peaks", None)

        # Inicializar métricas en caso de pocos picos R
        if rpeaks is None or len(rpeaks) < 3:
            rows.append({
                "start_s": start, "end_s": end,
                "HR_mean": np.nan, "SDNN_ms": np.nan, "RMSSD_ms": np.nan,
                # Nuevas métricas de alerta
                "HF": np.nan, "SampEn": np.nan, "n_beats": 0,
                "LF": np.nan, "LF/HF": np.nan, "pLF": np.nan, "pHF": np.nan,
                # Columnas estadísticas
                "median_rr": np.nan, "mode_rr": np.nan, "geom_mean_rr": np.nan, "harm_mean_rr": np.nan,
                "std_dev_rr": np.nan, "variance_rr": np.nan, "abs_dev_rr": np.nan, "iqr_rr": np.nan,
                "p25_rr": np.nan, "p75_rr": np.nan, "kurtosis_rr": np.nan, "skewness_rr": np.nan
            })
            continue

        # HR promedio (bpm)
        hr_sig = nk.ecg_rate(rpeaks, sampling_rate=fs, desired_length=len(segment))
        hr_mean = float(np.nanmean(hr_sig))

        # HRV tiempo (SDNN, RMSSD en ms)
        hrv_t = nk.hrv_time(rpeaks, sampling_rate=fs, show=False)
        SDNN = float(hrv_t.get("HRV_SDNN", pd.Series([np.nan])).iloc[0])
        RMSSD = float(hrv_t.get("HRV_RMSSD", pd.Series([np.nan])).iloc[0])

        # Obtener los intervalos R-R en milisegundos
        rr_samples = np.diff(rpeaks)
        rr_ms = rr_samples / fs * 1000.0
        rr_ms = rr_ms[np.isfinite(rr_ms)] # Limpiar valores NaN/Inf

        if len(rr_ms) > 0:
            
            # 1. TENDENCIA CENTRAL
            median_rr = np.median(rr_ms)
            mode_result = pd.Series(rr_ms).mode()
            mode_rr = float(mode_result.iloc[0]) if not mode_result.empty else np.nan
            geom_mean_rr = np.exp(np.mean(np.log(rr_ms))) if np.all(rr_ms > 0) else np.nan
            harm_mean_rr = len(rr_ms) / np.sum(1.0/rr_ms) if np.all(rr_ms > 0) else np.nan
            
            # 2. DISPERSIÓN / FORMA
            std_dev_rr = np.std(rr_ms)
            variance_rr = np.var(rr_ms)
            abs_dev_rr = np.mean(np.abs(rr_ms - np.mean(rr_ms)))
            p25_rr = np.percentile(rr_ms, 25)
            p75_rr = np.percentile(rr_ms, 75)
            iqr_rr = p75_rr - p25_rr
            kurtosis_rr = float(kurtosis(rr_ms))
            skewness_rr = float(skew(rr_ms))
        
        else:
            median_rr, mode_rr, geom_mean_rr, harm_mean_rr = np.nan, np.nan, np.nan, np.nan
            std_dev_rr, variance_rr, abs_dev_rr, iqr_rr = np.nan, np.nan, np.nan, np.nan
            p25_rr, p75_rr, kurtosis_rr, skewness_rr = np.nan, np.nan, np.nan, np.nan

        # 3. HRV Frecuencia (Integración de LF, HF, LF/HF, pLF, pHF)
        try:
            hrv_f = nk.hrv_frequency(rpeaks, sampling_rate=fs, show=False, psd_method="welch", interpolation_rate=4)
        except Exception:
            try:
                hrv_f = nk.hrv_frequency(rpeaks, sampling_rate=fs, show=False, psd_method="lomb", interpolation_rate=4)
            except Exception:
                hrv_f = None
        
        if (hrv_f is not None and not hrv_f.empty):
            HF = float(hrv_f.get("HRV_HF", pd.Series([np.nan])).iloc[0])
            LF = float(hrv_f.get("HRV_LF", pd.Series([np.nan])).iloc[0])
            LFHF = float(hrv_f.get("HRV_LFHF", pd.Series([np.nan])).iloc[0])
            pLF = float(hrv_f.get("HRV_LFn", pd.Series([np.nan])).iloc[0] * 100.0)
            pHF = float(hrv_f.get("HRV_HFn", pd.Series([np.nan])).iloc[0] * 100.0)
        else:
            HF, LF, LFHF, pLF, pHF = np.nan, np.nan, np.nan, np.nan, np.nan

        # 4. Entropía No Lineal (SampEn)
        try:
            hrv_nonlin = nk.hrv_nonlinear(rpeaks, sampling_rate=fs, show=False)
            SampEn = float(hrv_nonlin.get("HRV_SampEn", pd.Series([np.nan])).iloc[0])
        except Exception:
            SampEn = np.nan
            
        rows.append({
            "start_s": start + t_cut0, "end_s": end + t_cut0,
            "HR_mean": hr_mean, "SDNN_ms": SDNN, "RMSSD_ms": RMSSD,
            # Métricas de Alerta
            "HF": HF, "SampEn": SampEn, "n_beats": int(len(rpeaks)),
            "LF": LF, "LF/HF": LFHF, "pLF": pLF, "pHF": pHF,
            # Columnas estadísticas
            "median_rr": median_rr, "mode_rr": mode_rr, "geom_mean_rr": geom_mean_rr, "harm_mean_rr": harm_mean_rr,
            "std_dev_rr": std_dev_rr, "variance_rr": variance_rr, "abs_dev_rr": abs_dev_rr, "iqr_rr": iqr_rr,
            "p25_rr": p25_rr, "p75_rr": p75_rr, "kurtosis_rr": kurtosis_rr, "skewness_rr": skewness_rr
        })

    df = pd.DataFrame(rows).sort_values("start_s").reset_index(drop=True)

    # =======================
    # Guardar señal cruda downsampleada en el df (solo fila 0)
    # =======================
    # IMPORTANTE: Estas columnas son el origen del TypeError, se mantienen con dtype=object
    df["RAW_t"]= pd.Series([None]*len(df), dtype=object)
    df["RAW_ecg"] = pd.Series([None]*len(df), dtype=object)

    if len(df) > 0:
        df.at[0, "RAW_t"]= t_ds
        df.at[0, "RAW_ecg"] = x_ds
    
    df.attrs["onset_abs"]  = onset_abs
    df.attrs["offset_abs"] = offset_abs
    df.attrs["fs"]         = fs
    df.attrs["npz_path"]   = npz_path
    df.attrs["json_path"]  = json_path
    
    # Guarda metadatos para usar después
    df.attrs["sub"] = datos[2][0]    # ej: "001"
    df.attrs["ses"] = datos[2][1]    # ej: "01"
    df.attrs["run"] = datos[2][2]    # ej: "03"
    df.attrs["seg"] = datos[2][3] 

    return df

import numpy as np
import plotly.graph_objs as go

def _smooth_series(y, n):
    n = max(int(n), 1)
    y = np.asarray(y, dtype=float)
    if n == 1 or not np.isfinite(y).any():
        return y
    y_nan = np.where(np.isfinite(y), y, 0.0)
    valid = np.where(np.isfinite(y), 1.0, 0.0)
    kernel = np.ones(n)
    num = np.convolve(y_nan, kernel, mode="same")
    den = np.convolve(valid, kernel, mode="same")
    out = np.full_like(num, np.nan, dtype=float)
    mask = den > 0
    out[mask] = num[mask] / den[mask]
    return out

def viewer_plotly_params_from_df(
    df,
    events=None,
    metrics=("HR_mean","SDNN_ms","RMSSD_ms","HF","RR_entropy"),
    smooth_options=(1,3,5,9),
    default_metric="ALERTA_PERSISTENTE",
    default_smooth=5,
    open_in_browser=False
):
    """
    Usa:
      - métricas por ventana en df[metrics]
      - señal cruda si df tiene columnas 'RAW_t' y 'RAW_ecg' (tomadas de df.loc[0, ...])
    """
    # Detectar señal cruda en df
    has_raw = ("RAW_t" in df.columns) and ("RAW_ecg" in df.columns) and (len(df) > 0) and (df.loc[0, "RAW_t"] is not None)

    # Validar métricas presentes
    metrics = [m for m in metrics if m in df.columns]
    if not metrics and not has_raw:
        raise ValueError("df no contiene métricas ni señal cruda.")

    metrics_all = metrics.copy()
    RAW_KEY = None
    if has_raw:
        RAW_KEY = "ECG_raw"
        metrics_all.append(RAW_KEY)

    if default_smooth not in smooth_options:
        default_smooth = smooth_options[0]
    if default_metric not in metrics_all:
        default_metric = metrics_all[0]

    # Eje X para métricas por ventana
    x_win = df["start_s"].values if "start_s" in df.columns else np.arange(len(df))

    # Precompute data[(m, N)] = (x, y_raw, y_smooth)
    data = {}

    for m in metrics:
        y = df[m].values.astype(float)
        for N in smooth_options:
            data[(m, N)] = (x_win, y, _smooth_series(y, N))

    if has_raw:
        t_raw = df.loc[0, "RAW_t"]
        y_raw = df.loc[0, "RAW_ecg"]
        t_raw = np.asarray(t_raw, dtype=float)
        y_raw = np.asarray(y_raw, dtype=float)
        for N in smooth_options:
            data[(RAW_KEY, N)] = (t_raw, y_raw, _smooth_series(y_raw, N))

    sub  = df.attrs.get("sub", "???")
    ses  = df.attrs.get("ses", "??")
    run  = df.attrs.get("run", "??")
    seg  = df.attrs.get("seg", "??")

    base_title = f"Crisis | Sub {sub} Ses {ses} Run {run} Seg {seg}"

    # Construcción de figura
    fig = go.Figure()
    visibility = {}
    k = 0
    for m in metrics_all:
        for N in smooth_options:
            x_vec, y_raw, y_sm = data[(m, N)]
            vis = (m == default_metric and N == default_smooth)
            fig.add_trace(go.Scatter(x=x_vec, y=y_raw, mode="lines",
                                     name=f"{m} raw", opacity=0.35, visible=vis))
            fig.add_trace(go.Scatter(x=x_vec, y=y_sm,  mode="lines",
                                     name=f"{m} smooth (N={N})", visible=vis))
            visibility[(m, N)] = (k, k+1)
            k += 2

    # Líneas onset/offset
    shapes = []

    if not events:
        on = df.attrs.get("onset_abs", None)
        off = df.attrs.get("offset_abs", None)
        if on is not None and off is not None:
            shapes.append(dict(type="line", x0=on, x1=on, y0=0, y1=1,
                                   xref="x", yref="paper", line=dict(color="red", dash="dash")))
            shapes.append(dict(type="line", x0=off, x1=off, y0=0, y1=1,
                                   xref="x", yref="paper", line=dict(color="orange", dash="dash")))

    if events:
        for ev in events:
            on = ev.get("onset_sec")
            off = on + ev.get("duration_sec", 0) if on is not None else None
            if on is not None:
                shapes.append(dict(type="line", x0=on, x1=on, y0=0, y1=1,
                                   xref="x", yref="paper", line=dict(color="red", dash="dash")))
            if off is not None:
                shapes.append(dict(type="line", x0=off, x1=off, y0=0, y1=1,
                                   xref="x", yref="paper", line=dict(color="orange", dash="dash")))

    fig.update_layout(
        title=dict(
            text=f"{base_title}",
            x=0.5, xanchor="center",
        ),
        margin=dict(t=200),  # deja espacio para los menús y el título
        xaxis_title="Tiempo (s)",
        yaxis_title=f"{default_metric}",
        shapes=shapes,
        xaxis=dict(
            rangeselector=dict(buttons=[
                dict(count=60,  label="1m", step="second", stepmode="backward"),
                dict(count=300, label="5m", step="second", stepmode="backward"),
                dict(step="all")
            ]),
            rangeslider=dict(visible=True),
            type="linear"
        ),
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5)
    )
    # Dropdowns
    # 1) Métrica
    metric_buttons = []
    total_traces = 2 * len(metrics_all) * len(smooth_options)
    for m in metrics_all:
        vis_array = [False] * total_traces
        i_raw, i_sm = visibility[(m, default_smooth)]
        vis_array[i_raw] = True
        vis_array[i_sm]  = True
        metric_buttons.append(dict(
            label=m,
            method="update",
            args=[
                {"visible": vis_array},
                {
                    "title": {
                        "text": f"{base_title}",
                        "x": 0.5, "xanchor": "center"
                    },
                    "yaxis": {"title": m}
                }
            ]
        ))
    # 2) Smooth N
    smooth_buttons = []
    for N in smooth_options:
        vis_array = [False] * total_traces
        i_raw, i_sm = visibility[(default_metric, N)]
        vis_array[i_raw] = True
        vis_array[i_sm]  = True
        smooth_buttons.append(dict(
            label=f"N={N}",
            method="update",
            args=[
                {"visible": vis_array},
                {
                    "title": {
                        "text": f"{base_title}",
                        "x": 0.5, "xanchor": "center"
                    }
                }
            ]
        ))

    fig.update_layout(
        updatemenus=[
            dict(type="dropdown", direction="down", x=0.0,  y=1.2,
                 buttons=metric_buttons, showactive=True, xanchor="left"),
            dict(type="dropdown", direction="down", x=0.25, y=1.2,
                 buttons=smooth_buttons, showactive=True, xanchor="left")
        ],
        annotations=[
            dict(text="Métrica:", x=0.0,  y=1.28, xref="paper", yref="paper", showarrow=False),
            dict(text="Smooth:",  x=0.25, y=1.28, xref="paper", yref="paper", showarrow=False),
        ]
    )

    if open_in_browser:
        fig.show(renderer="browser")
    else:
        fig.show()

import pandas as pd
import numpy as np

def calcular_baseline_y_alertas(df, baseline_duration_sec=1800, persistence_sec=60, min_votes=4):
    """
    Calcula la línea de base, aplica los 6 criterios de alerta y filtra por:
    1. Criterio de Consenso (3 o más alertas individuales activas).
    2. Criterio de Persistencia Temporal (60 segundos).
    """
    if df.empty:
        print("El DataFrame está vacío. No se puede calcular el baseline.")
        return df

    # 1. ESTABLECER LA LÍNEA DE BASE (BASELINE)
    t0 = float(df['start_s'].min())
    onset = df.attrs.get('onset_abs', None)
    baseline_df = df[df['start_s'] < t0 + 300].copy()
    if baseline_df.empty:
        print(f"Advertencia: No hay suficientes datos para un baseline de {baseline_duration_sec} segundos.")
        baseline_df = df.head(120)

    # Excluir columnas RAW_t y RAW_ecg
    cols_to_agg = [col for col in baseline_df.columns if not col.startswith('RAW_')]
    baseline_stats = baseline_df[cols_to_agg].agg(['mean', 'std']).T
    
    # 2. INICIALIZAR COLUMNAS DE ALERTA INDIVIDUALES Y FINALES
    # 6 Criterios Individuales
    df['ALERTA_HR_2SD'] = False
    df['ALERTA_HR_REL_20PCT'] = False
    df['ALERTA_SDNN_REL_DROP'] = False
    df['ALERTA_RMSSD_REL_DROP'] = False
    df['ALERTA_LFHF_REL_RISE'] = False 
    df['ALERTA_SAMPEN_REL_DROP'] = False 
    
    # Nuevas columnas de resultado
    df['ALERTA_CONSENSO'] = False      # Alerta filtrada por votación (3/6)
    df['ALERTA_PERSISTENTE'] = False # Alerta filtrada por persistencia (60s)
    
    
    # 3. APLICAR LÓGICA DE ALERTA INDIVIDUAL
    for metric in ['HR_mean', 'SDNN_ms', 'RMSSD_ms', 'LF/HF', 'SampEn']:
        if metric not in baseline_stats.index:
             continue
             
        mu = baseline_stats.loc[metric, 'mean']
        sigma = baseline_stats.loc[metric, 'std']
        
        for i in df.index:
            current_val = df.loc[i, metric]
            
            if np.isfinite(current_val) and np.isfinite(mu):
                
                # Criterios basados en HR_mean
                if metric == 'HR_mean':
                    if current_val > (mu + 2 * sigma):
                        df.loc[i, 'ALERTA_HR_2SD'] = True
                    if current_val > (mu * 1.20):
                        df.loc[i, 'ALERTA_HR_REL_20PCT'] = True

                # Criterios basados en SDNN_ms
                if metric == 'SDNN_ms' and current_val < (mu * 0.80):
                    df.loc[i, 'ALERTA_SDNN_REL_DROP'] = True
                    
                # Criterios basados en RMSSD_ms
                if metric == 'RMSSD_ms' and current_val < (mu * 0.70):
                    df.loc[i, 'ALERTA_RMSSD_REL_DROP'] = True
                    
                # Criterios basados en LF/HF
                if metric == 'LF/HF' and current_val > (mu * 1.40):
                    df.loc[i, 'ALERTA_LFHF_REL_RISE'] = True

                # Criterios basados en SampEn
                if metric == 'SampEn' and current_val < (mu * 0.80):
                    df.loc[i, 'ALERTA_SAMPEN_REL_DROP'] = True
                
    
    # 4. APLICAR CRITERIO DE CONSENSO (VOTACIÓN 3/6)
    
    # Definir las 6 columnas de los predictores individuales
    individual_alerts = ['ALERTA_HR_2SD', 'ALERTA_HR_REL_20PCT', 'ALERTA_SDNN_REL_DROP', 
                         'ALERTA_RMSSD_REL_DROP', 'ALERTA_LFHF_REL_RISE', 'ALERTA_SAMPEN_REL_DROP']
    
    # Contar cuántas alertas son True (se suman los True/False, donde True=1)
    df['Contador_Alertas'] = df[individual_alerts].sum(axis=1)
    
    # La alerta de consenso se activa si 3 o más predictores están activos
    MIN_VOTES = min_votes
    df['ALERTA_CONSENSO'] = (df['Contador_Alertas'] >= MIN_VOTES)


    # 5. APLICAR CRITERIO DE PERSISTENCIA (FILTRO TEMPORAL)
    
    # Calcular cuántas ventanas equivalen a la duración de persistencia (usa ALERTA_CONSENSO)
    step_sec = df['start_s'].diff().mean() 
    if pd.isna(step_sec) or step_sec == 0:
        # Asume que step es la mitad del win_size (ej. 30s si win=60s)
        step_sec = (df['end_s'].iloc[0] - df['start_s'].iloc[0]) * 0.25 # Usa step=15s (si win=60s, step=15s)
        
    windows_needed = max(1, int(np.ceil(persistence_sec / step_sec)))
    
    # Aplicar un Rolling Sum sobre la nueva ALERTA_CONSENSO
    rolling_sum = df['ALERTA_CONSENSO'].rolling(
        window=windows_needed, 
        min_periods=windows_needed
    ).sum()

    # Si la suma es igual al número de ventanas necesarias, la alerta es persistente.
    df['ALERTA_PERSISTENTE'] = (rolling_sum == windows_needed)
    
    print(f"Consenso: Alerta activa si {MIN_VOTES} o más de 6 criterios se cumplen.")
    print(f"Persistencia: {persistence_sec} s (equivale a {windows_needed} ventanas consecutivas).")
    
    # Eliminar la columna auxiliar de conteo si no se necesita
    df.drop(columns=['Contador_Alertas'], inplace=True, errors='ignore')
    
    return df