from pathlib import Path
import re
import pandas as pd

def bids_from_fname(fname: Path):
    """
    Extrae sub, ses, task, run desde un nombre BIDS típico.
    Ej: sub-011_ses-01_task-szMonitoring_run-05_eeg.tsv
    """
    stem = fname.stem
    pat = (r"sub-(?P<sub>[^_]+)_"
           r"ses-(?P<ses>[^_]+)_"
           r"task-(?P<task>[^_]+)_"
           r"run-(?P<run>[^_]+)_")
    m = re.search(pat, stem)
    return m.groupdict() if m else None

def load_events_tsv(tsv_path: Path) -> pd.DataFrame:
    """
    Lee events.tsv de EEG y normaliza a: onset_sec, duration_sec, eventType.
    """
    df = pd.read_csv(tsv_path, sep="\t")
    col_on = next((c for c in df.columns if c.lower() in {"onset","onset_sec"}), None)
    col_du = next((c for c in df.columns if c.lower() in {"duration","duration_sec"}), None)
    col_ty = next((c for c in df.columns if c.lower() in {"eventtype","trial_type","event_type","type"}), None)
    if col_on is None or col_du is None:
        raise ValueError(f"Faltan columnas onset/duration en {tsv_path}")
    out = pd.DataFrame({
        "onset_sec":    df[col_on].astype(float),
        "duration_sec": df[col_du].astype(float),
        "eventType":    df[col_ty].astype(str) if col_ty else "unknown",
    })
    return out

def filter_events(df: pd.DataFrame, include=None, exclude=None) -> pd.DataFrame:
    """
    Filtro por include/exclude (case-insensitive).
    """
    if "eventType" not in df.columns:
        return df
    s = df["eventType"].fillna("").astype(str).str.lower()
    keep = pd.Series(True, index=df.index)
    if include:
        inc = {x.lower() for x in include}
        keep &= s.isin(inc)
    if exclude:
        exc = {x.lower() for x in exclude}
        keep &= ~s.isin(exc)
    return df[keep].reset_index(drop=True)
