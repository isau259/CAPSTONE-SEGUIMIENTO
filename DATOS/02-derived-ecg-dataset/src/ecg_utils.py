from pathlib import Path
import json
import numpy as np
import mne
from .config import ECG_FILE_PATTERN

def find_ecg_file(src_root: Path, sub: str, ses: str, task: str, run: str):
    """
    Busca el EDF de ECG correspondiente a un events.tsv (BIDS).
    """
    ecg_dir = src_root / f"sub-{sub}" / f"ses-{ses}" / "ecg"
    if not ecg_dir.exists():
        return None
    candidate = ecg_dir / ECG_FILE_PATTERN.format(sub=f"sub-{sub}",
                                                 ses=f"ses-{ses}",
                                                 task=task,
                                                 run=run)
    return candidate if candidate.exists() else None

def load_ecg_raw(edf_path: Path):
    return mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

def cut_ecg_segment(ecg_raw, onset, duration, pre_sec, post_sec):
    """
    Recorta [onset - pre_sec, onset + duration + post_sec] dentro de los límites.
    Devuelve (x, fs, (t0, t1)).
    """
    fs = float(ecg_raw.info["sfreq"])
    n  = ecg_raw.n_times
    t_total = n / fs

    t0 = max(0.0, onset - pre_sec)
    t1 = min(t_total, onset + duration + post_sec)
    s0 = int(round(t0 * fs))
    s1 = int(round(t1 * fs))

    x = ecg_raw.get_data(picks=[0])[:, s0:s1].astype(float)[0]
    return x, fs, (t0, t1)

def save_segment(dst_dir: Path, meta: dict, ecg: np.ndarray, fs: float):
    dst_dir.mkdir(parents=True, exist_ok=True)
    seg_id = meta["segment_id"]
    np.savez(dst_dir / f"{seg_id}.npz", ecg=ecg, fs=fs)
    with open(dst_dir / f"{seg_id}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
