from pathlib import Path
import pandas as pd
from .config import (
    SRC_ROOT, DST_ROOT, EEG_EVENTS_GLOB,
    INCLUDE_TYPES, EXCLUDE_TYPES, PRE_SEC, POST_SEC,
    DRY_RUN, VERBOSE
)
from .events_utils import bids_from_fname, load_events_tsv, filter_events
from .ecg_utils import find_ecg_file, load_ecg_raw, cut_ecg_segment, save_segment

def build_derived_dataset(
    src_root: Path = SRC_ROOT,
    dst_root: Path = DST_ROOT,
    include_types = INCLUDE_TYPES,
    exclude_types = EXCLUDE_TYPES,
    pre_sec: float = PRE_SEC,
    post_sec: float = POST_SEC,
    dry_run: bool = DRY_RUN,
    verbose: bool = VERBOSE
) -> pd.DataFrame:

    dst_root.mkdir(parents=True, exist_ok=True)
    index_rows = []

    eeg_tsvs = list(src_root.glob(EEG_EVENTS_GLOB))
    if verbose:
        print(f"Encontrados {len(eeg_tsvs)} events.tsv de EEG")

    for tsv in sorted(eeg_tsvs):
        bids = bids_from_fname(tsv)
        if not bids:
            if verbose: print(f"[skip] No BIDS parse: {tsv}")
            continue

        sub, ses, task, run = bids["sub"], bids["ses"], bids["task"], bids["run"]

        try:
            df_ev = load_events_tsv(tsv)
        except Exception as e:
            if verbose: print(f"[error] {tsv.name}: {e}")
            continue

        df_ev = filter_events(df_ev, include=include_types, exclude=exclude_types)
        if df_ev.empty:
            if verbose: print(f"[info] Sin eventos útiles: {tsv.name}")
            continue

        ecg_file = find_ecg_file(src_root, sub, ses, task, run)
        if ecg_file is None:
            if verbose: print(f"[warn] ECG no encontrado p/ {tsv.name}")
            continue

        if verbose:
            print(f"[OK] {tsv.name} → ECG: {ecg_file.name} (eventos: {len(df_ev)})")

        ecg_raw = load_ecg_raw(ecg_file)
        dst_dir = dst_root / f"sub-{sub}" / f"ses-{ses}" / f"run-{run}"

        for i, r in df_ev.iterrows():
            onset, duration = float(r["onset_sec"]), float(r["duration_sec"])
            etype = str(r.get("eventType", "unknown"))

            x, fs, (t0, t1) = cut_ecg_segment(ecg_raw, onset, duration, pre_sec, post_sec)

            seg_id = f"sub-{sub}_ses-{ses}_task-{task}_run-{run}_seg-{i:02d}"
            meta = {
                "segment_id": seg_id,
                "subject": sub,
                "session": ses,
                "task": task,
                "run": run,
                "source_ecg": str(ecg_file),
                "source_events": str(tsv),
                "event_index": int(i),
                "event_type": etype,
                "onset_sec": onset,
                "duration_sec": duration,
                "cut_start_sec": t0,
                "cut_end_sec": t1,
                "pre_sec": float(pre_sec),
                "post_sec": float(post_sec),
                "n_samples": int(len(x)),
                "fs_hz": float(fs),
            }

            if not dry_run:
                save_segment(dst_dir, meta, x, fs)

            row = meta.copy()
            row["dst_npz"]  = str((dst_dir / f"{seg_id}.npz").resolve())
            row["dst_json"] = str((dst_dir / f"{seg_id}.json").resolve())
            index_rows.append(row)

    df_index = pd.DataFrame(index_rows).sort_values(
        ["subject","session","run","event_index"]
    ).reset_index(drop=True)

    if not dry_run:
        df_index.to_csv(DST_ROOT / "index_master.csv", index=False)

    if verbose:
        print(f"\nListo. Segmentos: {len(df_index)}")
        if not dry_run:
            print(f"Índice maestro: {DST_ROOT / 'index_master.csv'}")

    return df_index
