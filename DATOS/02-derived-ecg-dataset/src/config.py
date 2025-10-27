from pathlib import Path

# Carpeta del proyecto (donde está "src/")
BASE_DIR = Path(__file__).resolve().parent.parent
# Carpeta "DATOS" (una arriba de 02-derived-ecg-dataset)
DATA_DIR = BASE_DIR.parent

# === Rutas robustas (independientes del CWD) ===
SRC_ROOT = DATA_DIR / "01-openneuro"                 # .../DATOS/01-openneuro
DST_ROOT = BASE_DIR / "datos"                        # .../DATOS/02-derived-ecg-dataset/datos

# === Filtros de eventos ===
INCLUDE_TYPES = None
EXCLUDE_TYPES = {"impd", "bckg"}

# === Recorte (s) ===
PRE_SEC  = 900
POST_SEC = 900

# === BIDS patterns ===
EEG_EVENTS_GLOB   = "**/eeg/*_events.tsv"
ECG_FILE_PATTERN  = "{sub}_{ses}_task-{task}_run-{run}_ecg.edf"

# === Modo ===
DRY_RUN = False
VERBOSE = True
