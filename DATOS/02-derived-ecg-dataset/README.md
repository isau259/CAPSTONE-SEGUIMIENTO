# Derived ECG Dataset (from OpenNeuro BIDS)

Este proyecto crea una **base de datos derivada** de **ECG** usando los **eventos de EEG** (BIDS) para:
- Filtrar tipos de crisis (p.ej., descartar `IMPD`).
- Recortar segmentos de ECG por cada evento válido con margen **preictal** y (opcional) **postictal**.
- Guardar cada segmento como archivo independiente (`.npz` + `.json`) y un **índice maestro** (`index_master.csv`).

## Instalaciones previas necesarias:


