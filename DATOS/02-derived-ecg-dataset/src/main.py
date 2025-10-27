import argparse
from pathlib import Path
from .dataset_builder import build_derived_dataset

def main():
    ap = argparse.ArgumentParser(
        description="Construye dataset derivado de ECG a partir de eventos EEG (BIDS)."
    )
    ap.add_argument("--src", type=str, help="Ruta al dataset BIDS original")
    ap.add_argument("--dst", type=str, help="Ruta de salida del dataset derivado")
    ap.add_argument("--include", type=str, nargs="*", default=None,
                    help="Tipos de evento a incluir (case-insensitive)")
    ap.add_argument("--exclude", type=str, nargs="*", default=None,
                    help="Tipos de evento a excluir (case-insensitive)")
    ap.add_argument("--pre", type=float, default=None, help="Segundos pre-ictales")
    ap.add_argument("--post", type=float, default=None, help="Segundos post-ictales")
    ap.add_argument("--dry-run", action="store_true", help="Simula (no guarda archivos)")
    ap.add_argument("--quiet", action="store_true", help="Menos verboso")

    args = ap.parse_args()

    kwargs = {}
    if args.src:  kwargs["src_root"]  = Path(args.src)
    if args.dst:  kwargs["dst_root"]  = Path(args.dst)
    if args.include is not None: kwargs["include_types"] = set(args.include) if len(args.include)>0 else None
    if args.exclude is not None: kwargs["exclude_types"] = set(args.exclude) if len(args.exclude)>0 else None
    if args.pre is not None:  kwargs["pre_sec"]  = args.pre
    if args.post is not None: kwargs["post_sec"] = args.post
    kwargs["dry_run"] = args.dry_run
    kwargs["verbose"] = not args.quiet

    build_derived_dataset(**kwargs)

if __name__ == "__main__":
    main()
