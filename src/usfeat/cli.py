"""Command line entry point.

    usfeat extract --data /data --out /out
    usfeat extract --data /data --out /out --extractors pyradiomics,dinov2 --limit 10
    usfeat inspect --data /data          # what would be processed, without doing it
    usfeat verify  --out /out            # what came out, and what is missing
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .config import ALL_EXTRACTORS, Config
from .logging_setup import setup_logging


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", "-c", type=Path, help="YAML config file")
    parser.add_argument("--data", "-d", type=Path, help="data root (overrides config)")
    parser.add_argument("--out", "-o", type=Path, help="output directory (overrides config)")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging to console")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="usfeat",
        description="Ultrasound feature extraction: TexLab, PyRadiomics, DINOv2/v3, "
                    "BiomedCLIP, SigLIP and ImageNet baselines, whole-image and per-ROI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"usfeat {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    extract = sub.add_parser("extract", help="run the extraction")
    _add_common(extract)
    extract.add_argument(
        "--extractors", "-e",
        help=f"comma-separated subset of: {','.join(ALL_EXTRACTORS)}",
    )
    extract.add_argument("--rois", help="comma-separated ROI subset (default: all present)")
    extract.add_argument("--no-whole", action="store_true",
                         help="skip the whole-image (no ROI) feature tables")
    extract.add_argument("--limit", type=int, help="process at most N images (smoke test)")
    extract.add_argument("--no-resume", action="store_true",
                         help="recompute rows that already exist in the output")
    extract.add_argument("--device", help="torch device: auto|cpu|cuda|mps")
    extract.add_argument("--roi-mode", choices=["crop", "mask", "both"],
                         help="how ROIs are presented to the neural extractors")
    extract.add_argument("--texlab-key-file", type=Path,
                         help="file holding the TexLab payload key")

    inspect = sub.add_parser("inspect", help="show the manifest without extracting")
    _add_common(inspect)

    verify = sub.add_parser("verify", help="report on an existing output directory")
    verify.add_argument("--out", "-o", type=Path, required=True)

    return parser


def _config_from_args(args: argparse.Namespace) -> Config:
    overrides: dict = {}
    if getattr(args, "data", None):
        overrides["data_root"] = args.data
    if getattr(args, "out", None):
        overrides["output_dir"] = args.out
    if getattr(args, "extractors", None):
        overrides["extractors"] = [e.strip() for e in args.extractors.split(",") if e.strip()]
    if getattr(args, "rois", None):
        overrides["rois"] = [r.strip() for r in args.rois.split(",") if r.strip()]
    if getattr(args, "no_whole", False):
        overrides["include_whole_image"] = False
    if getattr(args, "limit", None):
        overrides["limit"] = args.limit
    if getattr(args, "no_resume", False):
        overrides["resume"] = False

    cfg = Config.load(args.config, **overrides)

    if getattr(args, "device", None):
        cfg.deep.device = args.device
    if getattr(args, "roi_mode", None):
        cfg.deep.roi_mode = args.roi_mode
    if getattr(args, "texlab_key_file", None):
        cfg.texlab.key_file = str(args.texlab_key_file)
    return cfg


def cmd_extract(args: argparse.Namespace) -> int:
    cfg = _config_from_args(args)
    setup_logging(cfg.output_dir / "logs", verbose=args.verbose)

    from .pipeline import run

    summary = run(cfg)
    # A run that produced no tables at all is a failure worth a non-zero exit,
    # even though individual failures are not.
    return 0 if summary["tables"] else 1


def cmd_inspect(args: argparse.Namespace) -> int:
    cfg = _config_from_args(args)
    setup_logging(Path(cfg.output_dir) / "logs", verbose=True)

    from .discovery import discover
    from .metadata import load_metadata

    items = discover(cfg)
    store = load_metadata(cfg)

    by_roi: dict[str, int] = {}
    for item in items:
        for roi in item.roi_names(cfg):
            by_roi[roi] = by_roi.get(roi, 0) + 1

    print(f"\n{len(items)} images found under {cfg.data_root}\n")
    print("rows that would be produced, per feature table:")
    for roi, count in sorted(by_roi.items()):
        for extractor in cfg.extractors:
            print(f"  {extractor}__{roi}.parquet    {count} rows")
    print(f"\nmetadata columns to embed: {len(store.columns)}")
    for col in store.columns:
        print(f"  {col}")

    matched = sum(1 for i in items if store.lookup(i.source, i.metadata_key))
    print(f"\nmetadata matched for {matched}/{len(items)} images")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    import json

    out = Path(args.out)
    features = out / "features"
    summary_path = out / "logs" / "summary.json"
    errors_path = out / "logs" / "errors.csv"

    if not features.is_dir():
        print(f"no features directory at {features}", file=sys.stderr)
        return 1

    import pyarrow.parquet as pq

    from .writer import read_provenance

    print(f"\nfeature tables in {features}:\n")
    total = 0
    for path in sorted(features.glob("*.parquet")):
        meta = pq.read_metadata(path)
        prov = read_provenance(path)
        n_meta_cols = sum(1 for f in pq.read_schema(path).names if f.startswith("meta_"))
        origin = (prov.get("model_id")
                  or prov.get("pyradiomics_version")
                  or prov.get("engine")
                  or "-")
        print(f"  {path.name:44s} {meta.num_rows:6d} rows  {meta.num_columns:6d} cols  "
              f"({n_meta_cols} metadata)  {origin}")
        total += meta.num_rows
    print(f"\n  {total} rows across {len(list(features.glob('*.parquet')))} tables")

    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        print(f"\nrun summary: {summary['images_found']} images, "
              f"{summary['errors_logged']} errors, "
              f"finished {summary['finished_at']}")
        if summary.get("extractors_unavailable"):
            print(f"  extractors that did not start: {summary['extractors_unavailable']}")

    if errors_path.exists() and errors_path.stat().st_size > 0:
        import csv

        with errors_path.open() as fh:
            rows = list(csv.DictReader(fh))
        if rows:
            print(f"\n{len(rows)} errors logged. By extractor and type:")
            counts: dict[tuple, int] = {}
            for row in rows:
                key = (row["extractor"] or row["stage"], row["error_type"])
                counts[key] = counts.get(key, 0) + 1
            for (who, what), n in sorted(counts.items(), key=lambda kv: -kv[1]):
                print(f"  {n:5d}  {who:14s} {what}")
            print(f"\nfull detail: {errors_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handlers = {"extract": cmd_extract, "inspect": cmd_inspect, "verify": cmd_verify}
    try:
        return handlers[args.command](args)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 - top-level guard, prints cleanly
        import traceback

        traceback.print_exc()
        print(f"\nfatal: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
