#!/usr/bin/env python3
"""Summarise a finished run into examples/preview/.

Full feature matrices are large (a single ImageNet table over five images is
~4 MB) and they are derived data, so they do not belong in git. What does belong
is enough for someone to see the exact shape of what they will get before they
run anything: every table's dimensions, its real column names, and a genuine
excerpt of its values.

    python tools/make_preview.py --out examples/output --preview examples/preview
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from usfeat.writer import INDEX_COLUMNS, read_provenance  # noqa: E402

N_EXCERPT_ROWS = 3
N_EXCERPT_FEATURES = 6


def summarise(path: Path) -> dict:
    schema = pq.read_schema(path)
    meta = pq.read_metadata(path)
    prov = read_provenance(path)

    names = list(schema.names)
    index_cols = [c for c in names if c in INDEX_COLUMNS]
    meta_cols = [c for c in names if c.startswith("meta_")]
    feat_cols = [c for c in names if c not in index_cols and c not in meta_cols]

    extractor, _, roi = path.stem.partition("__")
    return {
        "file": path.name,
        "extractor": extractor,
        "roi": roi,
        "rows": meta.num_rows,
        "columns": meta.num_columns,
        "n_features": len(feat_cols),
        "n_metadata": len(meta_cols),
        "size_bytes": path.stat().st_size,
        "model": prov.get("model_id") or prov.get("pyradiomics_version")
                 or prov.get("engine") or "",
        "first_features": feat_cols[:4],
        "last_features": feat_cols[-2:],
        "_feat_cols": feat_cols,
        "_index_cols": index_cols,
        "_meta_cols": meta_cols,
    }


def write_excerpt(path: Path, info: dict, dest: Path) -> None:
    """A real slice of the real table: identity + metadata + a few features."""
    keep = (
        [c for c in ("image_id", "source", "channel", "roi", "roi_n_voxels") if c in info["_index_cols"]]
        + info["_meta_cols"][:4]
        + info["_feat_cols"][:N_EXCERPT_FEATURES]
    )
    df = pq.read_table(path, columns=keep).to_pandas().head(N_EXCERPT_ROWS)
    for col in df.columns:
        if pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].round(5)
    df.to_csv(dest, index=False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("examples/output"))
    ap.add_argument("--preview", type=Path, default=Path("examples/preview"))
    args = ap.parse_args()

    features = args.out / "features"
    if not features.is_dir():
        print(f"no features directory at {features}", file=sys.stderr)
        return 1

    excerpt_dir = args.preview / "excerpts"
    excerpt_dir.mkdir(parents=True, exist_ok=True)

    infos = []
    for path in sorted(features.glob("*.parquet")):
        info = summarise(path)
        write_excerpt(path, info, excerpt_dir / f"{path.stem}.csv")
        infos.append(info)

    summary_path = args.out / "logs" / "summary.json"
    run = json.loads(summary_path.read_text()) if summary_path.exists() else {}

    # ---- machine-readable manifest
    manifest = [{k: v for k, v in i.items() if not k.startswith("_")} for i in infos]
    (args.preview / "tables.json").write_text(
        json.dumps({"run": run, "tables": manifest}, indent=2), encoding="utf-8")

    # ---- human-readable index
    by_extractor: dict[str, list[dict]] = {}
    for info in infos:
        by_extractor.setdefault(info["extractor"], []).append(info)

    lines = [
        "# Example output",
        "",
        f"Produced by running the pipeline over {run.get('images_found', '?')} images "
        f"from `sample_data/` with default settings:",
        "",
        "```bash",
        "usfeat extract --data sample_data --out examples/output --limit 5",
        "```",
        "",
        f"- **{len(infos)} feature tables**, "
        f"{sum(i['rows'] for i in infos)} rows total",
        f"- **{run.get('errors_logged', 0)} errors**",
        f"- runtime {run.get('elapsed_seconds', '?')}s on CPU",
        "",
        "The `.parquet` files themselves are not committed (they are derived data, "
        "~25 MB). Regenerate them with the command above. What is committed:",
        "",
        "- `tables.json` — every table's shape, feature count and model id",
        "- `excerpts/*.csv` — the real first "
        f"{N_EXCERPT_ROWS} rows of each table, with identity, metadata and the "
        f"first {N_EXCERPT_FEATURES} feature columns",
        "- `../output/logs/` and `../output/metadata/` — the complete run logs and "
        "the metadata sheets as Parquet",
        "",
        "## Tables, with and without ROI",
        "",
        "`whole` is the no-ROI case: features over the entire frame. The other four "
        "are ROI-specific. Every extractor produces all five.",
        "",
        "| extractor | ROI | rows | features | model |",
        "|---|---|---|---|---|",
    ]
    for extractor in sorted(by_extractor):
        for info in sorted(by_extractor[extractor], key=lambda i: (i["roi"] != "whole", i["roi"])):
            model = info["model"] if len(info["model"]) < 46 else info["model"][:43] + "…"
            lines.append(
                f"| `{extractor}` | `{info['roi']}` | {info['rows']} | "
                f"{info['n_features']:,} | {model} |"
            )

    lines += [
        "",
        "## Why row counts differ",
        "",
        "`whole` covers every image. The ROI tables cover only the segmented subset, "
        "and only where that ROI exists and is large enough for the extractor. That "
        "is expected, and the run summary accounts for every difference.",
        "",
        "## Column layout",
        "",
        "Every table has the same three blocks, in this order:",
        "",
        "1. **identity** — `image_id`, `stem`, `source`, `channel`, `pair_id`, `roi`, "
        "plus ROI geometry (`roi_n_voxels`, `roi_n_components`, `roi_fraction`) and "
        "frame size",
        "2. **`meta_*`** — every column from your spreadsheets",
        "3. **features** — prefixed by family",
        "",
    ]

    example = next((i for i in infos if i["extractor"] == "pyradiomics" and i["roi"] == "lesion"), infos[0])
    lines += [
        f"For example, `{example['file']}` has {example['n_features']:,} feature columns "
        f"beginning `{example['first_features'][0]}` and ending "
        f"`{example['last_features'][-1]}`.",
        "",
        "## Provenance",
        "",
        "Each file also carries its own provenance in the Parquet metadata — model id, "
        "library versions, settings, config fingerprint:",
        "",
        "```python",
        "from usfeat.writer import read_provenance",
        "read_provenance('examples/output/features/dinov3__lesion.parquet')",
        "```",
        "",
    ]

    (args.preview / "README.md").write_text("\n".join(lines), encoding="utf-8")

    total = sum(i["size_bytes"] for i in infos)
    print(f"  {len(infos)} tables summarised ({total / 1e6:.1f} MB of parquet, not committed)")
    print(f"  wrote {args.preview / 'README.md'}")
    print(f"  wrote {args.preview / 'tables.json'}")
    print(f"  wrote {len(infos)} excerpts to {excerpt_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
