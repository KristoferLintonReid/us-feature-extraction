"""Load the provider's spreadsheets and attach them to every feature table.

Two sheets with different keys:
  unsegmented -- keyed on "File Name" (e.g. MPO-35606-I0000010.tiff)
  segmented   -- keyed on "Label" (an integer matching images_seg filenames)

Both are optional. If a sheet is absent or a key does not match, the feature
rows still come out; the clinical columns are simply null, and the mismatch is
counted and reported in the run summary so nothing goes unnoticed.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import Config
from .logging_setup import get_logger

log = get_logger("metadata")

_META_PREFIX = "meta_"


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xlsm", ".xls"):
        return pd.read_excel(path)
    if suffix in (".csv", ".txt"):
        return pd.read_csv(path)
    if suffix in (".tsv",):
        return pd.read_csv(path, sep="\t")
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"unsupported metadata format: {path}")


def _normalise_key(value: object) -> str:
    """Match keys tolerantly: strip, casefold, and drop a file extension.

    'MPO-35606-I0000010.tiff', 'MPO-35606-I0000010' and ' mpo-35606-i0000010 '
    all resolve to the same row.
    """
    s = str(value).strip()
    for ext in (".tiff", ".tif", ".png", ".jpg", ".jpeg", ".bmp"):
        if s.lower().endswith(ext):
            s = s[: -len(ext)]
            break
    # An integer-like label written as "12.0" by Excel must match "12".
    try:
        f = float(s)
        if f.is_integer():
            s = str(int(f))
    except (TypeError, ValueError):
        pass
    return s.casefold()


class MetadataStore:
    """Lookup from an item's metadata key to a dict of prefixed columns."""

    def __init__(self) -> None:
        self._by_source: dict[str, dict[str, dict]] = {}
        self.columns: list[str] = []
        self.tables: dict[str, pd.DataFrame] = {}
        self.miss_counts: dict[str, int] = {}

    def _ingest(
        self, source: str, df: pd.DataFrame, key_column: str, carry: list[str]
    ) -> None:
        if key_column not in df.columns:
            candidates = [c for c in df.columns if c.strip().casefold() == key_column.strip().casefold()]
            if not candidates:
                raise KeyError(
                    f"key column {key_column!r} not in sheet (columns: {list(df.columns)})"
                )
            key_column = candidates[0]

        keep = [c for c in (carry or df.columns) if c in df.columns]
        if key_column not in keep:
            keep = [key_column] + keep

        table = df[keep].copy()
        table.columns = [
            c if c == key_column else f"{_META_PREFIX}{str(c).strip().replace(' ', '_')}"
            for c in table.columns
        ]

        lookup: dict[str, dict] = {}
        duplicates = 0
        for record in table.to_dict(orient="records"):
            key = _normalise_key(record.pop(key_column))
            if key in lookup:
                duplicates += 1
                continue
            lookup[key] = record
        if duplicates:
            log.warning("%s sheet: %d duplicate keys ignored (first wins)", source, duplicates)

        self._by_source[source] = lookup
        self.tables[source] = df
        self.miss_counts.setdefault(source, 0)
        for col in table.columns:
            if col != key_column and col not in self.columns:
                self.columns.append(col)
        log.info("%s metadata: %d rows, %d carried columns", source, len(lookup), len(table.columns) - 1)

    def lookup(self, source: str, key: str) -> dict:
        table = self._by_source.get(source)
        if not table:
            return {}
        record = table.get(_normalise_key(key))
        if record is None:
            self.miss_counts[source] = self.miss_counts.get(source, 0) + 1
            return {}
        return dict(record)

    def blank(self) -> dict:
        return {c: None for c in self.columns}


def load_metadata(cfg: Config) -> MetadataStore:
    store = MetadataStore()
    meta = cfg.metadata

    for source, filename, key in (
        ("unsegmented", meta.unsegmented_file, meta.unsegmented_key),
        ("segmented", meta.segmented_file, meta.segmented_key),
    ):
        if not filename:
            log.info("no %s metadata sheet configured", source)
            continue
        path = Path(filename)
        if not path.is_absolute():
            path = cfg.data_root / path
        if not path.exists():
            log.warning("%s metadata sheet not found: %s -- continuing without it", source, path)
            continue
        try:
            store._ingest(source, _read_table(path), key, meta.carry_columns)
        except Exception as exc:  # noqa: BLE001 - a bad sheet must not sink the run
            log.error("failed to load %s metadata from %s: %s", source, path, exc)

    if not store.columns:
        log.warning("no metadata columns loaded; feature tables will carry no clinical fields")
    return store


def export_metadata(store: MetadataStore, out_dir: Path) -> list[Path]:
    """Write the raw sheets alongside the features, as Parquet."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for source, df in store.tables.items():
        path = out_dir / f"metadata_{source}.parquet"
        df.to_parquet(path, index=False)
        written.append(path)
        log.info("wrote %s (%d rows)", path.name, len(df))
    return written
