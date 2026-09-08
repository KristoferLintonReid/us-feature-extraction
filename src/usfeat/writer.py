"""Parquet output: one file per (feature family x ROI).

Rows accumulate in memory and are flushed in shards, so a long run over
thousands of images never holds the whole feature matrix at once and a crash
leaves the completed shards intact. Shards are merged into the final table at
the end of the run.

Every file carries the study metadata as columns *and* the run provenance as
Parquet key/value file metadata, so a table is self-describing once it leaves
this machine.
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .logging_setup import get_logger

log = get_logger("writer")

INDEX_COLUMNS = [
    "image_id",
    "stem",
    "source",
    "channel",
    "pair_id",
    "roi",
    "image_path",
    "roi_mask_path",
    "roi_n_voxels",
    "roi_n_components",
    "roi_fraction",
    "image_height",
    "image_width",
    "image_is_colour",
]


class FeatureWriter:
    """Buffered, shard-flushing writer for a single (extractor, roi) table."""

    def __init__(
        self,
        out_dir: Path,
        extractor: str,
        roi: str,
        provenance: dict,
        shard_size: int = 200,
    ):
        self.out_dir = Path(out_dir)
        self.extractor = extractor
        self.roi = roi
        self.provenance = provenance
        self.shard_size = shard_size
        self.name = f"{extractor}__{roi}"
        self.final_path = self.out_dir / f"{self.name}.parquet"
        self._shard_dir = self.out_dir / ".shards" / self.name
        self._rows: list[dict] = []
        self._shard_index = 0
        self.n_rows = 0
        self.seen_ids: set[str] = set()

    # ------------------------------------------------------------------ resume

    def load_existing_ids(self) -> set[str]:
        """Image ids already present, across the final table and any shards.

        Used by --resume so an interrupted run picks up where it stopped
        instead of recomputing hours of features.
        """
        ids: set[str] = set()
        for path in self._existing_parts():
            try:
                col = pq.read_table(path, columns=["image_id"]).column("image_id")
                ids.update(col.to_pylist())
            except Exception as exc:  # noqa: BLE001
                log.warning("could not read ids from %s: %s", path, exc)
        self.seen_ids = ids
        return ids

    def _existing_parts(self) -> list[Path]:
        parts = []
        if self.final_path.exists():
            parts.append(self.final_path)
        if self._shard_dir.is_dir():
            parts.extend(sorted(self._shard_dir.glob("part-*.parquet")))
        return parts

    # ------------------------------------------------------------------- write

    def add(self, row: dict) -> None:
        self._rows.append(row)
        self.n_rows += 1
        self.seen_ids.add(str(row.get("image_id", "")))
        if len(self._rows) >= self.shard_size:
            self._flush_shard()

    def _flush_shard(self) -> None:
        if not self._rows:
            return
        self._shard_dir.mkdir(parents=True, exist_ok=True)
        # Shard names must sort in write order for a deterministic final table.
        while True:
            path = self._shard_dir / f"part-{self._shard_index:05d}.parquet"
            self._shard_index += 1
            if not path.exists():
                break
        _write_table(self._frame(self._rows), path, self.provenance)
        log.debug("flushed %d rows to %s", len(self._rows), path.name)
        self._rows = []

    @staticmethod
    def _frame(rows: list[dict]) -> pd.DataFrame:
        df = pd.DataFrame(rows)
        ordered = [c for c in INDEX_COLUMNS if c in df.columns]
        meta = sorted(c for c in df.columns if c.startswith("meta_"))
        rest = [c for c in df.columns if c not in ordered and c not in meta]
        return df[ordered + meta + rest]

    def finalize(self) -> Path | None:
        """Merge shards (and any pre-existing table) into one Parquet file."""
        self._flush_shard()
        parts = self._existing_parts()
        if not parts:
            log.info("%s: no rows produced, no file written", self.name)
            return None

        frames = []
        for path in parts:
            try:
                frames.append(pq.read_table(path).to_pandas())
            except Exception as exc:  # noqa: BLE001
                log.error("dropping unreadable part %s: %s", path, exc)
        if not frames:
            return None

        df = pd.concat(frames, ignore_index=True, sort=False)
        before = len(df)
        df = df.drop_duplicates(subset=["image_id", "roi"], keep="last")
        if len(df) != before:
            log.info("%s: de-duplicated %d repeated rows", self.name, before - len(df))
        df = df.sort_values(["source", "channel", "stem"], kind="stable").reset_index(drop=True)

        self.out_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.final_path.with_suffix(".parquet.tmp")
        _write_table(self._frame(df.to_dict(orient="records")), tmp, self.provenance)
        tmp.replace(self.final_path)

        if self._shard_dir.is_dir():
            shutil.rmtree(self._shard_dir, ignore_errors=True)

        n_features = len([c for c in df.columns
                          if c not in INDEX_COLUMNS and not c.startswith("meta_")])
        log.info("wrote %s: %d rows x %d features", self.final_path.name, len(df), n_features)
        return self.final_path


def _write_table(df: pd.DataFrame, path: Path, provenance: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    kv = {k.encode(): json.dumps(v, default=str).encode() for k, v in provenance.items()}
    kv[b"written_at"] = datetime.now(timezone.utc).isoformat().encode()
    existing = table.schema.metadata or {}
    table = table.replace_schema_metadata({**existing, **kv})
    pq.write_table(table, path, compression="zstd")


def read_provenance(path: Path) -> dict:
    """Read back the provenance stamped into a feature file."""
    meta = pq.read_schema(path).metadata or {}
    out = {}
    for k, v in meta.items():
        key = k.decode(errors="replace")
        if key.startswith("pandas"):
            continue
        try:
            out[key] = json.loads(v.decode())
        except Exception:  # noqa: BLE001
            out[key] = v.decode(errors="replace")
    return out
