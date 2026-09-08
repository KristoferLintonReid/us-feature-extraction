"""The run loop: for each image, for each ROI, for each extractor.

Failure is expected and contained. A single bad image, a mask that will not
load, an ROI too small for texture analysis, a model that will not initialise --
none of these stop the run. Each is recorded in the error ledger with enough
context to reproduce it, and the loop continues. What you get at the end is a
complete set of feature tables plus an honest account of what is missing from
them and why.
"""
from __future__ import annotations

import json
import platform
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .config import Config
from .discovery import Item, discover
from .extractors import build_extractors
from .extractors.base import Extractor
from .imaging import load_image
from .logging_setup import ErrorLedger, get_logger
from .metadata import MetadataStore, export_metadata, load_metadata
from .roi import build_views
from .writer import FeatureWriter

log = get_logger("pipeline")


class Pipeline:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.features_dir = cfg.output_dir / "features"
        self.metadata_dir = cfg.output_dir / "metadata"
        self.logs_dir = cfg.output_dir / "logs"
        for d in (self.features_dir, self.metadata_dir, self.logs_dir):
            d.mkdir(parents=True, exist_ok=True)

        self.ledger = ErrorLedger(self.logs_dir / "errors.csv")
        self.writers: dict[tuple[str, str], FeatureWriter] = {}
        self.stats: Counter = Counter()
        self.skips: Counter = Counter()
        self.started = time.time()

    # ------------------------------------------------------------------- run

    def run(self) -> dict:
        cfg = self.cfg
        log.info("usfeat %s | data_root=%s | output=%s", __version__, cfg.data_root, cfg.output_dir)
        log.info("extractors requested: %s", ", ".join(cfg.extractors))

        items = discover(cfg)
        store = load_metadata(cfg)
        export_metadata(store, self.metadata_dir)

        extractors = build_extractors(cfg)
        self.stats["extractors_ready"] = len(extractors)

        try:
            for n, item in enumerate(items, start=1):
                self._process_item(item, extractors, store, n, len(items))
        finally:
            for extractor in extractors:
                try:
                    extractor.close()
                except Exception as exc:  # noqa: BLE001
                    log.warning("closing %s: %s", extractor.name, exc)

        written = self._finalize()
        summary = self._summarise(items, extractors, store, written)
        self.ledger.close()
        return summary

    # ---------------------------------------------------------------- per item

    def _process_item(
        self,
        item: Item,
        extractors: list[Extractor],
        store: MetadataStore,
        n: int,
        total: int,
    ) -> None:
        log.info("[%d/%d] %s", n, total, item.image_id)
        self.stats["images_seen"] += 1

        try:
            image = load_image(item.image_path)
        except Exception as exc:  # noqa: BLE001
            self.ledger.record(exc, stage="load_image", image_id=item.image_id,
                               image_path=item.image_path)
            log.error("  could not load image: %s: %s", type(exc).__name__, exc)
            self.stats["images_failed"] += 1
            return

        try:
            views = build_views(item, image, self.cfg)
        except Exception as exc:  # noqa: BLE001
            self.ledger.record(exc, stage="build_rois", image_id=item.image_id,
                               image_path=item.image_path)
            log.error("  could not build ROI views: %s", exc)
            self.stats["images_failed"] += 1
            return

        if not views:
            log.warning("  no ROI views for %s", item.image_id)
            return

        meta = store.blank()
        meta.update(store.lookup(item.source, item.metadata_key))
        base = item.index_fields()

        for view in views:
            row_index = {**base, **view.index_fields(), **meta}
            for extractor in extractors:
                self._run_one(extractor, view, row_index)

        self.stats["images_done"] += 1

    def _run_one(self, extractor: Extractor, view, row_index: dict) -> None:
        key = (extractor.name, view.roi)
        writer = self._writer(extractor, view.roi)

        if self.cfg.resume and view.item.image_id in writer.seen_ids:
            self.skips[f"{extractor.name}:already-done"] += 1
            return

        ok, reason = extractor.supports(view)
        if not ok:
            log.debug("  %-11s %-11s skipped: %s", extractor.name, view.roi, reason)
            self.skips[f"{extractor.name}:{reason.split('(')[0].strip()}"] += 1
            self.stats[f"skipped:{extractor.name}"] += 1
            return

        t0 = time.time()
        try:
            features = extractor.extract(view)
        except Exception as exc:  # noqa: BLE001 - containment is the whole point
            self.ledger.record(
                exc,
                stage="extract",
                extractor=extractor.name,
                image_id=view.item.image_id,
                roi=view.roi,
                image_path=view.item.image_path,
                mask_path=view.mask_path or "",
            )
            log.error("  %-11s %-11s FAILED: %s: %s",
                      extractor.name, view.roi, type(exc).__name__, str(exc)[:200])
            self.stats[f"failed:{extractor.name}"] += 1
            return

        writer.add({**row_index, **features})
        elapsed = time.time() - t0
        self.stats[f"rows:{extractor.name}"] += 1
        log.debug("  %-11s %-11s %d features in %.2fs",
                  extractor.name, view.roi, len(features), elapsed)

    def _writer(self, extractor: Extractor, roi: str) -> FeatureWriter:
        key = (extractor.name, roi)
        if key not in self.writers:
            provenance = {
                "usfeat_version": __version__,
                "config_fingerprint": self.cfg.fingerprint(),
                "roi": roi,
                "data_root": str(self.cfg.data_root),
                "python": platform.python_version(),
                "platform": platform.platform(),
                **extractor.describe(),
            }
            writer = FeatureWriter(self.features_dir, extractor.name, roi, provenance)
            if self.cfg.resume:
                existing = writer.load_existing_ids()
                if existing:
                    log.info("resume: %s already has %d rows", writer.name, len(existing))
            self.writers[key] = writer
        return self.writers[key]

    # --------------------------------------------------------------- finishing

    def _finalize(self) -> list[Path]:
        written = []
        for writer in self.writers.values():
            try:
                path = writer.finalize()
                if path:
                    written.append(path)
            except Exception as exc:  # noqa: BLE001
                self.ledger.record(exc, stage="write", extractor=writer.extractor, roi=writer.roi)
                log.error("failed to finalize %s: %s", writer.name, exc)
        return written

    def _summarise(
        self,
        items: list[Item],
        extractors: list[Extractor],
        store: MetadataStore,
        written: list[Path],
    ) -> dict:
        elapsed = time.time() - self.started
        summary = {
            "usfeat_version": __version__,
            "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "elapsed_seconds": round(elapsed, 1),
            "data_root": str(self.cfg.data_root),
            "output_dir": str(self.cfg.output_dir),
            "config_fingerprint": self.cfg.fingerprint(),
            "extractors_requested": list(self.cfg.extractors),
            "extractors_ready": [e.name for e in extractors],
            "extractors_unavailable": [
                e for e in self.cfg.extractors if e not in {x.name for x in extractors}
            ],
            "images_found": len(items),
            "images_with_masks": sum(1 for i in items if i.masks),
            "counts": dict(self.stats),
            "skips": dict(self.skips),
            "errors_logged": self.ledger.count,
            "metadata_unmatched": dict(store.miss_counts),
            "tables": [
                {"file": p.name, "rows": _row_count(p)} for p in sorted(written)
            ],
        }

        path = self.logs_dir / "summary.json"
        path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

        log.info("=" * 68)
        log.info("finished in %s", _hms(elapsed))
        log.info("images: %d found, %d processed, %d failed to load",
                 len(items), self.stats["images_done"], self.stats["images_failed"])
        for extractor in extractors:
            log.info("  %-12s %6d rows  %4d failed  %4d skipped",
                     extractor.name,
                     self.stats.get(f"rows:{extractor.name}", 0),
                     self.stats.get(f"failed:{extractor.name}", 0),
                     self.stats.get(f"skipped:{extractor.name}", 0))
        if summary["extractors_unavailable"]:
            log.warning("extractors that could not start: %s",
                        ", ".join(summary["extractors_unavailable"]))
        for source, misses in store.miss_counts.items():
            if misses:
                log.warning("%d images had no %s metadata row", misses, source)
        log.info("%d feature tables written to %s", len(written), self.features_dir)
        if self.ledger.count:
            log.warning("%d errors logged to %s", self.ledger.count, self.ledger.path)
        else:
            log.info("no errors logged")
        log.info("summary: %s", path)
        log.info("=" * 68)
        return summary


def _row_count(path: Path) -> int:
    try:
        import pyarrow.parquet as pq

        return pq.read_metadata(path).num_rows
    except Exception:  # noqa: BLE001
        return -1


def _hms(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:d}h{m:02d}m{s:02d}s" if h else f"{m:d}m{s:02d}s"


def run(cfg: Config) -> dict:
    return Pipeline(cfg).run()
