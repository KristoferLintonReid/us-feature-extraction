"""Run logging: a human-readable log plus a machine-readable error ledger.

Two channels on purpose. `run.log` is the narrative (what ran, how long,
what was skipped). `errors.csv` is the ledger you actually triage from --
one row per failed unit of work, with enough context to reproduce it.
"""
from __future__ import annotations

import csv
import logging
import sys
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOGGER_NAME = "usfeat"

_ERROR_FIELDS = [
    "timestamp",
    "stage",
    "extractor",
    "image_id",
    "roi",
    "image_path",
    "mask_path",
    "error_type",
    "error_message",
    "traceback",
]


class ErrorLedger:
    """Append-only CSV of every failure, flushed immediately.

    Flushed per-row so that a container that is killed mid-run still leaves
    a complete record of what had failed up to that point.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._count = 0
        new = not self.path.exists() or self.path.stat().st_size == 0
        self._fh = self.path.open("a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=_ERROR_FIELDS)
        if new:
            self._writer.writeheader()
            self._fh.flush()

    def record(
        self,
        exc: BaseException,
        *,
        stage: str,
        extractor: str = "",
        image_id: str = "",
        roi: str = "",
        image_path: Any = "",
        mask_path: Any = "",
    ) -> None:
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "stage": stage,
            "extractor": extractor,
            "image_id": image_id,
            "roi": roi,
            "image_path": str(image_path or ""),
            "mask_path": str(mask_path or ""),
            "error_type": type(exc).__name__,
            "error_message": str(exc).replace("\n", " ")[:2000],
            "traceback": "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            )[-4000:],
        }
        with self._lock:
            self._writer.writerow(row)
            self._fh.flush()
            self._count += 1

    @property
    def count(self) -> int:
        return self._count

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass


def setup_logging(log_dir: Path, verbose: bool = False) -> logging.Logger:
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-8s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S"
    )

    fh = logging.FileHandler(log_dir / "run.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.DEBUG if verbose else logging.INFO)
    sh.setFormatter(logging.Formatter("%(levelname)-8s %(message)s"))
    logger.addHandler(sh)

    return logger


def get_logger(suffix: str = "") -> logging.Logger:
    return logging.getLogger(f"{LOGGER_NAME}.{suffix}" if suffix else LOGGER_NAME)
