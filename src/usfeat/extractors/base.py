"""The contract every extractor implements."""
from __future__ import annotations

from abc import ABC, abstractmethod

from ..config import Config
from ..roi import RoiView


class Extractor(ABC):
    """One feature family.

    `extract` returns a flat dict of feature name -> scalar for a single ROI
    view. Raising is fine and expected: the pipeline catches, logs to the error
    ledger, and moves on to the next view.
    """

    name: str = "base"
    #: Prefix applied to every emitted column, keeping families distinguishable
    #: when tables are joined side by side.
    prefix: str = ""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    @abstractmethod
    def extract(self, view: RoiView) -> dict[str, float]:
        ...

    def supports(self, view: RoiView) -> tuple[bool, str]:
        """Whether this view is worth attempting. Returns (ok, reason-if-not)."""
        return True, ""

    def describe(self) -> dict:
        """Provenance recorded in the Parquet metadata."""
        return {"extractor": self.name}

    def close(self) -> None:
        """Release any resources (GPU memory, temp dirs, decrypted payloads)."""

    def _pref(self, key: str) -> str:
        return f"{self.prefix}{key}" if self.prefix else key
