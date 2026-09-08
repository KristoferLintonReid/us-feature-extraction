"""Feature extractors. Each family is independent and individually skippable."""
from __future__ import annotations

from ..config import Config
from ..logging_setup import get_logger
from .base import Extractor

log = get_logger("extractors")


def build_extractors(cfg: Config) -> list[Extractor]:
    """Instantiate the requested extractors.

    An extractor that cannot initialise (missing weights, no Octave, no licence
    key) is reported and dropped rather than aborting the run -- the other
    families are still worth having.
    """
    built: list[Extractor] = []
    for name in cfg.extractors:
        try:
            built.append(_build_one(name, cfg))
            log.info("extractor ready: %s", name)
        except Exception as exc:  # noqa: BLE001
            log.error("extractor %s unavailable, skipping: %s: %s",
                      name, type(exc).__name__, exc)
    if not built:
        raise RuntimeError("no extractors could be initialised; nothing to do")
    return built


def _build_one(name: str, cfg: Config) -> Extractor:
    if name == "pyradiomics":
        from .pyradiomics_ext import PyRadiomicsExtractor

        return PyRadiomicsExtractor(cfg)
    if name == "texlab":
        from .texlab import TexLabExtractor

        return TexLabExtractor(cfg)
    if name in ("dinov2", "dinov3", "biomedclip", "siglip", "imagenet"):
        from .deep import build_deep_extractor

        return build_deep_extractor(name, cfg)
    raise ValueError(f"unknown extractor {name!r}")
