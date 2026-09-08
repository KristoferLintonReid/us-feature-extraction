"""PyRadiomics over every feature class and every image filter.

"All PyRadiomics features" means: all seven feature classes (firstorder, shape2D,
GLCM, GLRLM, GLSZM, GLDM, NGTDM) computed on the original image and on every
supported filtered derivative (wavelet, LoG, square, square root, logarithm,
exponential, gradient, LBP2D). In 2D that lands around 1.5k features per ROI.

PyRadiomics is extremely chatty on stderr for benign conditions; its logger is
turned down to ERROR and routed into our run log so real problems still surface.
"""
from __future__ import annotations

import logging

import numpy as np
import SimpleITK as sitk

from ..config import Config
from ..logging_setup import get_logger
from ..roi import RoiView
from .base import Extractor

log = get_logger("pyradiomics")


class PyRadiomicsExtractor(Extractor):
    name = "pyradiomics"
    prefix = "pyrad_"

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        from radiomics import featureextractor, setVerbosity

        setVerbosity(logging.ERROR)
        rad_log = logging.getLogger("radiomics")
        rad_log.handlers.clear()
        rad_log.setLevel(logging.ERROR)
        rad_log.propagate = False

        self.pr = cfg.pyradiomics
        self.extractor = featureextractor.RadiomicsFeatureExtractor(**self._settings())
        self.extractor.enableAllFeatures()
        if self.pr.enable_all_image_types:
            self._enable_image_types()

        enabled = sorted(self.extractor.enabledImagetypes)
        log.info("pyradiomics image types: %s", ", ".join(enabled))

    # ------------------------------------------------------------- configuration

    def _settings(self) -> dict:
        settings: dict = {
            "binWidth": self.pr.bin_width,
            "normalize": self.pr.normalize,
            "normalizeScale": self.pr.normalize_scale,
            "force2D": self.pr.force_2d,
            "force2Ddimension": 0,
            "label": 1,
            # 2D ultrasound has no physical spacing we can trust; unit spacing
            # keeps shape features in pixel units and comparable across images.
            "interpolator": sitk.sitkBSpline,
            "correctMask": True,
            "geometryTolerance": 1e-3,
        }
        if self.pr.resample_pixel_spacing:
            settings["resampledPixelSpacing"] = list(self.pr.resample_pixel_spacing)
        return settings

    def _enable_image_types(self) -> None:
        types: dict[str, dict] = {}
        for name in self.pr.image_types:
            if name == "LoG":
                types["LoG"] = {"sigma": list(self.pr.log_sigma)}
            elif name == "LBP2D":
                types["LBP2D"] = {}
            else:
                types[name] = {}
        self.extractor.enableImageTypes(**types)

    # ------------------------------------------------------------------ support

    def supports(self, view: RoiView) -> tuple[bool, str]:
        n = view.n_voxels
        if n < self.pr.min_roi_voxels:
            return False, f"ROI has {n} voxels, below min_roi_voxels={self.pr.min_roi_voxels}"
        # Texture matrices need at least a 2x2 extent in both directions.
        rows = np.any(view.mask, axis=1).sum()
        cols = np.any(view.mask, axis=0).sum()
        if rows < 2 or cols < 2:
            return False, f"ROI is degenerate ({rows}x{cols} extent)"
        return True, ""

    # ------------------------------------------------------------------ extract

    def extract(self, view: RoiView) -> dict[str, float]:
        image = sitk.GetImageFromArray(view.image.gray.astype(np.float64))
        mask = sitk.GetImageFromArray(view.mask.astype(np.uint8))
        mask.CopyInformation(image)

        result = self.extractor.execute(image, mask, label=1)

        features: dict[str, float] = {}
        for key, value in result.items():
            if key.startswith("diagnostics_"):
                # Kept but namespaced: useful for QC, never confusable with a feature.
                features[f"{self.prefix}diag_{key[len('diagnostics_'):]}"] = _scalar(value, allow_str=True)
                continue
            features[self._pref(key)] = _scalar(value)

        if not features:
            raise RuntimeError("pyradiomics returned no features")
        return features

    def describe(self) -> dict:
        return {
            "extractor": self.name,
            "pyradiomics_version": _version(),
            "image_types": sorted(self.extractor.enabledImagetypes),
            "settings": {k: str(v) for k, v in self.extractor.settings.items()},
        }


def _version() -> str:
    try:
        import radiomics

        return str(radiomics.__version__)
    except Exception:  # noqa: BLE001
        return "unknown"


def _scalar(value, allow_str: bool = False):
    """Coerce a PyRadiomics result to something Parquet can store."""
    if isinstance(value, np.ndarray):
        return float(value.item()) if value.size == 1 else str(value.tolist())
    if isinstance(value, (np.floating, np.integer)):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if allow_str:
        return str(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)
