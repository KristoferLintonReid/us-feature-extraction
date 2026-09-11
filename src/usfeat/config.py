"""Configuration: layout of the input tree, and which extractors to run.

Everything a collaborator might need to change lives in a single YAML file.
Defaults match the folder layout agreed with the data provider, so in the
common case the YAML is never edited at all.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml

# ROI mask sub-directories inside images_seg/, in the order they are reported.
DEFAULT_ROI_DIRS: dict[str, str] = {
    "lesion": "lesion",
    "locule": "locule",
    "projection": "projection",
    "solid": "solid",
}

ALL_EXTRACTORS = [
    "pyradiomics",
    "dinov2",
    "dinov3",
    "biomedclip",
    "siglip",
    "imagenet",
]


@dataclass
class LayoutConfig:
    """Where the images and masks live, relative to the data root."""

    grayscale_dir: str = "images_grayscale"
    doppler_dir: str = "images_doppler"
    seg_dir: str = "images_seg"
    seg_image_subdir: str = "img"
    roi_dirs: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_ROI_DIRS))
    image_extensions: list[str] = field(
        default_factory=lambda: [".tiff", ".tif", ".png", ".jpg", ".jpeg", ".bmp"]
    )
    mask_extensions: list[str] = field(
        default_factory=lambda: [".png", ".tif", ".tiff", ".bmp"]
    )


@dataclass
class MetadataConfig:
    """Spreadsheets to embed alongside the features.

    `unsegmented` is keyed on a filename, `segmented` on an integer label that
    matches the numeric filenames under images_seg/. Both are optional: a
    missing sheet degrades to features with no clinical columns, logged as a
    warning rather than an error.
    """

    # Conventional filenames, resolved relative to data_root. Both are looked
    # up leniently: a missing sheet is a warning, not a failure.
    unsegmented_file: str | None = "metadata_unsegmented.xlsx"
    unsegmented_key: str = "File Name"
    segmented_file: str | None = "metadata_segmented.xlsx"
    segmented_key: str = "Label"
    # Columns to carry into every feature table. Empty list = carry all of them.
    carry_columns: list[str] = field(default_factory=list)


@dataclass
class DeepConfig:
    """Shared settings for the neural feature extractors."""

    # How an ROI is presented to the network.
    #   crop    - tight bounding box of the mask, padded, then resized
    #   mask    - full frame with everything outside the mask zeroed
    #   both    - emit both, column-prefixed
    roi_mode: str = "crop"
    bbox_padding: int = 10
    batch_size: int = 8
    device: str = "auto"  # auto | cpu | cuda | mps
    # Emit mean-pooled patch tokens in addition to the CLS/pooled vector.
    include_patch_mean: bool = True
    imagenet_backbones: list[str] = field(
        default_factory=lambda: ["resnet50.a1_in1k", "vit_base_patch16_224.augreg_in21k_ft_in1k"]
    )
    dinov2_model: str = "facebook/dinov2-base"
    # The canonical Meta repo. It is licence-gated, but the weights are baked
    # into the image at build time, so a collaborator never touches the gate.
    # For a build machine without access, the ungated timm re-host carries the
    # same LVD-1689M weights: timm/vit_base_patch16_dinov3.lvd1689m
    dinov3_model: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    biomedclip_model: str = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    siglip_model: str = "google/siglip-base-patch16-224"


@dataclass
class PyRadiomicsConfig:
    """PyRadiomics settings.

    Defaults turn on every feature class and every image filter, which is what
    "all pyradiomics features" means in practice (~1.5k features in 2D).
    """

    bin_width: float = 25.0
    normalize: bool = True
    normalize_scale: float = 100.0
    force_2d: bool = True
    resample_pixel_spacing: list[float] | None = None
    enable_all_image_types: bool = True
    image_types: list[str] = field(
        default_factory=lambda: [
            "Original",
            "Wavelet",
            "LoG",
            "Square",
            "SquareRoot",
            "Logarithm",
            "Exponential",
            "Gradient",
            "LBP2D",
        ]
    )
    log_sigma: list[float] = field(default_factory=lambda: [1.0, 2.0, 3.0])
    # ROIs smaller than this are skipped: texture matrices are meaningless below
    # a few dozen voxels and PyRadiomics will happily emit garbage instead of raising.
    min_roi_voxels: int = 32


@dataclass
class Config:
    data_root: Path = Path("/data")
    output_dir: Path = Path("/out")
    extractors: list[str] = field(default_factory=lambda: list(ALL_EXTRACTORS))
    include_whole_image: bool = True
    rois: list[str] = field(default_factory=lambda: list(DEFAULT_ROI_DIRS))
    layout: LayoutConfig = field(default_factory=LayoutConfig)
    metadata: MetadataConfig = field(default_factory=MetadataConfig)
    deep: DeepConfig = field(default_factory=DeepConfig)
    pyradiomics: PyRadiomicsConfig = field(default_factory=PyRadiomicsConfig)
    limit: int | None = None
    resume: bool = True

    # ---------------------------------------------------------------- loading

    @classmethod
    def load(cls, path: str | Path | None = None, **overrides: Any) -> "Config":
        raw: dict[str, Any] = {}
        if path:
            with open(path, "r", encoding="utf-8") as fh:
                raw = yaml.safe_load(fh) or {}

        nested = {
            "layout": LayoutConfig,
            "metadata": MetadataConfig,
            "deep": DeepConfig,
            "pyradiomics": PyRadiomicsConfig,
        }
        kwargs: dict[str, Any] = {}
        for key, value in raw.items():
            if key in nested:
                kwargs[key] = nested[key](**(value or {}))
            else:
                kwargs[key] = value

        for key, value in overrides.items():
            if value is not None:
                kwargs[key] = value

        cfg = cls(**kwargs)
        cfg.data_root = Path(cfg.data_root)
        cfg.output_dir = Path(cfg.output_dir)
        unknown = set(cfg.extractors) - set(ALL_EXTRACTORS)
        if unknown:
            raise ValueError(
                f"unknown extractor(s): {sorted(unknown)}; valid: {ALL_EXTRACTORS}"
            )
        return cfg

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["data_root"] = str(self.data_root)
        d["output_dir"] = str(self.output_dir)
        return d

    def fingerprint(self) -> str:
        """Stable hash of the settings that affect feature values.

        Recorded in every Parquet file so a table can always be traced back to
        the configuration that produced it.
        """
        relevant = {
            k: v
            for k, v in self.to_dict().items()
            if k not in {"output_dir", "limit", "resume", "data_root"}
        }
        blob = json.dumps(relevant, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]
