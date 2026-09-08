"""Build the (image, mask) views that every extractor is handed.

One `RoiView` per (image, ROI). The "whole" ROI is a real view too, with a mask
covering the entire frame -- that keeps the extractor interface uniform and
means the no-ROI feature sheets come out of exactly the same code path as the
ROI-specific ones.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import Config
from .discovery import Item
from .imaging import LoadedImage, load_mask, n_components
from .logging_setup import get_logger

log = get_logger("roi")


@dataclass
class RoiView:
    item: Item
    roi: str
    image: LoadedImage
    mask: np.ndarray
    mask_path: Path | None
    n_components: int

    @property
    def is_whole(self) -> bool:
        return self.roi == "whole"

    @property
    def n_voxels(self) -> int:
        return int(self.mask.sum())

    def index_fields(self) -> dict[str, object]:
        fields = dict(self.item.index_fields())
        fields.update(
            {
                "roi": self.roi,
                "roi_mask_path": str(self.mask_path) if self.mask_path else "",
                "roi_n_voxels": self.n_voxels,
                "roi_n_components": self.n_components,
                "roi_fraction": round(self.n_voxels / max(1, self.mask.size), 6),
                "image_height": int(self.image.gray.shape[0]),
                "image_width": int(self.image.gray.shape[1]),
                "image_is_colour": bool(self.image.is_colour),
            }
        )
        return fields


def build_views(item: Item, image: LoadedImage, cfg: Config) -> list[RoiView]:
    """All ROI views for one loaded image.

    A mask that fails to load is skipped with an entry in the run log; the other
    ROIs for that image still proceed. Callers record the failure in the ledger.
    """
    views: list[RoiView] = []

    if cfg.include_whole_image:
        whole = np.ones(image.gray.shape, dtype=bool)
        views.append(
            RoiView(
                item=item,
                roi="whole",
                image=image,
                mask=whole,
                mask_path=None,
                n_components=1,
            )
        )

    for roi in cfg.rois:
        path = item.masks.get(roi)
        if path is None:
            continue
        mask = load_mask(path, image.gray.shape)
        if not mask.any():
            log.warning("ROI %s for %s is empty; skipping", roi, item.image_id)
            continue
        views.append(
            RoiView(
                item=item,
                roi=roi,
                image=image,
                mask=mask,
                mask_path=path,
                n_components=n_components(mask),
            )
        )

    return views
