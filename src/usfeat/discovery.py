"""Walk the data root and build the work manifest.

The provider's tree has two disjoint halves:

  images_grayscale/ + images_doppler/   named by accession (MPO-35606-I0000010)
  images_seg/{img,lesion,locule,...}/   named by integer label (0, 1, 2, ...)

The same stem appears in both images_grayscale/ and images_doppler/ for a
Doppler acquisition -- grayscale holds the Doppler-removed version, doppler the
colour one. Those are two distinct images that share a `pair_id`, so downstream
analysis can pivot on acquisition rather than on file.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .config import Config
from .logging_setup import get_logger

log = get_logger("discovery")


@dataclass
class Item:
    """One image, plus whatever masks were found for it."""

    image_id: str
    image_path: Path
    source: str  # "unsegmented" | "segmented"
    channel: str  # "grayscale" | "doppler"
    stem: str
    pair_id: str
    masks: dict[str, Path] = field(default_factory=dict)
    metadata_key: str = ""

    def roi_names(self, cfg: Config) -> list[str]:
        names = ["whole"] if cfg.include_whole_image else []
        names += [r for r in cfg.rois if r in self.masks]
        return names

    def index_fields(self) -> dict[str, str]:
        return {
            "image_id": self.image_id,
            "stem": self.stem,
            "source": self.source,
            "channel": self.channel,
            "pair_id": self.pair_id,
            "image_path": str(self.image_path),
        }


def _index_dir(directory: Path, extensions: list[str]) -> dict[str, Path]:
    """Map filename stem -> path for one directory, case-insensitively."""
    if not directory.is_dir():
        return {}
    exts = {e.lower() for e in extensions}
    found: dict[str, Path] = {}
    for p in sorted(directory.iterdir()):
        if p.name.startswith("."):
            continue
        if p.is_file() and p.suffix.lower() in exts:
            if p.stem in found:
                log.warning(
                    "duplicate stem %r in %s (%s shadows %s)",
                    p.stem, directory, found[p.stem].name, p.name,
                )
                continue
            found[p.stem] = p
    return found


def discover(cfg: Config) -> list[Item]:
    root = cfg.data_root
    if not root.is_dir():
        raise FileNotFoundError(f"data root does not exist: {root}")

    lay = cfg.layout
    items: list[Item] = []

    # ---------------------------------------------------- unsegmented images
    gray = _index_dir(root / lay.grayscale_dir, lay.image_extensions)
    dopp = _index_dir(root / lay.doppler_dir, lay.image_extensions)
    log.info(
        "found %d grayscale and %d doppler images under %s",
        len(gray), len(dopp), root,
    )

    for channel, table in (("grayscale", gray), ("doppler", dopp)):
        for stem, path in table.items():
            items.append(
                Item(
                    image_id=f"{channel}/{stem}",
                    image_path=path,
                    source="unsegmented",
                    channel=channel,
                    stem=stem,
                    pair_id=stem,
                    metadata_key=path.name,
                )
            )

    both = set(gray) & set(dopp)
    if both:
        log.info("%d acquisitions have both a Doppler and a Doppler-removed image", len(both))

    # ------------------------------------------------------ segmented subset
    seg_root = root / lay.seg_dir
    seg_images = _index_dir(seg_root / lay.seg_image_subdir, lay.image_extensions)
    log.info("found %d segmented images under %s", len(seg_images), seg_root)

    roi_tables = {
        roi: _index_dir(seg_root / subdir, lay.mask_extensions)
        for roi, subdir in lay.roi_dirs.items()
        if roi in cfg.rois
    }
    for roi, table in roi_tables.items():
        log.info("  ROI %-11s %d masks", roi, len(table))

    for stem, path in seg_images.items():
        masks = {roi: t[stem] for roi, t in roi_tables.items() if stem in t}
        if not masks:
            log.warning("segmented image %s has no masks in any ROI directory", stem)
        items.append(
            Item(
                image_id=f"seg/{stem}",
                image_path=path,
                source="segmented",
                channel="grayscale",
                stem=stem,
                pair_id=f"seg/{stem}",
                masks=masks,
                metadata_key=stem,
            )
        )

    if cfg.limit:
        # Interleave the two halves so a --limit smoke test still exercises both
        # the mask path and the no-mask path.
        unseg = [i for i in items if i.source == "unsegmented"]
        seg = [i for i in items if i.source == "segmented"]
        half = max(1, cfg.limit // 2)
        items = unseg[:half] + seg[: cfg.limit - min(half, len(unseg))]
        log.info("--limit %d applied: %d items retained", cfg.limit, len(items))

    if not items:
        raise FileNotFoundError(
            f"no images found under {root}. Expected sub-directories "
            f"{lay.grayscale_dir}/, {lay.doppler_dir}/ and/or {lay.seg_dir}/"
        )

    log.info("manifest: %d images, %d with at least one mask",
             len(items), sum(1 for i in items if i.masks))
    return items
