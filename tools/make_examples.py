#!/usr/bin/env python3
"""Render the figures in examples/figures/.

The point of these is to make one thing unambiguous: what each extractor is
actually handed, with an ROI and without one. Feature tables are opaque; a
picture of the input is not.

    python tools/make_examples.py --data sample_data --out examples/figures
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from usfeat.config import Config  # noqa: E402
from usfeat.discovery import discover  # noqa: E402
from usfeat.imaging import crop_to_mask, load_image, to_rgb_uint8  # noqa: E402
from usfeat.roi import build_views  # noqa: E402

ROI_COLOURS = {
    "lesion": "#4C9BE8",
    "locule": "#F2B441",
    "projection": "#E8604C",
    "solid": "#5FBF77",
}
FIG_BG = "white"


def _overlay(gray: np.ndarray, mask: np.ndarray, colour: str, alpha: float = 0.45) -> np.ndarray:
    rgb = to_rgb_uint8(gray).astype(np.float32) / 255.0
    c = np.array(matplotlib.colors.to_rgb(colour), dtype=np.float32)
    rgb[mask] = (1 - alpha) * rgb[mask] + alpha * c
    return np.clip(rgb, 0, 1)


def _fit_square(rgb: np.ndarray, pad_value: float = 0.09) -> np.ndarray:
    """Letterbox to a square canvas.

    Panels in one figure have wildly different aspect ratios -- a full frame is
    wide, a locule crop is nearly square. Matplotlib then gives each axes a
    different height and the titles stop lining up. Padding to a common square
    keeps the grid aligned without distorting anything.
    """
    a = np.asarray(rgb, dtype=np.float32)
    if a.max() > 1.0:
        a = a / 255.0
    h, w = a.shape[:2]
    side = max(h, w)
    canvas = np.full((side, side, 3), pad_value, dtype=np.float32)
    y0, x0 = (side - h) // 2, (side - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = a[..., :3]
    return canvas


def _style(ax, title: str, subtitle: str = "") -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    if subtitle:
        ax.set_xlabel(subtitle, fontsize=8, color="#555555", labelpad=4)


def figure_roi_inputs(view_map: dict, image, out: Path) -> Path:
    """One row per ROI: the mask in context, the crop, and the masked frame."""
    rois = [r for r in ("lesion", "locule", "projection", "solid") if r in view_map]
    n = len(rois) + 1

    fig, axes = plt.subplots(n, 3, figsize=(9.5, 3.55 * n), facecolor=FIG_BG)
    if n == 1:
        axes = axes[None, :]

    gray = image.gray

    # Row 0: the no-ROI case. All three columns are the same full frame, because
    # that is precisely the point -- "whole" means the extractor sees everything.
    axes[0, 0].imshow(_fit_square(to_rgb_uint8(gray)))
    _style(axes[0, 0], "whole  (no ROI)", "the full frame, as supplied")
    axes[0, 1].imshow(_fit_square(to_rgb_uint8(gray)))
    _style(axes[0, 1], "→ neural extractors", "resized to 224/256 px")
    axes[0, 2].imshow(_fit_square(to_rgb_uint8(gray)))
    _style(axes[0, 2], "→ PyRadiomics", "mask = entire frame")

    for i, roi in enumerate(rois, start=1):
        view = view_map[roi]
        colour = ROI_COLOURS[roi]

        axes[i, 0].imshow(_fit_square(_overlay(gray, view.mask, colour)))
        _style(axes[i, 0], f"{roi}  (ROI mask)",
               f"{view.n_voxels:,} px · {view.n_components} component(s)")

        cropped = crop_to_mask(gray, view.mask, padding=10)
        if cropped is not None:
            axes[i, 1].imshow(_fit_square(to_rgb_uint8(cropped[0])))
        _style(axes[i, 1], "→ roi_mode: crop", "bbox + 10 px, then resized")

        masked = np.array(gray, copy=True)
        masked[~view.mask] = 0
        axes[i, 2].imshow(_fit_square(to_rgb_uint8(masked)))
        _style(axes[i, 2], "→ roi_mode: mask", "outside the ROI zeroed")

    fig.suptitle("What each extractor is handed, per ROI",
                 fontsize=13, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    path = out / "01_roi_inputs.png"
    fig.savefig(path, dpi=110, facecolor=FIG_BG)
    plt.close(fig)
    return path


def figure_all_rois(view_map: dict, image, out: Path) -> Path:
    """All four ROIs over one frame, so the relationships are visible at a glance."""
    rois = [r for r in ("lesion", "locule", "projection", "solid") if r in view_map]
    fig, axes = plt.subplots(1, len(rois) + 2, figsize=(2.5 * (len(rois) + 2), 3.5),
                             facecolor=FIG_BG)
    gray = image.gray

    axes[0].imshow(_fit_square(to_rgb_uint8(gray)))
    _style(axes[0], "original", "no ROI → whole tables")

    combined = to_rgb_uint8(gray).astype(np.float32) / 255.0
    for roi in rois:
        c = np.array(matplotlib.colors.to_rgb(ROI_COLOURS[roi]), dtype=np.float32)
        m = view_map[roi].mask
        combined[m] = 0.6 * combined[m] + 0.4 * c
    axes[1].imshow(_fit_square(np.clip(combined, 0, 1)))
    _style(axes[1], "all ROIs", "each gets its own tables")

    for ax, roi in zip(axes[2:], rois):
        ax.imshow(_fit_square(_overlay(gray, view_map[roi].mask, ROI_COLOURS[roi])))
        _style(ax, roi, f"{view_map[roi].n_voxels:,} px")

    fig.suptitle("One image → five independent sets of feature tables",
                 fontsize=12, fontweight="bold", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = out / "02_all_rois.png"
    fig.savefig(path, dpi=110, facecolor=FIG_BG)
    plt.close(fig)
    return path


def figure_doppler(cfg: Config, out: Path) -> Path | None:
    """A Doppler acquisition and its Doppler-removed twin, side by side."""
    items = {i.image_id: i for i in discover(cfg)}
    pairs = {}
    for item in items.values():
        if item.source == "unsegmented":
            pairs.setdefault(item.pair_id, {})[item.channel] = item
    both = [p for p in pairs.values() if len(p) == 2]
    if not both:
        return None
    pair = both[0]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 4.1), facecolor=FIG_BG)
    for ax, channel in zip(axes, ("doppler", "grayscale")):
        image = load_image(pair[channel].image_path)
        source = image.rgb if image.rgb is not None else image.gray
        ax.imshow(_fit_square(to_rgb_uint8(source)))
        _style(ax, f"channel = {channel}",
               "colour → neural extractors" if channel == "doppler"
               else "Doppler-removed, same acquisition")

    fig.suptitle(f"Same pair_id ({pair['doppler'].pair_id}), two rows, two channels",
                 fontsize=11, fontweight="bold", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = out / "03_doppler_pair.png"
    fig.savefig(path, dpi=110, facecolor=FIG_BG)
    plt.close(fig)
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("sample_data"))
    ap.add_argument("--out", type=Path, default=Path("examples/figures"))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    cfg = Config.load(None, data_root=args.data, output_dir=Path("/tmp/unused"))

    segmented = [i for i in discover(cfg) if i.masks]
    if not segmented:
        print("no segmented images found; cannot draw ROI figures", file=sys.stderr)
        return 1

    # Pick the image with the most ROIs, then the largest lesion -- the clearest one.
    item = max(segmented, key=lambda i: (len(i.masks), max(
        (1,), default=1)))
    image = load_image(item.image_path)
    view_map = {v.roi: v for v in build_views(item, image, cfg)}

    written = [
        figure_roi_inputs(view_map, image, args.out),
        figure_all_rois(view_map, image, args.out),
    ]
    dop = figure_doppler(cfg, args.out)
    if dop:
        written.append(dop)

    for path in written:
        print(f"  wrote {path}  ({path.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
