#!/usr/bin/env python3
"""Build the sample dataset from MMOTU (OTU_2d), in the provider's folder layout.

MMOTU is a public, Apache-2.0 ovarian tumour ultrasound dataset with lesion
masks -- the closest open analogue to the adnexal data this pipeline targets. It
is used here purely as a plumbing fixture: it proves the pipeline reads the
expected tree, handles grayscale and colour Doppler, produces per-ROI and
whole-image tables, and joins the metadata sheets.

MMOTU ships one mask per image (the lesion). The locule / projection / solid
masks in the sample pack are DERIVED from that lesion mask by morphology and
are anatomically meaningless -- they exist so the four-ROI code path is
exercised end to end. This is stated in the sample pack's own README too.

    python tools/build_sample_data.py --src <mmotu-dir> --out sample_data
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

# MMOTU OTU_2d class ids -> the paper's tumour categories.
MMOTU_CLASSES = {
    0: "chocolate cyst",
    1: "serous cystadenoma",
    2: "teratoma",
    3: "theca cell tumour",
    4: "simple cyst",
    5: "normal ovary",
    6: "mucinous cystadenoma",
    7: "high grade serous",
}
# A crude benign/malignant split, only so the metadata column has realistic values.
MALIGNANT = {7}
BORDERLINE = {3}


def derive_rois(lesion: np.ndarray) -> dict[str, np.ndarray]:
    """Synthesise solid / locule / projection masks from a lesion mask.

    Morphology only -- see the module docstring. Kept deterministic so the
    sample pack rebuilds identically.
    """
    import cv2

    m = lesion.astype(np.uint8)
    k = lambda n: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n, n))  # noqa: E731

    # Solid: the eroded core.
    solid = cv2.erode(m, k(25), iterations=1).astype(bool)

    # Locule: interior far from the boundary, as a distance-transform threshold.
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    peak = float(dist.max())
    locule = (dist > 0.55 * peak) if peak > 0 else np.zeros_like(lesion)

    # Projection: a band just inside the boundary.
    inner = cv2.erode(m, k(9), iterations=1).astype(bool)
    projection = lesion & ~inner

    return {
        "solid": solid,
        "locule": np.asarray(locule, dtype=bool),
        "projection": projection,
    }


def to_gray_rgb(rgb: np.ndarray) -> np.ndarray:
    """Doppler-removed version: luminance, written back as 3-channel."""
    g = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2])
    g = g.round().clip(0, 255).astype(np.uint8)
    return np.stack([g, g, g], axis=-1)


def colourness(rgb: np.ndarray) -> float:
    a = rgb.astype(np.float32)
    return float(np.abs(a[..., 0] - a[..., 1]).mean() + np.abs(a[..., 1] - a[..., 2]).mean())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True,
                    help="directory holding OTU_2d/images and OTU_2d/annotations")
    ap.add_argument("--out", type=Path, default=Path("sample_data"))
    ap.add_argument("--n-segmented", type=int, default=6)
    ap.add_argument("--n-doppler", type=int, default=3)
    ap.add_argument("--n-grayscale", type=int, default=4)
    args = ap.parse_args()

    images_dir = args.src / "OTU_2d" / "images"
    annot_dir = args.src / "OTU_2d" / "annotations"
    if not images_dir.is_dir():
        raise SystemExit(f"not found: {images_dir}")

    labels: dict[str, int] = {}
    for split in ("train_cls", "val_cls"):
        path = args.src / "OTU_2d" / f"{split}.txt"
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    name, cls = line.split()
                    labels[name.rsplit(".", 1)[0]] = int(cls)

    # Only keep images we actually have both halves for.
    available = []
    for img_path in sorted(images_dir.glob("*.JPG")):
        stem = img_path.stem
        mask_path = annot_dir / f"{stem}_binary.PNG"
        if not mask_path.exists():
            continue
        rgb = np.asarray(Image.open(img_path).convert("RGB"))
        mask = np.asarray(Image.open(mask_path).convert("L")) > 0
        if mask.sum() < 500:
            continue
        available.append((stem, img_path, mask_path, rgb, mask, colourness(rgb)))

    if not available:
        raise SystemExit(f"no usable image/mask pairs under {args.src}")
    print(f"{len(available)} usable image/mask pairs available")

    # Colour images become the Doppler examples; flat ones the grayscale examples.
    by_colour = sorted(available, key=lambda r: -r[5])
    doppler_pool = by_colour[: args.n_doppler]
    grayscale_pool = by_colour[-args.n_grayscale:]
    used = {r[0] for r in doppler_pool} | {r[0] for r in grayscale_pool}
    seg_pool = [r for r in available if r[0] not in used][: args.n_segmented]

    out = args.out
    if out.exists():
        shutil.rmtree(out)
    for sub in ("images_grayscale", "images_doppler",
                "images_seg/img", "images_seg/lesion", "images_seg/locule",
                "images_seg/projection", "images_seg/solid"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    unseg_rows, seg_rows = [], []

    # ---- unsegmented: Doppler pairs (colour + Doppler-removed under one stem)
    for i, (stem, _, _, rgb, _, _) in enumerate(doppler_pool):
        name = f"MMOTU-{stem}-I{i:07d}"
        Image.fromarray(rgb).save(out / "images_doppler" / f"{name}.tiff")
        Image.fromarray(to_gray_rgb(rgb)).save(out / "images_grayscale" / f"{name}.tiff")
        cls = labels.get(stem, -1)
        unseg_rows.append(_meta_row(name, stem, cls, "Doppler + Doppler-removed pair"))

    # ---- unsegmented: grayscale only
    offset = len(doppler_pool)
    for i, (stem, _, _, rgb, _, _) in enumerate(grayscale_pool):
        name = f"MMOTU-{stem}-I{offset + i:07d}"
        Image.fromarray(to_gray_rgb(rgb)).save(out / "images_grayscale" / f"{name}.tiff")
        cls = labels.get(stem, -1)
        unseg_rows.append(_meta_row(name, stem, cls, "grayscale only, no Doppler counterpart"))

    # ---- segmented subset, numbered like the provider's images_seg tree
    for label, (stem, _, _, rgb, mask, _) in enumerate(seg_pool):
        Image.fromarray(to_gray_rgb(rgb)).save(out / "images_seg" / "img" / f"{label}.tiff")
        Image.fromarray((mask * 255).astype(np.uint8)).save(
            out / "images_seg" / "lesion" / f"{label}.png")
        for roi, derived in derive_rois(mask).items():
            Image.fromarray((derived * 255).astype(np.uint8)).save(
                out / "images_seg" / roi / f"{label}.png")
        cls = labels.get(stem, -1)
        seg_rows.append({
            "Label": label,
            "Histology": _histology(cls),
            "MMOTU_source_id": stem,
            "MMOTU_class_id": cls,
            "MMOTU_class_name": MMOTU_CLASSES.get(cls, "unknown"),
            "Notes": "lesion mask is real MMOTU; locule/projection/solid are derived, not clinical",
        })

    pd.DataFrame(unseg_rows).to_excel(out / "metadata_unsegmented.xlsx", index=False)
    pd.DataFrame(seg_rows).to_excel(out / "metadata_segmented.xlsx", index=False)

    print(f"\nwrote {out}")
    print(f"  images_grayscale/  {len(list((out / 'images_grayscale').glob('*.tiff')))}")
    print(f"  images_doppler/    {len(list((out / 'images_doppler').glob('*.tiff')))}")
    print(f"  images_seg/img/    {len(list((out / 'images_seg' / 'img').glob('*.tiff')))}")
    for roi in ("lesion", "locule", "projection", "solid"):
        print(f"  images_seg/{roi:11s} {len(list((out / 'images_seg' / roi).glob('*.png')))}")
    return 0


def _histology(cls: int) -> str:
    if cls in MALIGNANT:
        return "malignant"
    if cls in BORDERLINE:
        return "borderline"
    if cls < 0:
        return "No surgery performed / no information"
    return "benign"


def _meta_row(name: str, stem: str, cls: int, note: str) -> dict:
    return {
        "File Name": f"{name}.tiff",
        "CDM ID": f"MMOTU-{stem}",
        "Center": "MMOTU (public)",
        "Iota7_basic.histology": "malignant" if cls in MALIGNANT else "benign",
        "Iota7_basic.extra_colourscore": int(1 + (cls % 4)),
        "MMOTU_class_id": cls,
        "MMOTU_class_name": MMOTU_CLASSES.get(cls, "unknown"),
        "Notes": note,
    }


if __name__ == "__main__":
    raise SystemExit(main())
