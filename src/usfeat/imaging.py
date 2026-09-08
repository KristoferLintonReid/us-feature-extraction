"""Image and mask loading, and the ROI presentations the extractors consume.

Ultrasound TIFFs arrive in inconsistent shapes: 8-bit grayscale, RGB where all
three channels are identical, RGB with genuine colour (Doppler), occasionally
with an alpha channel or a singleton page dimension. Everything is funnelled
through `load_image` into a predictable (H, W) grayscale array plus, when the
source was colour, the original (H, W, 3) RGB.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .logging_setup import get_logger

log = get_logger("imaging")

Image.MAX_IMAGE_PIXELS = None  # clinical frames are small; the bomb guard only false-positives


@dataclass
class LoadedImage:
    gray: np.ndarray          # (H, W) float32, native intensity range preserved
    rgb: np.ndarray | None    # (H, W, 3) uint8 when the source carried real colour
    path: Path
    is_colour: bool


def _squeeze_pages(arr: np.ndarray) -> np.ndarray:
    """Drop singleton leading axes from multi-page or 3D TIFFs."""
    while arr.ndim > 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 4:  # (pages, H, W, C) with real pages -- take the first
        log.debug("multi-page image with %d pages, using page 0", arr.shape[0])
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in (1,) and arr.shape[-1] not in (3, 4):
        arr = arr[0]
    return arr


def load_image(path: Path) -> LoadedImage:
    """Load any 2D image to grayscale float32, keeping RGB when it is real colour."""
    suffix = path.suffix.lower()
    arr: np.ndarray | None = None

    if suffix in (".tif", ".tiff"):
        try:
            import tifffile

            arr = np.asarray(tifffile.imread(str(path)))
        except Exception as exc:  # noqa: BLE001 - fall through to PIL deliberately
            log.debug("tifffile failed on %s (%s); falling back to PIL", path.name, exc)

    if arr is None:
        with Image.open(path) as im:
            im.load()
            arr = np.asarray(im)

    arr = _squeeze_pages(np.asarray(arr))

    rgb: np.ndarray | None = None
    is_colour = False

    if arr.ndim == 2:
        gray = arr
    elif arr.ndim == 3 and arr.shape[-1] in (3, 4):
        rgb_raw = arr[..., :3]
        # "Colour" only if the channels actually differ. A grayscale frame saved
        # as RGB must not be treated as Doppler.
        c = rgb_raw.astype(np.float32)
        is_colour = bool(np.any(np.abs(c[..., 0] - c[..., 1]) > 1e-3) or
                         np.any(np.abs(c[..., 1] - c[..., 2]) > 1e-3))
        gray = (0.299 * c[..., 0] + 0.587 * c[..., 1] + 0.114 * c[..., 2])
        if is_colour:
            rgb = _to_uint8(rgb_raw)
    elif arr.ndim == 3:
        # (C, H, W) or a stack; collapse to the first plane.
        gray = arr[0] if arr.shape[0] <= 4 else arr[..., 0]
    else:
        raise ValueError(f"unsupported image shape {arr.shape} for {path}")

    gray = np.asarray(gray, dtype=np.float32)
    if gray.ndim != 2:
        raise ValueError(f"could not reduce {path} to 2D (got {gray.shape})")
    return LoadedImage(gray=gray, rgb=rgb, path=path, is_colour=is_colour)


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr)
    if a.dtype == np.uint8:
        return a
    a = a.astype(np.float32)
    lo, hi = float(a.min()), float(a.max())
    if hi <= lo:
        return np.zeros(a.shape, dtype=np.uint8)
    return ((a - lo) / (hi - lo) * 255.0).round().astype(np.uint8)


def load_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    """Load a mask PNG as a boolean array matching `shape`.

    Masks are binarised on "anything non-zero", which covers both 0/1 and 0/255
    conventions, and any per-structure label values.
    """
    with Image.open(path) as im:
        im.load()
        arr = np.asarray(im)

    arr = _squeeze_pages(np.asarray(arr))
    if arr.ndim == 3:
        arr = arr[..., :3].max(axis=-1) if arr.shape[-1] in (3, 4) else arr[0]

    mask = np.asarray(arr) > 0

    if mask.shape != shape:
        # Nearest-neighbour resize rather than a silent skip: a mask saved at a
        # different scale than its image is common and recoverable.
        log.warning(
            "mask %s is %s but image is %s; resizing nearest-neighbour",
            path.name, mask.shape, shape,
        )
        pil = Image.fromarray(mask.astype(np.uint8) * 255)
        pil = pil.resize((shape[1], shape[0]), Image.NEAREST)
        mask = np.asarray(pil) > 0

    return mask


def n_components(mask: np.ndarray) -> int:
    """Count connected components -- locules and projections are often multiple."""
    try:
        import cv2

        n, _ = cv2.connectedComponents(mask.astype(np.uint8))
        return max(0, n - 1)
    except Exception:
        try:
            from scipy import ndimage

            return int(ndimage.label(mask)[1])
        except Exception:
            return int(mask.any())


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """(ymin, xmin, ymax, xmax) inclusive-exclusive bounds of the mask."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return None
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    return int(ymin), int(xmin), int(ymax) + 1, int(xmax) + 1


def crop_to_mask(
    image: np.ndarray, mask: np.ndarray, padding: int = 10
) -> tuple[np.ndarray, np.ndarray] | None:
    """Bounding-box crop of image and mask, padded and clipped to the frame."""
    box = bbox(mask)
    if box is None:
        return None
    ymin, xmin, ymax, xmax = box
    h, w = mask.shape
    ymin = max(0, ymin - padding)
    xmin = max(0, xmin - padding)
    ymax = min(h, ymax + padding)
    xmax = min(w, xmax + padding)
    return image[ymin:ymax, xmin:xmax], mask[ymin:ymax, xmin:xmax]


def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Zero everything outside the ROI, keeping the original frame geometry."""
    out = np.array(image, copy=True)
    out[~mask] = 0
    return out


def to_rgb_uint8(gray_or_rgb: np.ndarray) -> np.ndarray:
    """Normalise anything to an (H, W, 3) uint8 array for the neural extractors."""
    a = np.asarray(gray_or_rgb)
    if a.ndim == 2:
        a = _to_uint8(a)
        return np.stack([a, a, a], axis=-1)
    if a.ndim == 3 and a.shape[-1] >= 3:
        return _to_uint8(a[..., :3])
    raise ValueError(f"cannot present shape {a.shape} as RGB")
