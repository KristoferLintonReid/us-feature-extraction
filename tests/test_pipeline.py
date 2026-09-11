"""Tests that do not need model weights.

Run with:  python -m pytest tests -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from usfeat.config import Config  # noqa: E402
from usfeat.discovery import discover  # noqa: E402
from usfeat.imaging import (  # noqa: E402
    bbox, crop_to_mask, load_image, load_mask, n_components, to_rgb_uint8,
)
from usfeat.metadata import _normalise_key, load_metadata  # noqa: E402
from usfeat.roi import build_views  # noqa: E402


# --------------------------------------------------------------------- fixtures


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """A miniature version of the provider's layout."""
    rng = np.random.default_rng(0)

    for sub in ("images_grayscale", "images_doppler",
                "images_seg/img", "images_seg/lesion", "images_seg/solid"):
        (tmp_path / sub).mkdir(parents=True)

    gray = rng.integers(0, 255, (64, 80), dtype=np.uint8)
    Image.fromarray(gray).save(tmp_path / "images_grayscale" / "CASE-001.tiff")
    Image.fromarray(gray).save(tmp_path / "images_grayscale" / "CASE-002.tiff")

    # Genuinely coloured, so the Doppler detection has something to find.
    colour = rng.integers(0, 255, (64, 80, 3), dtype=np.uint8)
    Image.fromarray(colour).save(tmp_path / "images_doppler" / "CASE-002.tiff")

    for label in (0, 1):
        Image.fromarray(gray).save(tmp_path / "images_seg" / "img" / f"{label}.tiff")
        mask = np.zeros((64, 80), dtype=np.uint8)
        mask[16:48, 20:60] = 255
        Image.fromarray(mask).save(tmp_path / "images_seg" / "lesion" / f"{label}.png")
        core = np.zeros((64, 80), dtype=np.uint8)
        core[24:40, 30:50] = 255
        Image.fromarray(core).save(tmp_path / "images_seg" / "solid" / f"{label}.png")

    return tmp_path


@pytest.fixture
def cfg(tree: Path, tmp_path: Path) -> Config:
    return Config.load(
        None,
        data_root=tree,
        output_dir=tmp_path / "out",
        extractors=["pyradiomics"],
        rois=["lesion", "solid"],
    )


# -------------------------------------------------------------------- discovery


def test_discovery_finds_every_image(cfg: Config):
    items = discover(cfg)
    assert len(items) == 5  # 2 grayscale + 1 doppler + 2 segmented


def test_doppler_and_grayscale_share_a_pair_id(cfg: Config):
    items = {i.image_id: i for i in discover(cfg)}
    gray = items["grayscale/CASE-002"]
    dopp = items["doppler/CASE-002"]
    assert gray.pair_id == dopp.pair_id == "CASE-002"
    assert gray.channel != dopp.channel


def test_segmented_items_carry_only_configured_rois(cfg: Config):
    seg = [i for i in discover(cfg) if i.source == "segmented"]
    assert len(seg) == 2
    for item in seg:
        assert set(item.masks) == {"lesion", "solid"}


def test_limit_keeps_both_halves(tree: Path, tmp_path: Path):
    cfg = Config.load(None, data_root=tree, output_dir=tmp_path / "out", limit=2)
    items = discover(cfg)
    assert len(items) == 2
    assert {i.source for i in items} == {"unsegmented", "segmented"}


def test_missing_data_root_is_a_clear_error(tmp_path: Path):
    cfg = Config.load(None, data_root=tmp_path / "nope", output_dir=tmp_path / "out")
    with pytest.raises(FileNotFoundError):
        discover(cfg)


# ----------------------------------------------------------------------- imaging


def test_grayscale_tiff_loads_as_2d(tree: Path):
    img = load_image(tree / "images_grayscale" / "CASE-001.tiff")
    assert img.gray.ndim == 2
    assert img.is_colour is False
    assert img.rgb is None


def test_colour_tiff_is_detected_and_keeps_rgb(tree: Path):
    img = load_image(tree / "images_doppler" / "CASE-002.tiff")
    assert img.is_colour is True
    assert img.rgb is not None and img.rgb.shape[-1] == 3
    assert img.gray.ndim == 2


def test_grayscale_saved_as_rgb_is_not_mistaken_for_doppler(tmp_path: Path):
    """A flat grayscale frame stored in three channels must not read as colour."""
    gray = np.full((32, 32), 120, dtype=np.uint8)
    Image.fromarray(np.stack([gray] * 3, axis=-1)).save(tmp_path / "flat.tiff")
    assert load_image(tmp_path / "flat.tiff").is_colour is False


def test_mask_binarises_from_either_convention(tmp_path: Path):
    for value in (1, 255):
        arr = np.zeros((20, 20), dtype=np.uint8)
        arr[5:15, 5:15] = value
        path = tmp_path / f"m{value}.png"
        Image.fromarray(arr).save(path)
        assert load_mask(path, (20, 20)).sum() == 100


def test_mismatched_mask_is_resized_not_dropped(tmp_path: Path):
    arr = np.zeros((10, 10), dtype=np.uint8)
    arr[2:8, 2:8] = 255
    Image.fromarray(arr).save(tmp_path / "small.png")
    mask = load_mask(tmp_path / "small.png", (20, 20))
    assert mask.shape == (20, 20)
    assert mask.any()


def test_bbox_and_crop(tmp_path: Path):
    mask = np.zeros((50, 50), dtype=bool)
    mask[10:20, 30:45] = True
    assert bbox(mask) == (10, 30, 20, 45)

    image = np.arange(2500, dtype=np.float32).reshape(50, 50)
    cropped_image, cropped_mask = crop_to_mask(image, mask, padding=2)
    assert cropped_image.shape == cropped_mask.shape == (14, 19)


def test_bbox_of_empty_mask_is_none():
    assert bbox(np.zeros((10, 10), dtype=bool)) is None
    assert crop_to_mask(np.zeros((10, 10)), np.zeros((10, 10), dtype=bool)) is None


def test_component_counting():
    mask = np.zeros((40, 40), dtype=bool)
    mask[2:8, 2:8] = True
    mask[20:28, 20:28] = True
    assert n_components(mask) == 2


def test_to_rgb_uint8_shapes():
    assert to_rgb_uint8(np.zeros((8, 9), dtype=np.float32)).shape == (8, 9, 3)
    assert to_rgb_uint8(np.zeros((8, 9, 3), dtype=np.uint8)).shape == (8, 9, 3)


# --------------------------------------------------------------------------- ROI


def test_whole_view_is_always_present(cfg: Config, tree: Path):
    item = next(i for i in discover(cfg) if i.source == "unsegmented")
    views = build_views(item, load_image(item.image_path), cfg)
    assert [v.roi for v in views] == ["whole"]
    assert views[0].mask.all()


def test_segmented_item_yields_whole_plus_each_roi(cfg: Config):
    item = next(i for i in discover(cfg) if i.source == "segmented")
    views = build_views(item, load_image(item.image_path), cfg)
    assert [v.roi for v in views] == ["whole", "lesion", "solid"]
    assert views[1].n_voxels == 32 * 40


def test_empty_mask_is_skipped(cfg: Config, tree: Path):
    """An all-zero mask must drop that ROI, not produce a degenerate row."""
    blank = np.zeros((64, 80), dtype=np.uint8)
    Image.fromarray(blank).save(tree / "images_seg" / "solid" / "0.png")
    item = next(i for i in discover(cfg) if i.image_id == "seg/0")
    views = build_views(item, load_image(item.image_path), cfg)
    assert [v.roi for v in views] == ["whole", "lesion"]


def test_index_fields_are_complete(cfg: Config):
    item = next(i for i in discover(cfg) if i.source == "segmented")
    view = build_views(item, load_image(item.image_path), cfg)[1]
    fields = view.index_fields()
    for key in ("image_id", "roi", "roi_n_voxels", "roi_n_components",
                "roi_fraction", "image_height", "image_width", "image_is_colour"):
        assert key in fields


# ---------------------------------------------------------------------- metadata


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("MPO-35606-I0000010.tiff", "mpo-35606-i0000010"),
        ("MPO-35606-I0000010", "mpo-35606-i0000010"),
        ("  MPO-35606-I0000010.TIFF  ", "mpo-35606-i0000010"),
        (12, "12"),
        (12.0, "12"),
        ("12.0", "12"),
    ],
)
def test_key_normalisation(raw, expected):
    assert _normalise_key(raw) == expected


def test_metadata_joins_and_prefixes(tree: Path, tmp_path: Path):
    import pandas as pd

    pd.DataFrame([
        {"File Name": "CASE-001.tiff", "CDM ID": "P1", "Iota7_basic.histology": "benign"},
        {"File Name": "CASE-002.tiff", "CDM ID": "P2", "Iota7_basic.histology": "malignant"},
    ]).to_excel(tree / "metadata_unsegmented.xlsx", index=False)
    pd.DataFrame([{"Label": 0, "Histology": "borderline"}]).to_excel(
        tree / "metadata_segmented.xlsx", index=False)

    cfg = Config.load(None, data_root=tree, output_dir=tmp_path / "out")
    store = load_metadata(cfg)

    assert "meta_CDM_ID" in store.columns
    assert store.lookup("unsegmented", "CASE-001.tiff")["meta_CDM_ID"] == "P1"
    # Doppler and grayscale share a filename, so both resolve to the same row.
    assert store.lookup("unsegmented", "CASE-002.tiff")["meta_Iota7_basic.histology"] == "malignant"
    assert store.lookup("segmented", "0")["meta_Histology"] == "borderline"


def test_absent_metadata_degrades_quietly(cfg: Config):
    store = load_metadata(cfg)          # no sheets exist in this fixture
    assert store.columns == []
    assert store.lookup("unsegmented", "anything") == {}


def test_unmatched_key_is_counted(tree: Path, tmp_path: Path):
    import pandas as pd

    pd.DataFrame([{"File Name": "OTHER.tiff", "CDM ID": "P9"}]).to_excel(
        tree / "metadata_unsegmented.xlsx", index=False)
    cfg = Config.load(None, data_root=tree, output_dir=tmp_path / "out")
    store = load_metadata(cfg)
    assert store.lookup("unsegmented", "CASE-001.tiff") == {}
    assert store.miss_counts["unsegmented"] == 1


# ------------------------------------------------------------------------ config


def test_unknown_extractor_is_rejected():
    with pytest.raises(ValueError, match="unknown extractor"):
        Config.load(None, extractors=["not_a_real_extractor"])


def test_fingerprint_tracks_feature_affecting_settings(tmp_path: Path):
    a = Config.load(None, data_root=tmp_path, output_dir=tmp_path / "a")
    b = Config.load(None, data_root=tmp_path, output_dir=tmp_path / "b")
    # Output location does not change the features, so it must not change the hash.
    assert a.fingerprint() == b.fingerprint()

    c = Config.load(None, data_root=tmp_path, output_dir=tmp_path / "a")
    c.pyradiomics.bin_width = 12.5
    assert c.fingerprint() != a.fingerprint()


# ------------------------------------------------------------------------ writer


def test_writer_shards_dedupes_and_stamps_provenance(tmp_path: Path):
    from usfeat.writer import FeatureWriter, read_provenance

    writer = FeatureWriter(tmp_path, "demo", "lesion", {"extractor": "demo"}, shard_size=2)
    for i in range(5):
        writer.add({"image_id": f"img{i}", "roi": "lesion", "source": "x",
                    "channel": "grayscale", "stem": f"s{i}", "feat_0": float(i)})
    # A repeat of an existing id must collapse to one row, keeping the later value.
    writer.add({"image_id": "img0", "roi": "lesion", "source": "x",
                "channel": "grayscale", "stem": "s0", "feat_0": 99.0})

    path = writer.finalize()
    assert path is not None

    import pandas as pd

    df = pd.read_parquet(path)
    assert len(df) == 5
    assert df.loc[df.image_id == "img0", "feat_0"].iloc[0] == 99.0
    assert read_provenance(path)["extractor"] == "demo"


def test_writer_reports_existing_ids_for_resume(tmp_path: Path):
    from usfeat.writer import FeatureWriter

    first = FeatureWriter(tmp_path, "demo", "whole", {})
    first.add({"image_id": "a", "roi": "whole", "source": "x",
               "channel": "grayscale", "stem": "a", "f": 1.0})
    first.finalize()

    second = FeatureWriter(tmp_path, "demo", "whole", {})
    assert second.load_existing_ids() == {"a"}


def test_writer_with_no_rows_writes_nothing(tmp_path: Path):
    from usfeat.writer import FeatureWriter

    assert FeatureWriter(tmp_path, "demo", "whole", {}).finalize() is None
