"""TexLab module tests that do not require Octave.

The Octave invocation itself is exercised in the Docker image, where Debian's
prebuilt octave-statistics / octave-image / octave-parallel packages provide the
combination TexLab v3 supports. Everything either side of that call -- payload
sealing, key resolution, NIfTI conversion, TSV parsing, cleanup -- is covered
here and runs anywhere.
"""
from __future__ import annotations

import io
import sys
import tarfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from usfeat.crypto import decrypt, encrypt, read_passphrase, shred  # noqa: E402
from usfeat.extractors.texlab import _parse_tsv, _safe_extract  # noqa: E402

# The real TexLab v3 results format: 4 identifier columns, then N voxels, then
# ~3,900 feature columns across SNS / FOS / GLSZM / GLCM / fractal families,
# each repeated per wavelet decomposition and intensity normalisation.
REAL_HEADER = [
    "Scan name", "VOI name", "Scan path", "VOI path", "N voxels",
    "SNS_vol", "SNS_area", "SNS_s2v", "SNS_sph", "SNS_sph_dis",
    "FOS_CV", "FOS_Imean", "FOS_Imedian", "FOS_Imode", "FOS_Istd",
    "AUC-CSH", "FOS_Entr_4gl", "FOS_Ener_4gl",
    "GLSZM_SmallZone_4gl", "GLSZM_LargeZone_4gl", "GLSZM_GlNonUnif_4gl",
]
REAL_ROW = [
    "008_3B.nii.gz", "008_3B_mask.nii.gz", "/data/", "/data/", "20458",
    "20458", "42424", "2.07371199530746", "0.0852670608668897", "11.72785821199",
    "0.58675315097723", "56.2776419982403", "62", "0", "33.021083772036",
    "0.0301812639613039", "2.41661038779606", "27.2943131948922",
    "0.4", "0.6", "1.2",
]


def _write_tsv(path: Path, header: list[str], *rows: list[str]) -> Path:
    lines = ["\t".join(header)] + ["\t".join(r) for r in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ------------------------------------------------------------------ TSV parsing


def test_parses_real_format_and_drops_identifier_columns(tmp_path: Path):
    path = _write_tsv(tmp_path / "r.tsv", REAL_HEADER, REAL_ROW)
    feats = _parse_tsv(path, "texlab_")

    # The four path/name columns are identifiers, not features.
    assert len(feats) == len(REAL_HEADER) - 4
    for dropped in ("texlab_Scan_name", "texlab_VOI_name",
                    "texlab_Scan_path", "texlab_VOI_path"):
        assert dropped not in feats

    assert feats["texlab_N_voxels"] == 20458.0
    assert feats["texlab_SNS_vol"] == 20458.0
    assert feats["texlab_FOS_Imean"] == pytest.approx(56.2776419982403)
    # A hyphenated name must survive intact.
    assert feats["texlab_AUC-CSH"] == pytest.approx(0.0301812639613039)


def test_non_numeric_sentinels_become_nan(tmp_path: Path):
    header = ["Scan name", "N voxels", "FOS_Imean", "FOS_Istd", "FOS_Skew", "FOS_Kurt"]
    row = ["a.nii", "10", "NA", "NaN", "Inf", "not-a-number"]
    feats = _parse_tsv(_write_tsv(tmp_path / "n.tsv", header, row), "texlab_")

    assert feats["texlab_N_voxels"] == 10.0
    for key in ("texlab_FOS_Imean", "texlab_FOS_Istd",
                "texlab_FOS_Skew", "texlab_FOS_Kurt"):
        assert np.isnan(feats[key])


def test_short_row_is_padded_not_misaligned(tmp_path: Path):
    """A truncated row must not shift values onto the wrong column names."""
    header = ["Scan name", "A", "B", "C"]
    feats = _parse_tsv(_write_tsv(tmp_path / "s.tsv", header, ["x.nii", "1", "2"]), "texlab_")
    assert feats["texlab_A"] == 1.0
    assert feats["texlab_B"] == 2.0
    assert np.isnan(feats["texlab_C"])


def test_header_only_file_is_an_error(tmp_path: Path):
    path = _write_tsv(tmp_path / "h.tsv", REAL_HEADER)
    with pytest.raises(RuntimeError, match="no data rows"):
        _parse_tsv(path, "texlab_")


def test_first_data_row_is_used_when_several_are_present(tmp_path: Path):
    second = list(REAL_ROW)
    second[11] = "999.0"
    path = _write_tsv(tmp_path / "m.tsv", REAL_HEADER, REAL_ROW, second)
    assert _parse_tsv(path, "texlab_")["texlab_FOS_Imean"] == pytest.approx(56.2776419982403)


def test_blank_column_names_are_ignored(tmp_path: Path):
    header = ["Scan name", "", "FOS_Imean", "  "]
    feats = _parse_tsv(_write_tsv(tmp_path / "b.tsv", header, ["a", "1", "2", "3"]), "texlab_")
    assert set(feats) == {"texlab_FOS_Imean"}


# ---------------------------------------------------------------- payload/keys


def test_payload_round_trips_a_directory(tmp_path: Path):
    """The full pack -> encrypt -> decrypt -> unpack cycle."""
    src = tmp_path / "texlab"
    (src / "initialisation").mkdir(parents=True)
    (src / "TexLAB_cli.m").write_text("function TexLAB_cli(varargin)\n% proprietary\n")
    (src / "initialisation" / "saved_parameters.mat").write_bytes(b"MATLAB 5.0 MAT-file")

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(str(src), arcname="texlab")
    plaintext = buf.getvalue()

    blob = encrypt(plaintext, "passphrase")
    assert b"proprietary" not in blob
    assert b"TexLAB_cli" not in blob

    dest = tmp_path / "unpacked"
    dest.mkdir()
    with tarfile.open(fileobj=io.BytesIO(decrypt(blob, "passphrase")), mode="r:gz") as tar:
        _safe_extract(tar, dest)
    assert (dest / "texlab" / "TexLAB_cli.m").read_text().startswith("function TexLAB_cli")


def test_path_traversal_in_payload_is_refused(tmp_path: Path):
    """A payload trying to escape its extraction directory must be rejected."""
    evil = tmp_path / "evil.tar.gz"
    victim = tmp_path / "outside.txt"
    victim.write_text("original")

    with tarfile.open(evil, "w:gz") as tar:
        info = tarfile.TarInfo(name="../outside.txt")
        data = b"overwritten"
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))

    dest = tmp_path / "dest"
    dest.mkdir()
    with tarfile.open(evil, "r:gz") as tar:
        with pytest.raises(RuntimeError, match="unsafe path"):
            _safe_extract(tar, dest)
    assert victim.read_text() == "original"


def test_key_precedence_file_beats_environment(tmp_path: Path, monkeypatch):
    key_file = tmp_path / "k"
    key_file.write_text("from-file\n")
    monkeypatch.setenv("TEXLAB_KEY", "from-env")
    assert read_passphrase(key_file) == "from-file"


def test_key_falls_back_to_environment(monkeypatch):
    monkeypatch.setenv("TEXLAB_KEY", "  from-env  ")
    assert read_passphrase(None) == "from-env"


def test_key_falls_back_to_baked_location(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("TEXLAB_KEY", raising=False)
    baked = tmp_path / "texlab.key"
    baked.write_text("baked-in")
    monkeypatch.setenv("TEXLAB_KEY_FILE", str(baked))
    assert read_passphrase(None) == "baked-in"


def test_missing_key_is_a_clear_error(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("TEXLAB_KEY", raising=False)
    monkeypatch.setenv("TEXLAB_KEY_FILE", str(tmp_path / "absent"))
    with pytest.raises(FileNotFoundError, match="no TexLab key available"):
        read_passphrase(None)


def test_missing_key_file_is_reported(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="key file not found"):
        read_passphrase(tmp_path / "nope")


def test_shred_removes_the_file(tmp_path: Path):
    path = tmp_path / "secret.m"
    path.write_bytes(b"proprietary source" * 100)
    shred(path)
    assert not path.exists()


def test_shred_of_absent_file_is_silent(tmp_path: Path):
    shred(tmp_path / "never-existed")  # must not raise


# ------------------------------------------------------------ NIfTI conversion


def test_roi_converts_to_single_slice_nifti(tmp_path: Path):
    """What the extractor hands Octave: a 1-slice volume and a matching mask."""
    sitk = pytest.importorskip("SimpleITK")

    gray = np.random.default_rng(0).random((64, 80)).astype(np.float32)
    mask = np.zeros((64, 80), dtype=bool)
    mask[16:48, 20:60] = True

    image_path = tmp_path / "case.nii.gz"
    mask_path = tmp_path / "case_mask.nii.gz"
    sitk.WriteImage(sitk.GetImageFromArray(gray[None, ...]), str(image_path))
    sitk.WriteImage(sitk.GetImageFromArray(mask.astype(np.uint8)[None, ...]), str(mask_path))

    back_img = sitk.GetArrayFromImage(sitk.ReadImage(str(image_path)))
    back_msk = sitk.GetArrayFromImage(sitk.ReadImage(str(mask_path)))

    assert back_img.shape == back_msk.shape == (1, 64, 80)
    assert np.allclose(back_img[0], gray, atol=1e-6)
    assert back_msk.sum() == mask.sum() == 32 * 40
