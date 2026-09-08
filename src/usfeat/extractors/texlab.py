"""TexLab features, run from an encrypted payload under GNU Octave.

TexLab is proprietary, so the source ships as an AES-256-GCM blob. At start of
run it is decrypted into a private directory -- /dev/shm (RAM-backed) when the
platform has one -- unpacked, used, and shredded on the way out. Nothing
readable is written to a persistent layer.

TexLab itself is a MATLAB/Octave toolbox that takes a NIfTI image and a NIfTI
mask and writes a tab-separated results file (~3,900 columns). Each ROI view is
converted to that pair, handed to `TexLAB_cli`, and the TSV is parsed back into
a flat feature dict.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

import numpy as np

from ..config import Config
from ..crypto import decrypt, read_passphrase, shred
from ..logging_setup import get_logger
from ..roi import RoiView
from .base import Extractor

log = get_logger("texlab")

# TSV columns that identify the run rather than describe the image.
_NON_FEATURE_COLUMNS = {
    "scan name",
    "voi name",
    "scan path",
    "voi path",
}


class TexLabExtractor(Extractor):
    name = "texlab"
    prefix = "texlab_"

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.tc = cfg.texlab
        self._workdir: Path | None = None
        self._texlab_root: Path | None = None
        self._saved_params: Path | None = None

        self._check_octave()
        self._unseal()

    # ------------------------------------------------------------ preparation

    def _check_octave(self) -> None:
        binary = shutil.which(self.tc.octave_binary)
        if not binary:
            raise RuntimeError(
                f"{self.tc.octave_binary} not found on PATH. TexLab needs GNU Octave "
                "(the Docker image installs it; a bare-metal run needs "
                "octave + the statistics, image and parallel packages)."
            )
        self._octave = binary
        try:
            version = subprocess.run(
                [binary, "--version"], capture_output=True, text=True, timeout=60
            ).stdout.splitlines()[0]
        except Exception:  # noqa: BLE001
            version = "unknown"
        self._octave_version = version
        log.info("octave: %s (%s)", binary, version)

    def _private_dir(self) -> Path:
        """A directory the decrypted payload can live in, RAM-backed if possible."""
        shm = Path("/dev/shm")
        parent = shm if shm.is_dir() and os.access(shm, os.W_OK) else None
        path = Path(tempfile.mkdtemp(prefix="usfeat-texlab-", dir=str(parent) if parent else None))
        os.chmod(path, 0o700)
        if parent is None:
            log.warning(
                "no writable /dev/shm; TexLab payload will be unpacked to %s on disk "
                "and shredded afterwards", path,
            )
        return path

    def _unseal(self) -> None:
        payload = Path(self.tc.payload)
        if not payload.exists():
            raise FileNotFoundError(
                f"TexLab payload not found at {payload}. Build it with "
                "tools/pack_texlab.py, or disable the extractor "
                "(--extractors without 'texlab')."
            )

        passphrase = read_passphrase(self.tc.key_file)
        blob = payload.read_bytes()
        plaintext = decrypt(blob, passphrase)

        self._workdir = self._private_dir()
        archive = self._workdir / "payload.tar.gz"
        archive.write_bytes(plaintext)
        del plaintext

        with tarfile.open(archive, "r:gz") as tar:
            _safe_extract(tar, self._workdir)
        shred(archive)

        root = self._workdir / "texlab"
        if not root.is_dir():
            raise RuntimeError("TexLab payload does not contain the expected 'texlab/' tree")
        self._texlab_root = root

        params = root / "saved_parameters.mat"
        if not params.exists():
            found = list(root.rglob("saved_parameters.mat"))
            if not found:
                raise RuntimeError(
                    "TexLab payload has no saved_parameters.mat; the CLI cannot run "
                    "without one (generate it once from the TexLab GUI)."
                )
            params = found[0]
        self._saved_params = params
        log.info("TexLab payload unsealed into a private directory")

    # ------------------------------------------------------------------ support

    def supports(self, view: RoiView) -> tuple[bool, str]:
        n = view.n_voxels
        if n < self.tc.min_roi_voxels:
            return False, f"ROI has {n} voxels, below min_roi_voxels={self.tc.min_roi_voxels}"
        return True, ""

    # ------------------------------------------------------------------ extract

    def extract(self, view: RoiView) -> dict[str, float]:
        import SimpleITK as sitk

        assert self._texlab_root and self._saved_params

        with tempfile.TemporaryDirectory(prefix="usfeat-run-") as scratch_str:
            scratch = Path(scratch_str)
            stem = "case"
            image_path = scratch / f"{stem}.nii.gz"
            mask_path = scratch / f"{stem}_mask.nii.gz"
            results_path = scratch / f"{stem}_texlab_results.tsv"

            # TexLab expects a volume; a single-slice 3D image is the 2D case.
            gray = view.image.gray.astype(np.float32)[None, ...]
            mask = view.mask.astype(np.uint8)[None, ...]

            sitk.WriteImage(sitk.GetImageFromArray(gray), str(image_path))
            sitk.WriteImage(sitk.GetImageFromArray(mask), str(mask_path))

            self._run_octave(image_path, mask_path, results_path)

            if not results_path.exists():
                raise RuntimeError("TexLab produced no results file")
            return _parse_tsv(results_path, self.prefix)

    def _run_octave(self, image: Path, mask: Path, results: Path) -> None:
        assert self._texlab_root and self._saved_params
        script = (
            f"addpath(genpath('{self._texlab_root}'));"
            f"TexLAB_cli('{image}','{mask}','{results}',"
            f"'modality','{self.tc.modality}',"
            f"'saved_parameters','{self._saved_params}');"
        )
        env = dict(os.environ)
        env.setdefault("OCTAVE_HISTFILE", "/dev/null")

        proc = subprocess.run(
            [self._octave, "--no-gui", "--quiet", "--eval", script],
            capture_output=True,
            text=True,
            timeout=self.tc.timeout_seconds,
            cwd=str(self._texlab_root),
            env=env,
        )
        if proc.returncode != 0 or not results.exists():
            raise RuntimeError(
                f"TexLab/Octave exited {proc.returncode}: "
                + _octave_diagnosis(proc.stderr, proc.stdout)
            )

    # --------------------------------------------------------------- lifecycle

    def describe(self) -> dict:
        return {
            "extractor": self.name,
            "engine": "octave",
            "octave_version": getattr(self, "_octave_version", "unknown"),
            "modality": self.tc.modality,
            "payload": str(self.tc.payload),
            "source": "encrypted (proprietary)",
        }

    def close(self) -> None:
        """Shred the decrypted payload. Called even when the run fails."""
        if not self._workdir:
            return
        try:
            for path in sorted(self._workdir.rglob("*"), key=lambda p: -len(p.parts)):
                if path.is_file():
                    shred(path)
            shutil.rmtree(self._workdir, ignore_errors=True)
            log.info("TexLab payload shredded")
        except Exception as exc:  # noqa: BLE001
            log.warning("could not fully shred TexLab workdir: %s", exc)
        finally:
            self._workdir = None
            self._texlab_root = None

    def __del__(self):  # pragma: no cover - safety net for hard exits
        try:
            self.close()
        except Exception:
            pass


def _octave_diagnosis(stderr: str | None, stdout: str | None) -> str:
    """Pull the actual error out of Octave's output.

    Octave is extremely noisy: loading the statistics package alone emits a
    dozen "shadows a core library function" warnings, and they are what you see
    if you naively take the tail of stderr. The real cause is usually a single
    line starting "error:". Those are surfaced first, with the warnings kept
    only as a fallback so nothing is lost when the pattern does not match.
    """
    text = (stderr or "") + "\n" + (stdout or "")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    errors, context = [], []
    for line in lines:
        low = line.lower()
        if low.startswith("error:"):
            errors.append(line)
        elif "cannot open shared object" in low or "no such file" in low \
                or "undefined" in low or "failed to load" in low:
            context.append(line)

    chosen = errors + [c for c in context if c not in errors]
    if not chosen:
        chosen = [ln for ln in lines if not ln.lower().startswith("warning")][-6:] or lines[-6:]
    return " | ".join(chosen)[:1200]


def _safe_extract(tar: tarfile.TarFile, dest: Path) -> None:
    """Extract with path traversal refused."""
    dest = dest.resolve()
    for member in tar.getmembers():
        target = (dest / member.name).resolve()
        if not str(target).startswith(str(dest)):
            raise RuntimeError(f"refusing unsafe path in payload: {member.name}")
    tar.extractall(dest)


def _parse_tsv(path: Path, prefix: str) -> dict[str, float]:
    """Parse a TexLab results TSV into a flat feature dict.

    The file is a header row plus one row per (scan, VOI). Only one VOI is ever
    passed, so the first data row is the answer; anything beyond it is logged.
    """
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 2:
        raise RuntimeError(f"TexLab results file has no data rows ({len(lines)} lines)")

    header = lines[0].split("\t")
    values = lines[1].split("\t")
    if len(values) < len(header):
        values += [""] * (len(header) - len(values))
    if len(lines) > 2:
        log.debug("TexLab returned %d data rows; using the first", len(lines) - 1)

    features: dict[str, float] = {}
    for name, raw in zip(header, values):
        key = name.strip()
        if not key:
            continue
        if key.lower() in _NON_FEATURE_COLUMNS:
            continue
        column = f"{prefix}{key.replace(' ', '_')}"
        raw = raw.strip()
        if raw in ("", "NA", "NaN", "nan", "Inf", "-Inf"):
            features[column] = float("nan")
            continue
        try:
            features[column] = float(raw)
        except ValueError:
            features[column] = float("nan")

    if not features:
        raise RuntimeError("TexLab results file parsed to zero features")
    return features
