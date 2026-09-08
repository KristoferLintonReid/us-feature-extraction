#!/usr/bin/env python3
"""Build the encrypted TexLab payload.

Run this on a machine that has the TexLab source. It produces a single
`texlab.enc` file, which is what gets baked into the Docker image and shipped.
The plaintext source never leaves this machine.

    python tools/pack_texlab.py \
        --texlab-dir ~/Downloads/TexLab_V3-main/TexLAB_v3 \
        --saved-parameters ~/Desktop/TexLAB-master/initialisation/saved_parameters.mat \
        --out build/texlab.enc \
        --key-out build/texlab.key

The key file is written separately and must NOT be committed. Bake it into the
image with `--build-arg TEXLAB_KEY=$(cat build/texlab.key)`, or hand it to the
collaborator out of band if you want to keep the ability to revoke.
"""
from __future__ import annotations

import argparse
import io
import secrets
import shutil
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from usfeat.crypto import encrypt  # noqa: E402

# Directories that bloat the payload without being needed at runtime.
EXCLUDE_DIRS = {".git", "__pycache__", ".ipynb_checkpoints", "texLab_test_data", "help"}
EXCLUDE_SUFFIXES = {".mp4", ".docx", ".pdf", ".zip", ".png", ".jpg"}


def build_tar(texlab_dir: Path, saved_parameters: Path, keep_docs: bool) -> bytes:
    """Tar the TexLab tree under a fixed 'texlab/' prefix."""
    buf = io.BytesIO()
    n_files = 0

    def _filter(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
        nonlocal n_files
        parts = set(Path(info.name).parts)
        if parts & EXCLUDE_DIRS:
            return None
        if not keep_docs and Path(info.name).suffix.lower() in EXCLUDE_SUFFIXES:
            return None
        # Normalise ownership so the payload hashes reproducibly.
        info.uid = info.gid = 0
        info.uname = info.gname = "root"
        if info.isfile():
            n_files += 1
        return info

    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(str(texlab_dir), arcname="texlab", filter=_filter)
        if saved_parameters and saved_parameters.exists():
            tar.add(str(saved_parameters), arcname="texlab/saved_parameters.mat")
            n_files += 1

    print(f"  packed {n_files} files")
    return buf.getvalue()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--texlab-dir", type=Path, required=True,
                    help="TexLab source tree (the directory containing TexLAB_cli.m)")
    ap.add_argument("--saved-parameters", type=Path,
                    help="saved_parameters.mat produced by the TexLab GUI")
    ap.add_argument("--out", type=Path, default=Path("build/texlab.enc"))
    ap.add_argument("--key-out", type=Path, default=Path("build/texlab.key"),
                    help="where to write the generated passphrase")
    ap.add_argument("--key", help="use this passphrase instead of generating one")
    ap.add_argument("--keep-docs", action="store_true",
                    help="keep videos/PDFs in the payload (much larger)")
    args = ap.parse_args()

    if not args.texlab_dir.is_dir():
        print(f"error: {args.texlab_dir} is not a directory", file=sys.stderr)
        return 1
    cli = args.texlab_dir / "TexLAB_cli.m"
    if not cli.exists():
        print(f"error: {cli} not found -- is this the TexLab root?", file=sys.stderr)
        return 1

    saved = args.saved_parameters
    if saved and not saved.exists():
        print(f"error: saved parameters not found: {saved}", file=sys.stderr)
        return 1
    if not saved:
        found = list(args.texlab_dir.rglob("saved_parameters.mat"))
        if found:
            saved = found[0]
            print(f"  using saved parameters found in tree: {saved}")
        else:
            print("warning: no saved_parameters.mat -- TexLAB_cli will not run without one",
                  file=sys.stderr)

    print(f"packing {args.texlab_dir} ...")
    plaintext = build_tar(args.texlab_dir, saved, args.keep_docs)
    print(f"  plaintext archive: {len(plaintext) / 1e6:.1f} MB")

    passphrase = args.key or secrets.token_urlsafe(48)
    blob = encrypt(plaintext, passphrase)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(blob)
    print(f"  encrypted payload: {args.out} ({len(blob) / 1e6:.1f} MB)")

    if not args.key:
        args.key_out.parent.mkdir(parents=True, exist_ok=True)
        args.key_out.write_text(passphrase, encoding="utf-8")
        args.key_out.chmod(0o600)
        print(f"  key: {args.key_out}  (mode 0600 -- do not commit this)")

    # Prove the payload round-trips before it is shipped anywhere.
    from usfeat.crypto import decrypt

    assert decrypt(blob, passphrase) == plaintext
    print("  verified: payload decrypts cleanly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
