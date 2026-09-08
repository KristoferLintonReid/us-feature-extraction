"""AES-256-GCM container for the proprietary TexLab payload.

Format (little more than a header plus an authenticated blob):

    magic    8 bytes   b"USFTXL01"
    salt    16 bytes   scrypt salt
    nonce   12 bytes   GCM nonce
    body    remainder  ciphertext || 16-byte GCM tag

The key is scrypt(passphrase, salt, n=2**15, r=8, p=1) -> 32 bytes. GCM means a
tampered or truncated payload fails to decrypt rather than silently producing
garbage.

What this does and does not buy you is set out in docs/SECURITY.md: it keeps the
TexLab source out of a collaborator's hands during ordinary use, and it is not
proof against someone with root inside the container who is determined to get at
the decrypted files while a run is in flight.
"""
from __future__ import annotations

import hashlib
import os
import secrets
from pathlib import Path

MAGIC = b"USFTXL01"
SALT_LEN = 16
NONCE_LEN = 12
SCRYPT_N = 2 ** 15
SCRYPT_R = 8
SCRYPT_P = 1
KEY_LEN = 32


def derive_key(passphrase: str, salt: bytes) -> bytes:
    return hashlib.scrypt(
        passphrase.encode("utf-8"),
        salt=salt,
        n=SCRYPT_N,
        r=SCRYPT_R,
        p=SCRYPT_P,
        dklen=KEY_LEN,
        maxmem=64 * 1024 * 1024,
    )


def _aesgcm(key: bytes):
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    return AESGCM(key)


def encrypt(plaintext: bytes, passphrase: str) -> bytes:
    salt = secrets.token_bytes(SALT_LEN)
    nonce = secrets.token_bytes(NONCE_LEN)
    key = derive_key(passphrase, salt)
    body = _aesgcm(key).encrypt(nonce, plaintext, MAGIC)
    return MAGIC + salt + nonce + body


def decrypt(blob: bytes, passphrase: str) -> bytes:
    if len(blob) < len(MAGIC) + SALT_LEN + NONCE_LEN + 16:
        raise ValueError("payload is truncated")
    if blob[: len(MAGIC)] != MAGIC:
        raise ValueError("payload magic mismatch: not a usfeat TexLab container")
    off = len(MAGIC)
    salt = blob[off : off + SALT_LEN]
    off += SALT_LEN
    nonce = blob[off : off + NONCE_LEN]
    off += NONCE_LEN
    key = derive_key(passphrase, salt)
    try:
        return _aesgcm(key).decrypt(nonce, blob[off:], MAGIC)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(
            "TexLab payload failed to decrypt -- wrong key, or the file is corrupt"
        ) from exc


def read_passphrase(key_file: str | Path | None) -> str:
    """Resolve the payload passphrase.

    Order of precedence: explicit key file, then TEXLAB_KEY in the environment,
    then the key baked into the image at build time.
    """
    if key_file:
        path = Path(key_file)
        if not path.exists():
            raise FileNotFoundError(f"TexLab key file not found: {path}")
        return path.read_text(encoding="utf-8").strip()

    env = os.environ.get("TEXLAB_KEY")
    if env:
        return env.strip()

    baked = Path(os.environ.get("TEXLAB_KEY_FILE", "/opt/usfeat/texlab/texlab.key"))
    if baked.exists():
        return baked.read_text(encoding="utf-8").strip()

    raise FileNotFoundError(
        "no TexLab key available: pass texlab.key_file, set TEXLAB_KEY, or use an "
        "image built with the key baked in"
    )


def shred(path: Path) -> None:
    """Overwrite then unlink, so the plaintext does not linger in free blocks.

    On the tmpfs the payload normally unpacks into this is belt-and-braces; it
    matters when the container has no /dev/shm and the fallback is a real disk.
    """
    try:
        if path.is_file():
            size = path.stat().st_size
            with open(path, "r+b", buffering=0) as fh:
                fh.write(os.urandom(min(size, 1 << 20)))
                fh.flush()
                os.fsync(fh.fileno())
            path.unlink()
    except Exception:  # noqa: BLE001 - best effort; cleanup must never raise
        pass
