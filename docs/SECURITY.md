# Protecting the TexLab source

## What is being protected

TexLab is proprietary. The collaborator must be able to **run** TexLab feature extraction on
their own data, on their own machine, without being able to **read** the TexLab source.

## What ships

The TexLab source is packaged by `tools/pack_texlab.py` into a single file, `texlab.enc`:

```
magic    8 bytes    "USFTXL01"
salt    16 bytes    scrypt salt
nonce   12 bytes    AES-GCM nonce
body    remainder   ciphertext || 16-byte GCM tag
```

- **Cipher**: AES-256-GCM (authenticated — a tampered or truncated payload fails to decrypt
  rather than silently producing garbage).
- **Key derivation**: `scrypt(passphrase, salt, N=2^15, r=8, p=1) → 32 bytes`.
- **Passphrase**: 48 random URL-safe bytes, generated at packaging time.

At the start of a run, `TexLabExtractor` decrypts the payload into a private directory —
`/dev/shm` (RAM-backed tmpfs) when available, `mkdtemp` otherwise — with mode `0700`, unpacks
it, uses it, and shreds it (overwrite, then unlink) when the run ends, including on failure.
The plaintext is never written to a persistent image layer or a mounted volume.

Octave, not MATLAB, executes it. TexLab v3 supports Octave, so there is no MATLAB licence for
the collaborator to obtain.

## What this actually buys you — and what it does not

**It does defeat:** reading the image layers, `docker cp`, browsing the container filesystem
between runs, casual curiosity, and accidental disclosure. The source is not present in
readable form at rest, anywhere.

**It does not defeat:** someone with root inside the container who is deliberately trying to
extract the source *while a run is in flight*. The key is baked into the image, so it can be
recovered by anyone determined to do so; and once recovered, the payload decrypts. Between
`_unseal()` and `close()` the plaintext exists in that container's `/dev/shm`, and root can
read another process's tmpfs.

This is **obfuscation-grade protection against an honest collaborator**, which is the actual
threat model here. It is deliberately not being sold as anything stronger.

If the requirement hardens, there are two upgrade paths, in increasing order of strength:

1. **Hold the key.** Build without `--build-arg TEXLAB_KEY` and send a key file per run
   (`--texlab-key-file`, or `TEXLAB_KEY` in the environment). The image alone is then inert,
   and access is revocable. Costs the collaborator one extra file.
2. **Compile it.** MATLAB Compiler (`mcc`) turns TexLab into a standalone binary that runs
   against the free MATLAB Runtime. No source exists in the artefact at all, in any form, at
   any point. This is genuine protection rather than obfuscation. It needs a MATLAB Compiler
   licence, which was not available on the machine this was built on — `mcc` is absent from
   the R2020a / R2021a / R2025b installs there, only the deployment libraries are present.

   The code is structured so this is a contained change: `TexLabExtractor._unseal()` and
   `_run_octave()` are the only two methods that would be replaced, by a call to the compiled
   binary. Nothing else in the pipeline knows how TexLab is executed.

## Operational rules

- **Never commit `texlab.key`.** Both `.gitignore` and `.dockerignore` exclude `*.key`.
- **Never commit `texlab.enc` either.** It is a build artefact; `.gitignore` excludes `*.enc`.
  Keep it wherever you keep release binaries.
- The key reaches the image as a `--build-arg`. Build args are visible in image history, which
  is one more reason the honest description of this scheme is "obfuscation".
- Rotate the passphrase by re-running `pack_texlab.py`; every payload gets a fresh salt and
  nonce regardless.

## Other secrets

`HF_TOKEN` is used **only at build time**, to download model weights into the image. It is not
present in the runtime stage and is not needed at run time — the runtime sets
`HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`, so the container makes no network calls at
all. Do not commit it, and rotate it if it has been shared.
