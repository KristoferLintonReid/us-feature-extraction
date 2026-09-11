# Security and data handling

## Your data never leaves the machine that runs the container

This is the property that matters most to a collaborator's information-governance process, and
it is enforced in four independent ways rather than merely promised:

- `run.sh` and `docker-compose.yml` start the container with **no network interface**
  (`--network none` / `network_mode: none`). It could not transmit anything if it tried.
- **All model weights are baked into the image at build time.** Nothing is downloaded at run
  time; the runtime stage also sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`.
- **The pipeline source contains no networking code** — no `requests`, no `urllib`, no sockets,
  no telemetry, no upload path. Verifiable in one command:
  `grep -rE "requests|urllib|http|socket" src/`
- **The data volume is mounted read-only** (`-v /your/data:/data:ro`), so the pipeline cannot
  modify the originals. It writes only to the output directory you nominate.

Because these are independent, any one of them failing does not open a path out. The weakest
link is an operator removing `--network none`, which is why that flag lives in the committed
runner rather than in documentation, and why `USFEAT_ALLOW_NETWORK=1` warns loudly when used.

Verify it yourself:

```bash
docker run --rm --network none usfeat:latest --version
```

## Credentials

`HF_TOKEN` is used **only at build time**, to download model weights into the image. It is
passed as a BuildKit secret mount:

```bash
HF_TOKEN=hf_xxx docker build -t usfeat:latest --secret id=hf_token,env=HF_TOKEN .
```

Mounted secrets do not appear in `docker history` or in any image layer, unlike `--build-arg`.
The runtime stage has no credentials at all and does not need any.

Do not commit tokens. `.gitignore` and `.dockerignore` exclude `*.key` and `secrets/`. Rotate
any token that has been shared.

## Container hardening

The container runs unprivileged (`USER usfeat`, uid 1000). `--cap-drop=ALL` can be added
safely. `--read-only` should also work but has not been tested here, so try it against
`sample_data/` first.

## Not a medical device

This is a research feature-extraction tool. It produces numerical descriptors of images and
makes no clinical claim. It must not be used for diagnosis or to inform patient management.
