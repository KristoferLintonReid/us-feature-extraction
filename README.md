# usfeat — ultrasound feature extraction

Extracts **seven independent feature families** from 2D ultrasound, **whole-image and per-ROI**,
and writes one Parquet table per (family × ROI) with the study metadata embedded in every file.

| Family | Source | Features per ROI |
|---|---|---|
| `texlab` | TexLab v3 (proprietary, encrypted, runs under Octave) | ~3,900 |
| `pyradiomics` | PyRadiomics — all 7 classes × 9 image filters | ~1,150 |
| `dinov2` | `facebook/dinov2-base` | 1,536 (CLS + patch-mean) |
| `dinov3` | `facebook/dinov3-vitb16-pretrain-lvd1689m` | 1,536 (pooled + patch-mean) |
| `biomedclip` | BiomedCLIP PubMedBERT ViT-B/16 | 512 |
| `siglip` | `google/siglip-base-patch16-224` | 1,536 (pooled + patch-mean) |
| `imagenet` | ResNet-50 + ViT-B/16, supervised ImageNet | 2,816 |

Every family is optional and independent. One that cannot start (missing weights, no Octave,
no TexLab key) reports itself and the rest of the run proceeds.

---

## Quick start (Docker)

```bash
# 1. Put your data where the container can see it (see "Input layout" below).
# 2. Run.
docker run --rm \
  -v /path/to/your/data:/data:ro \
  -v /path/to/output:/out \
  usfeat:latest extract --data /data --out /out
```

That is the whole thing. Results land in `/path/to/output`.

There is a wrapper if you prefer not to type the mounts:

```bash
./run.sh /path/to/your/data /path/to/output
```

**Before running the whole dataset**, do a smoke test on ten images — it takes a couple of
minutes and catches layout problems immediately:

```bash
docker run --rm -v /path/to/your/data:/data:ro -v /path/to/output:/out \
  usfeat:latest extract --data /data --out /out --limit 10
```

And to see what *would* be processed without processing it:

```bash
docker run --rm -v /path/to/your/data:/data:ro usfeat:latest inspect --data /data
```

---

## Input layout

This is the structure the pipeline expects. It matches what was agreed — no renaming needed.

```
<root>/
├── images_grayscale/              Grayscale + Doppler-removed images
│     ├── MPO-35606-I0000010.tiff
│     └── ...
├── images_doppler/                Colour Doppler images
│     ├── MPO-35606-I0000010.tiff
│     └── ...
├── images_seg/                    The ~700 with segmentations
│     ├── img/          0.tiff, 1.tiff, ...
│     ├── lesion/       0.png, ...
│     ├── locule/
│     ├── projection/
│     └── solid/
├── metadata_unsegmented.xlsx      keyed on "File Name"
└── metadata_segmented.xlsx        keyed on "Label"
```

A file present in **both** `images_grayscale/` and `images_doppler/` under the same name is
treated as one acquisition in two forms: the two rows get the same `pair_id`, with
`channel` distinguishing them. So you can analyse Doppler and Doppler-removed
independently, or pivot them together.

### Metadata

Two spreadsheets, both optional. Any file format works (`.xlsx`, `.csv`, `.tsv`, `.parquet`).

**`metadata_unsegmented.xlsx`** — the columns already listed:
`File Name`, `CDM ID`, `Center`, `Iota7_basic.histology`, `Iota7_basic.extra_colourscore`

**`metadata_segmented.xlsx`**: `Label`, `Histology`

Every column in both sheets is carried into every feature table, prefixed `meta_`. Nothing
needs selecting in advance — send whatever you have and it comes through. Key matching is
lenient: `MPO-35606-I0000010.tiff`, `MPO-35606-I0000010` and `  mpo-35606-i0000010 ` all
resolve to the same row, and Excel's habit of turning label `12` into `12.0` is handled.

If a sheet is missing or a key does not match, the features are still produced — the
`meta_` columns are simply null, and the run summary tells you how many rows that affected.

---

## Output

```
<out>/
├── features/
│   ├── texlab__whole.parquet          ← no ROI: whole image
│   ├── texlab__lesion.parquet
│   ├── texlab__locule.parquet
│   ├── texlab__projection.parquet
│   ├── texlab__solid.parquet
│   ├── pyradiomics__whole.parquet
│   ├── pyradiomics__lesion.parquet
│   ├── ...                            (7 families × 5 ROIs = up to 35 tables)
├── metadata/
│   ├── metadata_unsegmented.parquet   the sheets as supplied
│   └── metadata_segmented.parquet
└── logs/
    ├── run.log                        full narrative of the run
    ├── errors.csv                     one row per failure, with traceback
    └── summary.json                   machine-readable outcome
```

Each feature table has:

- **index columns** — `image_id`, `stem`, `source`, `channel`, `pair_id`, `roi`, `image_path`,
  `roi_mask_path`, `roi_n_voxels`, `roi_n_components`, `roi_fraction`, `image_height`,
  `image_width`, `image_is_colour`
- **`meta_*` columns** — everything from your spreadsheets
- **feature columns** — prefixed by family (`pyrad_`, `dinov2_`, `texlab_`, …)

Plus **provenance in the Parquet file metadata**: model id, library versions, settings, and a
config fingerprint. So a table stays self-describing after it leaves the machine that made it:

```python
import pandas as pd
from usfeat.writer import read_provenance

df = pd.read_parquet("out/features/dinov2__lesion.parquet")
print(read_provenance("out/features/dinov2__lesion.parquet"))
```

---

## Errors: nothing stops the run

A bad image, an unloadable mask, an ROI too small for texture analysis, a model that will not
initialise — none of these abort anything. Each is caught, recorded, and the loop continues.

`logs/errors.csv` has one row per failure: timestamp, stage, extractor, image id, ROI, both
file paths, exception type, message, and the traceback. That is the file to send back when
something looks wrong.

To see the shape of a finished run at a glance:

```bash
docker run --rm -v /path/to/output:/out usfeat:latest verify --out /out
```

```
feature tables in /out/features:

  dinov2__lesion.parquet          6 rows    1550 cols  (9 metadata)  model=facebook/dinov2-base
  pyradiomics__lesion.parquet     6 rows    1170 cols  (9 metadata)  model=-
  ...
run summary: 16 images, 0 errors, finished 2026-09-08T23:14:12+00:00
```

Runs are **resumable**. If a long run is interrupted, re-run the same command: completed rows
are detected and skipped. `--no-resume` forces recomputation.

---

## Useful options

```bash
extract --extractors pyradiomics,dinov2    # only these families
extract --rois lesion,solid                # only these ROIs
extract --no-whole                         # skip the whole-image tables
extract --limit 10                         # smoke test
extract --device cuda                      # GPU (default: auto-detect)
extract --roi-mode both                    # see below
extract --no-resume                        # recompute everything
```

### How an ROI reaches the neural networks

`--roi-mode` controls the presentation, and it changes what the features mean:

- **`crop`** (default) — padded bounding-box crop, so the lesion fills the frame. Texture and
  internal structure dominate.
- **`mask`** — full frame with everything outside the ROI zeroed. Keeps position and scale
  context, at the cost of a lot of black.
- **`both`** — emits both, column-suffixed `_crop` and `_mask`. Doubles the feature count.

For `whole` these collapse to the same thing, so it is computed once.

PyRadiomics and TexLab are unaffected by this — they always take image + binary mask directly.

---

## Running without Docker

```bash
pip install -r requirements.txt
pip install -e .
usfeat extract --data /path/to/data --out /path/to/out
```

TexLab additionally needs GNU Octave with the `statistics`, `image` and `parallel` packages,
plus the payload and its key. Everything else runs on the Python dependencies alone.

---

## Building the image

```bash
# 1. Package the proprietary TexLab source (run once, on a machine that has it).
python tools/pack_texlab.py \
  --texlab-dir /path/to/TexLAB_v3 \
  --saved-parameters /path/to/saved_parameters.mat \
  --out build/texlab.enc --key-out build/texlab.key

# 2. Build, baking in the weights and the payload key.
docker build -t usfeat:latest \
  --build-arg HF_TOKEN=hf_xxx \
  --build-arg TEXLAB_KEY="$(cat build/texlab.key)" .
```

`build/texlab.key` must never be committed — `.gitignore` and `.dockerignore` both exclude it.
See [docs/SECURITY.md](docs/SECURITY.md) for what the encryption does and does not protect against.

### Verifying TexLab in the built image

TexLab is the one component that needs Octave, so confirm it in the image before shipping:

```bash
docker run --rm \
  -v "$PWD/sample_data":/data:ro -v /tmp/texlab-check:/out \
  usfeat:latest extract --data /data --out /out --extractors texlab --limit 2
```

Expect `texlab__whole.parquet` and `texlab__lesion.parquet` with roughly 3,900 feature columns.
If `logs/errors.csv` shows `pkg load` failures instead, the `octave-statistics`,
`octave-image` or `octave-parallel` packages did not install — check the apt step.

The image is CPU-only by default. For a GPU host:

```bash
docker build -t usfeat:gpu \
  --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu121 ... .
```

---

## Sample data

`sample_data/` holds a 16-image pack in exactly the layout above, built from
[MMOTU](https://github.com/cv516Buaa/MMOTU_DS2Net) (public ovarian ultrasound, Apache-2.0):
10 unsegmented images (7 grayscale — 3 of them Doppler-removed counterparts — and 3 colour
Doppler) and 6 segmented images with all four ROI masks, plus both metadata spreadsheets.

Run against it to confirm your setup before touching real data:

```bash
./run.sh sample_data /tmp/usfeat-sample-out
```

One caveat, stated plainly: MMOTU ships **one** mask per image, the lesion. The
locule / projection / solid masks in the sample pack are **derived from the lesion mask by
morphology and are not anatomically meaningful**. They exist so the four-ROI code path is
exercised end to end. The lesion masks are real.

---

## Licence

The pipeline code is proprietary to JuveX.AI. TexLab is proprietary and ships encrypted.
MMOTU sample data is Apache-2.0. Model weights carry their own upstream licences
(DINOv2/DINOv3: Meta; BiomedCLIP: MIT; SigLIP: Apache-2.0).
