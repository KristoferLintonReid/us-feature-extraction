# Sample data

A 16-image pack in exactly the folder layout the pipeline expects. Use it to confirm your
Docker setup works before pointing the pipeline at real data.

```bash
./run.sh sample_data /tmp/usfeat-sample-out
```

Expect roughly 5–15 minutes on CPU, and 35 Parquet files in the output.

## What is in here

| | Count | Notes |
|---|---|---|
| `images_grayscale/` | 7 | 4 grayscale-only, plus 3 Doppler-removed counterparts |
| `images_doppler/` | 3 | colour Doppler; same filenames as their Doppler-removed pair |
| `images_seg/img/` | 6 | the segmented subset, numbered `0.tiff`–`5.tiff` |
| `images_seg/{lesion,locule,projection,solid}/` | 6 each | one PNG mask per ROI per image |
| `metadata_unsegmented.xlsx` | 10 rows | `File Name`, `CDM ID`, `Center`, `Iota7_basic.histology`, `Iota7_basic.extra_colourscore` |
| `metadata_segmented.xlsx` | 6 rows | `Label`, `Histology` |

The three Doppler images share filenames with three of the grayscale images on purpose — that
is the "Doppler + Doppler-removed" pairing, and it exercises the `pair_id` logic.

## Provenance, and one important caveat

Images come from **[MMOTU](https://github.com/cv516Buaa/MMOTU_DS2Net)** (OTU_2d), a public
ovarian tumour ultrasound dataset released under **Apache-2.0**. It is the closest open
analogue to adnexal mass ultrasound.

**MMOTU ships one mask per image: the lesion.** The `locule`, `projection` and `solid` masks
here are **derived from the lesion mask by morphological operations** — erosion for `solid`, a
distance-transform threshold for `locule`, a boundary band for `projection`. They are
**anatomically meaningless**. They exist so that the four-ROI code path is exercised
end to end on real image data. The `lesion` masks are genuine.

The `Histology` and `Iota7_basic.*` columns are mapped from MMOTU's 8-class tumour labels into
the shape of the real metadata schema. They are structurally realistic, not clinically
meaningful. Treat this pack as a plumbing test, never as a validation set.

Rebuild it at any time with:

```bash
python tools/build_sample_data.py --src <mmotu-checkout> --out sample_data
```
