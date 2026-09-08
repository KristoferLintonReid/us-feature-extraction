# Example output

Produced by running the pipeline over 5 images from `sample_data/` with default settings:

```bash
usfeat extract --data sample_data --out examples/output --limit 5
```

- **30 feature tables**, 102 rows total
- **0 errors**
- runtime 117.9s on CPU

The `.parquet` files themselves are not committed (they are derived data, ~25 MB). Regenerate them with the command above. What is committed:

- `tables.json` — every table's shape, feature count and model id
- `excerpts/*.csv` — the real first 3 rows of each table, with identity, metadata and the first 6 feature columns
- `../output/logs/` and `../output/metadata/` — the complete run logs and the metadata sheets as Parquet

## Tables, with and without ROI

`whole` is the no-ROI case: features over the entire frame. The other four are ROI-specific. Every extractor produces all five.

| extractor | ROI | rows | features | model |
|---|---|---|---|---|
| `biomedclip` | `whole` | 5 | 512 | hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-… |
| `biomedclip` | `lesion` | 3 | 512 | hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-… |
| `biomedclip` | `locule` | 3 | 512 | hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-… |
| `biomedclip` | `projection` | 3 | 512 | hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-… |
| `biomedclip` | `solid` | 3 | 512 | hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-… |
| `dinov2` | `whole` | 5 | 1,536 | facebook/dinov2-base |
| `dinov2` | `lesion` | 3 | 1,536 | facebook/dinov2-base |
| `dinov2` | `locule` | 3 | 1,536 | facebook/dinov2-base |
| `dinov2` | `projection` | 3 | 1,536 | facebook/dinov2-base |
| `dinov2` | `solid` | 3 | 1,536 | facebook/dinov2-base |
| `dinov3` | `whole` | 5 | 1,536 | facebook/dinov3-vitb16-pretrain-lvd1689m |
| `dinov3` | `lesion` | 3 | 1,536 | facebook/dinov3-vitb16-pretrain-lvd1689m |
| `dinov3` | `locule` | 3 | 1,536 | facebook/dinov3-vitb16-pretrain-lvd1689m |
| `dinov3` | `projection` | 3 | 1,536 | facebook/dinov3-vitb16-pretrain-lvd1689m |
| `dinov3` | `solid` | 3 | 1,536 | facebook/dinov3-vitb16-pretrain-lvd1689m |
| `imagenet` | `whole` | 5 | 2,816 | resnet50.a1_in1k,vit_base_patch16_224.augre… |
| `imagenet` | `lesion` | 3 | 2,816 | resnet50.a1_in1k,vit_base_patch16_224.augre… |
| `imagenet` | `locule` | 3 | 2,816 | resnet50.a1_in1k,vit_base_patch16_224.augre… |
| `imagenet` | `projection` | 3 | 2,816 | resnet50.a1_in1k,vit_base_patch16_224.augre… |
| `imagenet` | `solid` | 3 | 2,816 | resnet50.a1_in1k,vit_base_patch16_224.augre… |
| `pyradiomics` | `whole` | 5 | 1,147 | v3.1.0 |
| `pyradiomics` | `lesion` | 3 | 1,147 | v3.1.0 |
| `pyradiomics` | `locule` | 3 | 1,147 | v3.1.0 |
| `pyradiomics` | `projection` | 3 | 1,147 | v3.1.0 |
| `pyradiomics` | `solid` | 3 | 1,147 | v3.1.0 |
| `siglip` | `whole` | 5 | 1,536 | google/siglip-base-patch16-224 |
| `siglip` | `lesion` | 3 | 1,536 | google/siglip-base-patch16-224 |
| `siglip` | `locule` | 3 | 1,536 | google/siglip-base-patch16-224 |
| `siglip` | `projection` | 3 | 1,536 | google/siglip-base-patch16-224 |
| `siglip` | `solid` | 3 | 1,536 | google/siglip-base-patch16-224 |

## Why row counts differ

`whole` covers every image. The ROI tables cover only the segmented subset, and only where that ROI exists and is large enough for the extractor. That is expected, and the run summary accounts for every difference.

## Column layout

Every table has the same three blocks, in this order:

1. **identity** — `image_id`, `stem`, `source`, `channel`, `pair_id`, `roi`, plus ROI geometry (`roi_n_voxels`, `roi_n_components`, `roi_fraction`) and frame size
2. **`meta_*`** — every column from your spreadsheets
3. **features** — prefixed by family

For example, `pyradiomics__lesion.parquet` has 1,147 feature columns beginning `pyrad_diag_Versions_PyRadiomics` and ending `pyrad_lbp-2D_ngtdm_Strength`.

## Provenance

Each file also carries its own provenance in the Parquet metadata — model id, library versions, settings, config fingerprint:

```python
from usfeat.writer import read_provenance
read_provenance('examples/output/features/dinov3__lesion.parquet')
```
