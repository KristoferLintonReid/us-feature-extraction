# Architecture

## The shape of the problem

Six feature families, five ROI presentations, two disjoint halves of a dataset, two metadata
schemas, and a hard requirement that nothing aborts the run. The design falls out of that:
**one uniform unit of work**, and **containment at every boundary**.

## The unit of work

Everything reduces to a `(RoiView, Extractor)` pair.

```
Item          one image on disk, plus whatever masks were found for it
  ↓  build_views()
RoiView       one (image, binary mask, roi-name) triple
  ↓  extractor.extract()
dict          feature name -> scalar
  ↓  FeatureWriter
Parquet       features/{extractor}__{roi}.parquet
```

The whole-image case is not a special case. `whole` is a real `RoiView` whose mask covers the
entire frame, so the no-ROI tables come out of the identical code path as the ROI-specific
ones. Nothing branches on "does this have a mask".

## Module map

| Module | Responsibility |
|---|---|
| `config.py` | One dataclass tree, loaded from YAML, overridable from the CLI. Produces a `fingerprint()` — a hash of every setting that affects feature *values* — stamped into each output file. |
| `discovery.py` | Walks the tree, returns `Item`s. Owns the grayscale/Doppler pairing and the two naming conventions. |
| `imaging.py` | The messy part. TIFFs arrive 8-bit, 16-bit, RGB-that-is-really-gray, RGB-that-is-really-Doppler, multi-page. Everything funnels to `(H, W)` float32 plus optional RGB. |
| `roi.py` | Turns an `Item` + loaded image into `RoiView`s. Skips empty masks. |
| `metadata.py` | Loads both sheets, normalises keys leniently, prefixes columns `meta_`. |
| `extractors/` | One module per family, all behind `Extractor.extract(view) -> dict`. |
| `writer.py` | Buffered shard-flushing Parquet writer, one per (family, ROI). Owns dedup, ordering, provenance, and resume. |
| `pipeline.py` | The loop, and all the error containment. |

## Error containment

Three nested levels, each of which continues rather than propagating:

1. **Extractor construction** — a family that cannot initialise is logged and dropped from the
   run. Five families still produce data when the sixth has no weights.
2. **Image load** — a corrupt or unreadable file costs that one image, recorded in the ledger.
3. **Single extraction** — `(view, extractor)` failures are caught individually. One ROI too
   small for texture analysis does not cost the other ROIs, or the other families.

Two output channels, deliberately separate:

- `run.log` — the narrative, for reading top to bottom.
- `errors.csv` — one row per failure with stage, extractor, image id, ROI, both paths,
  exception type, message and traceback. Flushed per row, so a killed container still leaves a
  complete record.

`supports()` is distinct from failure: an ROI below `min_roi_voxels`, or with a degenerate
extent, is a *skip* (counted, logged at debug) not an *error*. Mixing those two would bury real
problems under expected ones.

## Why one file per (family × ROI)

The alternative — one wide table — is worse in every dimension that matters here. Families have
wildly different column counts (512 to ~2,800), different failure modes, and different row
counts (whole-image covers every image; `solid` covers only the segmented subset with a solid
component). Splitting means:

- a family that fails entirely costs one file, not the whole output;
- `resume` is per-file, so an interrupted run restarts at the right granularity;
- loading only DINOv2-on-lesion does not mean reading every family's columns at once;
- provenance is per-file and specific — model id, library version, settings.

Joining is trivial and always on `image_id`.

## Resume

`FeatureWriter.load_existing_ids()` reads the `image_id` column from the final table and any
surviving shards. The pipeline skips a `(view, extractor)` whose image id is already present.
Shards are merged on finalize, deduplicated on `(image_id, roi)` keeping the last write.

This means an interrupted run is restarted by re-issuing the same command. It also means
adding a new extractor to a completed output directory computes only the new family.

## Presentation choices worth knowing about

**Deep extractors resize to the backbone's native size without a centre crop.** The processors
for these models typically specify resize-to-256 then crop-to-224. Cropping would silently
discard lesion tissue near the frame edge, so the pipeline resizes directly to the square
input. ViTs handle this through interpolated position encodings, and it is applied uniformly,
so features stay comparable across images.

**ROI presentation is a real modelling choice**, exposed as `--roi-mode`. `crop` fills the frame
with the lesion (texture dominates); `mask` keeps position and scale but feeds the network a
lot of black. Neither is universally right, so both are available and `both` emits each,
column-suffixed.

**PyRadiomics runs at unit pixel spacing.** 2D ultrasound TIFFs carry no physical spacing that
can be trusted, so shape features are in pixel units. This is comparable across images from the
same machine and depth setting, and not comparable across different ones — a caveat that
belongs to the data, not the code.

**Doppler is not discarded.** Colour is detected by checking whether the RGB channels actually
differ, so a grayscale frame stored as RGB is not mistaken for Doppler. Where colour is real,
the neural extractors receive the RGB; PyRadiomics receives the luminance, since it is defined
on scalar intensity.
