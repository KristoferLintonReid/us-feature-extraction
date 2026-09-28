# Clinical metadata specification

What the pipeline needs alongside the images, and why each field earns its place.

Everything here is optional in the sense that the pipeline will run without it — missing columns
come through as nulls and the run summary reports how many rows were affected. But the features
are only as useful as the labels attached to them, so this is the list worth pushing on.

Send whatever exists in whatever format (`.xlsx`, `.csv`, `.tsv`); every column is carried into
every feature table with a `meta_` prefix. Nothing needs selecting in advance.

---

## Essential

Without these the feature tables cannot be analysed properly.

| Field | Why |
|---|---|
| `File Name` / `Label` | Join key to the images. Already supplied. |
| **`CDM ID` for the segmented set** | **Currently missing.** See "The patient-ID gap" below — this is the most important item on the page. |
| **Specific histological diagnosis** | The WHO subtype, not just benign/malignant. See below. |
| `Center` | Site is the single largest confounder in multi-centre imaging. Needed to test whether features generalise across sites rather than memorise them. |
| Reference standard used | Histology, expert consensus, or interval follow-up — and which, per case. A model validated against mixed standards without knowing which is which cannot be interpreted. |

### The patient-ID gap

The unsegmented sheet has `CDM ID`; the segmented sheet has only `Label` and `Histology`. That
creates three problems:

1. **Train/test splits leak.** Without a patient identifier, two images of the same lesion can
   land on opposite sides of a split, which inflates every performance estimate.
2. **The two halves cannot be joined.** There is no way to tell whether a segmented image is the
   same patient as an unsegmented one, so the same patient may be counted twice.
3. **Per-patient aggregation is impossible.** Reporting is stuck at image level even where
   patient level is the clinically meaningful unit.

A patient identifier on the segmented sheet — even a pseudonymised one, consistent with the
unsegmented sheet — resolves all three.

### On diagnosis granularity

The supplied `Iota7_basic.histology` is binary (benign / malignant), and the segmented sheet's
`Histology` has broader categories. Both collapse distinctions that matter:

- **Borderline tumours** sit between benign and malignant and behave differently from either.
  Folding them into one or the other makes the label noisy in exactly the region where
  discrimination is hardest and most clinically valuable.
- **Subtype matters for imaging.** A serous cystadenoma, a mature teratoma and an endometrioma
  are all "benign" and look nothing alike. A model learning "benign" is learning an average of
  unrelated appearances.

The specific diagnosis as recorded in the histopathology report is what is wanted here, free
text is fine — mapping to a coding system can happen at this end.

---

## Valuable

Materially improves what can be concluded, but analysis is possible without.

| Field | Why |
|---|---|
| Age at scan | Ovarian pathology prevalence is strongly age-dependent; also needed for fairness analysis. |
| Menopausal status | Changes both the differential and the interpretation of most descriptors. |
| FIGO stage (malignant cases) | Distinguishes early from advanced disease; early-stage performance is the clinically interesting number. |
| Laterality, and which images are the same lesion | Multiple images per lesion must be grouped, for the same leakage reason as patient ID. |
| CA-125 / ROMA | The realistic comparator is imaging *plus* markers, not imaging alone. |
| Existing risk scores (O-RADS, ADNEX, RMI, IOTA Simple Rules) | Gives a benchmark to beat that a reader will recognise. |
| Full IOTA descriptor set | Only `extra_colourscore` was supplied. The remaining descriptors are the established feature set and the natural baseline to compare learned features against. |
| Date of scan and date of surgery | Establishes the interval between imaging and reference standard. |

---

## If available

| Field | Why |
|---|---|
| Scanner manufacturer and model | Texture features are sensitive to acquisition. Without this, site effects and scanner effects cannot be separated. |
| Transducer, preset, depth, gain | Same reason, finer grained. |
| Transvaginal vs transabdominal | Different resolution and field of view; effectively different populations of image. |
| Whether callipers or annotations are burned into the image | Burned-in graphics are learnable shortcuts and need to be identified, not discovered later. |
| Ethnicity | Required for any fairness claim. Often unavailable; worth asking. |
| Image compression history | Whether images are as exported from the scanner or have been compressed and converted. This is not a minor point — texture features are materially affected by compression, and a model trained on uncompressed images can fail on compressed ones for reasons that have nothing to do with biology. |

---

## Format

No particular structure is required. Two sheets keyed on `File Name` and `Label` respectively is
exactly right and matches what has already been sent. Additional columns can simply be appended;
they will be picked up automatically.

Key matching is deliberately lenient: `MPO-35606-I0000010.tiff`, `MPO-35606-I0000010` and
` mpo-35606-i0000010 ` all resolve to the same row, and Excel's habit of writing label `12` as
`12.0` is handled.

If a field exists but only for a subset of cases, send it anyway — partial coverage is still
informative, and the run summary reports how complete each column is.
