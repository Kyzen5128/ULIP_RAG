# IKEA → ULIP_RAG 2.0 data loop

This directory builds a new, versioned IKEA product dataset without reading or
overwriting the legacy 733-product gallery. The current snapshot root is:

```text
/mnt/P300/data/ULIP/ULIP_RAG_2_0/ikea-us-20260716-v1
```

The filesystem JSONL snapshot is the source of truth. MongoDB is not part of
this pipeline and is not modified by the normal commands.

## Stages and gates

```text
IKEA US PLP API
  → strict type taxonomy / accessory rejection
  → immutable catalog/products_plp.jsonl + catalog/rejected.jsonl
  → product-page enrichment (official dimensions + DIMMA GLB URL)
  → derived catalog/products.jsonl (never overwrites crawl-stage rows)
  → structural and semantic catalog QA
  → byte-validated primary/context images and official GLBs
  → official GLB decode + metric extent QA
  → missing-GLB TRELLIS stratified QA (not automatic approval)
  → canonical GLB + deterministic 8192-point PLY
  → 12-view render + corrupt/blank-render QA
  → ULIP-compatible JSON samples + self-generated product RAG corpus
```

Validation is fail-closed:

- Unknown product types go to `rejected.jsonl`; marketing text cannot promote
  an accessory into a furniture category.
- Missing dimensions remain null. They are never replaced with zero.
- Official GLBs whose extents differ by more than 12% from official product
  dimensions are quarantined.
- TRELLIS output is always `needs_visual_qa`; rescaling a generated mesh to
  official dimensions is not independent evidence that its shape is correct.
- Only geometry status `ok`, a valid primary image, and a complete render set
  can enter the generated training samples.
- Product front direction and clearance requirements remain unknown until a
  separately reviewed source provides them.

Visual promotion is an explicit, hash-bound step: use
`prepare_geometry_review.py` to create a pending decision queue and
`apply_geometry_review.py` to publish a separate reviewed manifest. Neither
tool mutates source geometry state, and `pending` rows remain ineligible.

## Main commands

Use the ULIP environment except for TRELLIS generation:

```bash
/home/kyzen/miniconda3/envs/ulip/bin/python ikea/loop2/crawl_catalog.py
/home/kyzen/miniconda3/envs/ulip/bin/python ikea/loop2/enrich_catalog.py --save-html
/home/kyzen/miniconda3/envs/ulip/bin/python ikea/loop2/validate_catalog.py --require-details
/home/kyzen/miniconda3/envs/ulip/bin/python ikea/loop2/audit_catalog_semantics.py
/home/kyzen/miniconda3/envs/ulip/bin/python ikea/loop2/download_assets.py
/home/kyzen/miniconda3/envs/ulip/bin/python ikea/loop2/make_qa_contact_sheets.py
/home/kyzen/miniconda3/envs/ulip/bin/python ikea/loop2/verify_snapshot.py --through assets
/home/kyzen/miniconda3/envs/ulip/bin/python ikea/loop2/build_style_pairs.py
/home/kyzen/miniconda3/envs/ulip/bin/python ikea/loop2/build_product_spatial_metadata.py
/home/kyzen/miniconda3/envs/ulip/bin/python ikea/loop2/canonicalize_geometry.py
/home/kyzen/miniconda3/envs/ulip/bin/python ikea/loop2/render_catalog.py --workers 8
/home/kyzen/miniconda3/envs/ulip/bin/python ikea/loop2/build_training_dataset.py
/home/kyzen/miniconda3/envs/ulip/bin/python ikea/loop2/verify_snapshot.py --through dataset
/home/kyzen/miniconda3/envs/ulip/bin/python ikea/loop2/build_rag_index.py --split train
/home/kyzen/miniconda3/envs/ulip/bin/python ikea/loop2/build_rag_index.py --split all
```

TRELLIS fallback is deliberately sampled before any full run:

```bash
PYTHONPATH=/home/kyzen/TRELLIS \
  /home/kyzen/miniconda3/envs/TRELLIS/bin/python \
  ikea/loop2/generate_trellis.py --sample-per-category 2
```

An official GLB that already failed canonicalization may use the explicit,
provenance-preserving fallback flag. Mere presence or a dimension quarantine
does not qualify:

```bash
PYTHONPATH=/home/kyzen/TRELLIS \
  /home/kyzen/miniconda3/envs/TRELLIS/bin/python \
  ikea/loop2/generate_trellis.py --include-official-errors --only-id PRODUCT_ID
```

For a full TRELLIS review, render all rows first, then create a source-filtered
hash-bound queue and paginated sheets. Official dimension quarantines remain
excluded rather than being visually promoted:

```bash
/home/kyzen/miniconda3/envs/ulip/bin/python ikea/loop2/prepare_geometry_review.py \
  --geometry-manifest /path/to/geometry_rows.jsonl \
  --render-manifest /path/to/render_rows.jsonl \
  --source trellis --output /path/to/review_queue.jsonl
/home/kyzen/miniconda3/envs/ulip/bin/python ikea/loop2/make_geometry_qa_sheets.py \
  --geometry-manifest /path/to/geometry_rows.jsonl --source trellis \
  --samples-per-group 0 --rows-per-sheet 10 --output-tag trellis_full
```

## Coordinate and scale conventions

- Catalog dimensions: millimetres; width/depth/height are separate nullable
  fields.
- Blender canonical workspace: X=width, Y=depth, Z=up.
- glTF/GLB on disk: standard glTF Y-up conversion.
- PLY used by the existing IKEA ULIP loader: X=width, Y=up,
  Z=depth with its sign/front direction unverified.
- Existing ULIP point preprocessing still unit-normalizes point clouds, so
  exact size fit must remain a metadata/rule gate. Style, furniture relation,
  and human preference are the learned spatial-ranking targets.

## What this does not yet provide

- Verified canonical front or usable-side labels.
- Drawer/door/chair-pullout clearance annotations.
- Built and validated 3D-FRONT/3D-FUTURE canonical spatial-supervision
  artifacts. The raw datasets are now present under
  /mnt/P300/data/ULIP/datasets/3D-FRONT and
  /mnt/P300/data/ULIP/datasets/3D-FUTURE, but the approved Gate 1 raw audit
  and later canonical build have not been completed in the current repo
  records. Their download/usage terms still apply.
- Human room-product style or placement preference labels.
- A frozen, leakage-safe V2T physical-room split. The latest handoff records a
  48-capture subset, 47 provisional physical-room groups, and 9 DA3-completed
  captures; the physical-room mapping is not frozen and the subset remains for
  real-domain calibration/evaluation rather than serving as the sole
  style/relation training source.

Those missing datasets block the learned style/relation/preference stages, but
do not block rebuilding the new semantic ULIP gallery and deterministic size
gate.

## Leakage-safe RAG artifacts

The dataset builder assigns every product family and every product-specific
RAG document to the same deterministic 80/10/10 split. Stage 1 and Stage 2
must use `product_corpus_train.jsonl` plus `corpus_index_train.faiss`; the
trainer command additionally uses `--require-rag-split train` and fails closed
if a validation/test product fact is present. `product_corpus.jsonl` and the
`--split all` index are serving artifacts only, never RAG training input.

## Semantic training and held-out evaluation

`run_semantic_training_v2.sh` initializes from the hash-pinned IKEA 1.0
checkpoint and trains only approved 2.0 products. After building all three
modalities with `ikea/build_vectors.py`, evaluate the deterministic test split
without joining modalities by filesystem order:

```bash
ikea/loop2/run_semantic_training_v2.sh

/home/kyzen/miniconda3/envs/ulip/bin/python \
  ikea/loop2/evaluate_semantic_vectors.py \
  --vector-dir /path/to/semantic_ulip_v2 \
  --dataset-json-dir /path/to/snapshot/dataset/json \
  --split test \
  --output /path/to/snapshot/evaluation/semantic_test.json
```

The evaluator performs an exact product-ID join across point, image, and text
metadata and reports text/point and image/point Recall@1/5/10 plus MRR@10.
This is held-out cross-modal product retrieval, not V2T room-query, style, or
placement evaluation.
