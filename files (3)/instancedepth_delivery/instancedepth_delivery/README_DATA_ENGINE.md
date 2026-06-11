# GID-Style Data Engine for the Custom RGB-D Dataset

Reconstructs the **GID dataset construction pipeline** (paper Sec. 3) on your
own recordings, replacing the paper's SAM [27] + DEVA [13] two-stage
annotation with **SAM3 promptable-concept video segmentation + tracking**, and
emits annotations directly consumable by the uploaded InstanceDepth code.

```
RGB frames + sensor depth ──► SAM3 (one video session per text concept)
        │                          │ per-frame {obj_id → mask, score}
        │                          ▼
        │              identity.py: cross-concept merge → IoU dedup
        │                          → highest-IoU track re-linking   (paper Sec. 3)
        │                          → short-track filtering
        ▼                          ▼
 depth_io.py (unit→m)      annotate.py: per-frame
                              • object_masks/<f>.png   uint16 (pixel = track id)
                              • ground_masks/<f>.png   uint8 binary
                              • GT instance depth layer = mean valid GT depth
                                inside the mask                     (Sec. 4.2.1)
                              • bbox, area, category, score → annotations.json
                           split.py: 20% of videos → test            (Sec. 3)
                           statistics.py: Fig. 3a/3b analogs → meta.json
```

## Input layout (yours)

```
InstanceDepth/Dataset/
  Batch 1 … Batch 10/
    <timestamp e.g. 20260105_012545>/
      left_rgb/         RGB frames (.jpg/.png)
      left_filled/      16-bit depth PNGs
      left_filled_np/   float depth .npy
```

## Output layout (GID-style)

```
gid_custom/
  meta.json   train.txt   test.txt
  Batch 1/<timestamp>/
    object_masks/<frame>.png    # uint16 id map, 0 = background
    ground_masks/<frame>.png
    annotations.json            # per-frame instances: track_id, category_id,
                                # bbox_xyxy, area, depth_layer_m, depth_valid_px
    preview/                    # optional colored overlays
```

## Run

```bash
pip install -e .[notebooks] -q          # inside your sam3 checkout (native backend)
python -m instancedepth.data_engine.run_generate \
    --config instancedepth/configs/gid_custom.yaml
# Dry-run the orchestration without GPU/weights:
#   set sam3.backend: "mock" in the YAML, optionally --limit 1
```

## Train-time consumption

```python
from instancedepth.data.gid_dataset import (
    GIDDatasetConfig, GIDInstanceDepthDataset, collate_gid, build_refine_targets)

ds = GIDInstanceDepthDataset(GIDDatasetConfig(annotations_root="gid_custom",
                                              split="train"))
loader = DataLoader(ds, batch_size=2, collate_fn=collate_gid, shuffle=True)

batch = next(iter(loader))
out = model(batch["image"])                              # uploaded InstanceDepth
# Phase 1: hdi_losses.SigLogLoss(out["init_depth"], batch["depth"])
#          + range_segmentation_loss(out["range_logits"], batch["depth"], 10.0, 5)
# Phase 2: indices = matcher(out, batch["targets"]); criterion(out, batch["targets"], indices)
# Phase 3: dt, valid = build_refine_targets(out["refine_meta"]["pair_query_idx"],
#                                           out["refine_meta"]["batch_index"],
#                                           indices, batch["targets"])
#          refine_criterion(out["d_hat"][valid], dt[valid])
```

## Key design decisions and their confidence labels

| Decision | Label | Basis |
|---|---|---|
| Per-frame instance masks + consistent IDs + ground masks + bboxes | [Paper Specified] | Sec. 3, Fig. 2 |
| Identity repair via highest-IoU matching to prior masks | [Paper Specified] | Sec. 3 "matched to prior masks using the highest IoU" |
| 20% test split at the video level | [Paper Specified] | Sec. 3 |
| Metric clamp 0.01–10.0 m, MAX_d = 10 m | [Paper Specified] | Fig. 2 caption; model configs |
| GT instance depth layer = mean **valid** GT depth in mask | [Strongly Inferred] | Sec. 4.2.1 "depth layer … the average depth of the instance"; invalid-pixel exclusion mirrors the losses' `target > 0` masking |
| SAM3 replaces SAM + DEVA (one model does masks **and** IDs) | [Reasonable Assumption] | SAM3 video mode subsumes both pipeline stages; no annotator in the loop |
| One SAM3 session per text concept, merged afterwards | [Reasonable Assumption] | SAM3 constraint: a session tracks one concept (repo issue #206) |
| Ground mask from text prompts ("floor", "ground") instead of dot prompts | [Reasonable Assumption] | paper's dots were manual; text prompts are the automated analog |
| Overlap flattening: nearest depth layer wins the pixel | [Reasonable Assumption] | occluder is in front by definition; layer stats computed pre-flattening |
| Depth unit auto-detection (mm vs m), npy preferred over PNG | [Reasonable Assumption] | RealSense/Kinect convention (the sensors GID used) |

## Verified (synthetic end-to-end test)

Discovery over `Batch */<timestamp>/`, mm→m auto-detection, depth-layer GT
accuracy (3.00 m / 5.00 m discs recovered exactly), occluder-wins id maps,
uint16 mask round-trip, stratified split files, statistics JSON, dataset →
`collate_gid` → the uploaded `HungarianMatcher` (patched) →
`InstanceSetCriterion` → finite losses, and `build_refine_targets` →
`RefinementCriterion`. SAM3 backends (`native`, `hf`) are wired to the
documented APIs but require your GPU + checkpoints to exercise; the `mock`
backend covers the orchestration path in CI.
