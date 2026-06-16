# InstanceDepth — Complete Reproduction Runbook

End-to-end guide to reproduce **everything in the paper** (ICCV 2025, *Instance-Level
Video Depth in Groups Beyond Occlusions*) on your custom RGB-D dataset:
GID-style data construction (§3), the three-phase training schedule (§4.3),
evaluation (§5.1 / Table 2), all three ablation studies (Tables 4–6), inference,
and the NYUDv2 cross-dataset check (Table 3).

Each stage states **what it reproduces in the paper**. Run every command from the
repository root (`~/InstanceDepthRepo`) after Step 2.

> **Two dataset-specific edits are mandatory before training** (Steps 3 and 5):
> your depth folder names (`left_depth` / `left_depth_np`) and the image size
> (`--image-size 504 896`, because 720×1280 is not divisible by 14). They are not
> part of the four patched model files.

---

## Map: paper component → where you do it

| Paper component | Stage here |
|---|---|
| GID dataset construction: masks, ground masks, tracking IDs, 20% split, Fig. 3 stats (§3) | Stage A (Steps 5–6) |
| Phase 1 — Global Depth Range Pretraining, 55k, 1e-5 (§4.3) | Stage B, Phase 1 |
| Phase 2 — Instance Depth Layer Specialization, 25k, 1e-5 (§4.3) | Stage B, Phase 2 |
| Phase 3 — Occlusion-Aware Joint Refinement, 25k, 1e-6 (§4.3) | Stage B, Phase 3 |
| Test-set metrics RMS/REL/RMSlog/Log10/σ1–3 (Table 2) | Stage C |
| Two-stage ablation Baseline / +H / +H+I (Table 4) | Stage D.1 |
| Depth-range partitioning 1/2/3 m (Table 5) | Stage D.2 |
| Loss ablation L_obj / L_dist / both (Table 6) | Stage D.3 |
| Final depth prediction on new clips | Stage E (`infer.py`) |
| NYUDv2 with DA-V2 encoder (Table 3) | Appendix N |

---

## Step 0 — Hardware & expectations

- Linux (or Windows + WSL2) with an NVIDIA GPU. The paper used a single RTX 4090
  (24 GB). At `504×896`, plan **batch 4–8 for Phase 1** (instance head not run)
  and **batch 1–2 for Phases 2–3**. If you OOM, drop to `--image-size 378 672`.
- Total schedule = 55k + 25k + 25k = **105k iterations**. On a 4090 this is roughly
  1–3 days end to end depending on resolution/batch.
- **About target numbers:** Table 2's values (RMS 0.397, REL 0.045, σ1 0.983) are
  on the paper's GID (101.5k frames, hardware depth sensors). Your custom set is
  smaller and different, so **absolute numbers will differ**. The reproducible
  *claim* is the **relative** improvement Baseline → +H → +H+I (Table 4). Track that.

## Step 1 — Environment

```bash
conda create -n instdepth python=3.12 -y && conda activate instdepth
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r ~/instancedepth_delivery/requirements.txt
pip install -e /path/to/your/sam3          # your SAM3 checkout (annotation only)
python -c "import torch; print(torch.cuda.is_available())"      # must print True
```

## Step 2 — Assemble the repo and verify the patches

```bash
bash ~/instancedepth_delivery/setup_repo.sh \
     ~/flat_files ~/instancedepth_delivery ~/InstanceDepthRepo
cd ~/InstanceDepthRepo

# confirm the 4 patches are present (each must print a line):
grep -n symmetric_range_error instancedepth/models/hdi/holistic_depth.py
grep -n symmetric_range_error instancedepth/models/instance_depth.py
grep -n symmetric_range_error instancedepth/configs/run_config.py
grep -n "F.interpolate" instancedepth/models/instance/matcher.py
```

These were verified by execution: without the matcher patch, phase-2 matching
crashes (`mat1 and mat2 shapes cannot be multiplied`); without the signed Eq. 3,
the holistic depth collapses toward 0. If any grep is empty, re-copy the patched
files from `~/instancedepth_delivery/patched/`.

Drop `infer.py` (shipped with this runbook) into the repo root next to `train.py`.

## Step 3 — Edit the data-engine config (REQUIRED)

Open `instancedepth/configs/gid_custom.yaml`:

```yaml
dataset_root: "/abs/path/to/InstanceDepth/Dataset"   # contains Batch_1 .. Batch_10

layout:
  rgb_dir: "left_rgb"
  depth_png_dir: "left_depth"          # <-- your folder (default was left_filled)
  depth_npy_dir: "left_depth_np"       # <-- your folder (default was left_filled_np)

depth:
  unit: "auto"                         # pin to "mm" or "m" if auto guesses wrong
  min_depth_m: 0.01                    # paper Fig. 2 range
  max_depth_m: 10.0                    # MUST equal max_depth in instance_depth.yaml
  prefer_npy: true

sam3:
  backend: "native"                    # "native" | "hf" | "mock"
  checkpoint_path: "/abs/path/to/sam3.pt"
  device: "cuda"
  object_prompts: ["person"]           # your scenes are people; add "dog"/"ball" if present
  ground_prompts: ["floor", "ground"]
  min_object_score: 0.5

split:
  test_fraction: 0.20                  # paper §3: 20% of videos to the test set
  stratify_by_batch: true
```

If you add object prompts beyond `person`, also raise `instance.num_classes` in
`instancedepth/configs/instance_depth.yaml` to the number of categories.

---

# Stage A — Reproduce GID dataset construction (§3)

This builds, for every frame: per-instance masks with consistent track IDs
(`object_masks/*.png`, uint16, pixel = track id), a ground mask, a per-instance
GT **depth layer** (mean valid GT depth in the mask, §4.2.1), bounding boxes, the
video-level **20% test split**, and **Fig. 3** statistics. SAM3 replaces the
paper's SAM+DEVA two-stage pipeline (masks **and** IDs in one pass).

### A.1 Dry-run with no GPU (validate plumbing)

Temporarily set `sam3.backend: "mock"` in the YAML, then:

```bash
python -m instancedepth.data_engine.run_generate \
    --config instancedepth/configs/gid_custom.yaml --limit 1
```

PASS = no errors; a `gid_custom/` folder appears; the log shows
`depth unit auto-detect: median raw=… -> scale=…` with a sensible scale
(`0.001` if your depth is in millimeters, `1` if meters). If it's wrong, set
`depth.unit: "mm"` or `"m"`. Then set `backend: "native"` back.

### A.2 Pilot on 2 sequences, then LOOK at the overlays

```bash
python -m instancedepth.data_engine.run_generate \
    --config instancedepth/configs/gid_custom.yaml --limit 2
# inspect gid_custom/**/preview/*.jpg — masks should hug people, IDs stable across frames
```

Tune `sam3.min_object_score` (raise if specks, lower if missed people) and the
`object_prompts` until overlays look right. To re-run one specific sequence,
match its `Batch/timestamp` id (use your actual folder names):

```bash
python -m instancedepth.data_engine.run_generate \
    --config instancedepth/configs/gid_custom.yaml --only "Batch_3/20260120_094343"
```

### A.3 Full annotation run

```bash
python -m instancedepth.data_engine.run_generate \
    --config instancedepth/configs/gid_custom.yaml
```

This processes all sequences and then **finalizes**: writes `train.txt`,
`test.txt` (20% of videos, stratified by batch), and `meta.json` (Fig. 3a depth
histogram + Fig. 3b per-video object/frame stats). If you annotate in chunks with
`--limit`/`--only`, the split is *not* rebuilt until you run:

```bash
python -m instancedepth.data_engine.run_generate \
    --config instancedepth/configs/gid_custom.yaml --finalize
```

**Output layout:**
```
gid_custom/
  meta.json   train.txt   test.txt
  Batch_*/<timestamp>/
    object_masks/<frame>.png   ground_masks/<frame>.png   annotations.json   preview/
```

If SAM3's output keys differ from your repo version (API drift), adjust the key
names in one place: `instancedepth/data_engine/sam3_engine.py::_frame_payload`.

---

# Stage B — Three-phase training (§4.3)

The phases run in order; each initializes from the previous via `--init-from`.
`set_phase(n)` applies the paper's freezing. **Always pass `--image-size 504 896`**
(your 720×1280 aspect, divisible by 14 — no distortion). The raw data stays at
native resolution; the loader resizes RGB+depth+masks together.

### Phase 1 — Global Depth Range Pretraining (55k, LR 1e-5)
Trains the backbone + HDI (range decoder, heads, Eq. 1–4). Loss = SigLog on the
holistic depth + range-segmentation cross-entropy.

```bash
python train.py --model-config instancedepth/configs/instance_depth.yaml \
    --data-root gid_custom --phase 1 --image-size 504 896 \
    --batch-size 4 --out runs/phase1
```
Expect `loss` to fall with `siglog` and `range` terms. Defaults already encode
55k iters / LR 1e-5 (override with `--iters` / `--lr` only for experiments).

### Phase 2 — Instance Depth Layer Specialization (25k, LR 1e-5)
Freezes the depth encoder; trains the Mask2Former instance decoder with
Hungarian-matched mask/class/depth-layer losses (Eqs. 5–7).

```bash
python train.py --model-config instancedepth/configs/instance_depth.yaml \
    --data-root gid_custom --phase 2 --image-size 504 896 \
    --batch-size 2 --init-from runs/phase1/ckpt_final.pth --out runs/phase2
```
Watch `loss_mask`, `loss_dice`, `loss_class`, `loss_depth`. Lower batch here —
the instance head + 200 decoder queries are memory-heavy at full resolution.

### Phase 3 — Occlusion-Aware Joint Refinement (25k, LR 1e-6)
Freezes the instance decoder; fine-tunes encoder + decoder + Φ_o with
`L_ref = λ1·L_obj + λ2·L_dist` (Eqs. 10–12) plus a dense SigLog term that keeps
the holistic depth supervised.

```bash
python train.py --model-config instancedepth/configs/instance_depth.yaml \
    --data-root gid_custom --phase 3 --image-size 504 896 \
    --batch-size 2 --init-from runs/phase2/ckpt_final.pth --out runs/phase3
```
Early `obj=0 dist=0` lines are **normal** until the instance head clears the
pair-selection thresholds (cls>0.9, mask>0.8, IoU>0.1). Once pairs fire, `obj`
and `dist` become non-zero.

**Common knobs (all phases):** `--resume runs/phaseN/ckpt_XXXXXX.pth` to continue
an interrupted run; `--save-every`, `--log-every`; `--batch-size 1` or
`--image-size 378 672` if you hit `CUDA out of memory`.

### Optional but recommended — initialize the encoder from Depth-Anything-V2 (F.5)
The biggest accuracy lever on a small custom set (and what the paper does for
NYUDv2). The plumbing exists (`checkpoint.load_pretrained`); add this just after
the model is built in `train.py` (after `model = build_instance_depth_from_yaml(...)`),
gated on a new `--da-v2-weights` arg:

```python
if getattr(args, "da_v2_weights", None):
    sd = torch.load(args.da_v2_weights, map_location="cpu"); sd = sd.get("model", sd)
    from instancedepth.utils.checkpoint import load_pretrained
    load_pretrained(model, sd, src_prefix="pretrained.", dst_prefix="backbone.vit.")
```
Inspect your DA-V2 checkpoint's keys (`list(sd.keys())[:5]`) and adjust
`src_prefix` accordingly. Use it on Phase 1 only.

---

# Stage C — Evaluation (§5.1, Table 2 metrics)

Computes RMS, REL, RMSlog, Log10, σ1–3 on valid GT pixels, averaged over frames.
**Use the same `--image-size` as training.**

```bash
# Full model (final occlusion-rectified depth = the paper's reported model):
python evaluate.py --model-config instancedepth/configs/instance_depth.yaml \
    --data-root gid_custom --checkpoint runs/phase3/ckpt_final.pth \
    --split test --image-size 504 896 --out-json results_full.json
```

Lower is better for RMS / REL / RMSlog / Log10; higher is better for σ1–3.

---

# Stage D — Ablation studies (reproduce Tables 4, 5, 6)

### D.1 — Two-stage framework (Table 4: Baseline / +H / +H+I)

- **+H (Holistic Depth Init only):** evaluate the Phase-1 checkpoint's Stage-1
  output with `--use-init-depth`:
  ```bash
  python evaluate.py --model-config instancedepth/configs/instance_depth.yaml \
      --data-root gid_custom --checkpoint runs/phase1/ckpt_final.pth \
      --split test --image-size 504 896 --use-init-depth --out-json results_H.json
  ```
- **+H+I (full):** the Stage C command above (`results_full.json`).
- **Baseline (DA-V2 encoder + plain DPT decoder):** this is a vanilla depth model
  *without* HDI or instance rectification and is **not part of this repo**. Run a
  standard Depth-Anything-V2 + DPT depth head on the same `gid_custom` split to
  fill this row, or report only the +H vs +H+I delta (which is the paper's core
  contribution). State this clearly when you publish the table.

### D.2 — Depth-range partitioning (Table 5: 1 m / 2 m / 3 m)

The partition size sets `rd = max_depth / partition_meters` (10/5/3 ranges). The
paper runs this ablation **without** the instance stage, so train Phase 1 for each
setting and evaluate with `--use-init-depth`. For each value `K ∈ {1.0, 2.0, 3.0}`:

```bash
# edit instancedepth/configs/instance_depth.yaml -> hdi.partition_meters: K
python train.py --model-config instancedepth/configs/instance_depth.yaml \
    --data-root gid_custom --phase 1 --image-size 504 896 \
    --batch-size 4 --out runs/part_${K}
python evaluate.py --model-config instancedepth/configs/instance_depth.yaml \
    --data-root gid_custom --checkpoint runs/part_${K}/ckpt_final.pth \
    --split test --image-size 504 896 --use-init-depth --out-json results_part_${K}.json
```
The paper finds 2 m best; 1 m fragments depth regions, 3 m under-constrains.

### D.3 — Refinement losses (Table 6: L_obj / L_dist / both)

`train.py` constructs `RefinementCriterion()` with defaults
(`lambda_obj=1.0, lambda_dist=0.5`). To ablate, change that one line (around the
loss-setup block) for each run, then train Phase 3 and evaluate:

| Row | edit in `train.py` |
|---|---|
| `L_obj` only  | `ref_crit = RefinementCriterion(lambda_obj=1.0, lambda_dist=0.0)` |
| `L_dist` only | `ref_crit = RefinementCriterion(lambda_obj=0.0, lambda_dist=1.0)` |
| both (default)| `ref_crit = RefinementCriterion()` |

```bash
# after editing, for each variant:
python train.py --model-config instancedepth/configs/instance_depth.yaml \
    --data-root gid_custom --phase 3 --image-size 504 896 --batch-size 2 \
    --init-from runs/phase2/ckpt_final.pth --out runs/loss_variant
python evaluate.py --model-config instancedepth/configs/instance_depth.yaml \
    --data-root gid_custom --checkpoint runs/loss_variant/ckpt_final.pth \
    --split test --image-size 504 896 --out-json results_loss_variant.json
```
The paper finds `L_obj` dominant, `L_dist` smaller (it mainly helps large
main/guest depth gaps). All variants start from the same Phase-2 checkpoint.

---

# Stage E — Inference on new clips (final depth prediction)

`infer.py` (shipped with this runbook) runs the full pipeline and writes the
occlusion-rectified depth at the original resolution.

```bash
python infer.py \
    --model-config instancedepth/configs/instance_depth.yaml \
    --checkpoint   runs/phase3/ckpt_final.pth \
    --input        /path/to/new/frames_dir \
    --out-dir      predictions \
    --image-size   504 896 \
    --save-color --save-masks
```
Outputs per frame (restored to e.g. 720×1280): `predictions/depth_npy/*.npy`
(float32 meters), `predictions/depth_color/*.png` (heatmap), and
`predictions/masks/*.png` (uint16 instance ids). Add `--use-init-depth` to output
the Stage-1 holistic depth instead, or `--no-invert-color` to flip the colormap.
Accepts a single image, a directory, or a glob (`--input "clip/*.jpg"`).

---

# Step F — Sanity test of the whole stack (no real data)

```bash
python ~/instancedepth_delivery/test_e2e.py     # synthetic data engine -> matcher -> losses
```

---

# Appendix N — NYUDv2 cross-dataset experiment (Table 3, optional, advanced)

The paper trains the **Holistic Depth Initialization stage only**, on NYUDv2,
**initialized from the Depth-Anything-V2 encoder**. NYUDv2 has no instance/track
annotations, so this is HDI-only and does not use Stages A's instance outputs.

Recipe:
1. **Convert NYUDv2 to the `gid_custom` layout.** HDI only needs RGB + metric
   depth, so write, per frame, an `annotations.json` whose `frames` map points at
   the NYUDv2 RGB and depth files with `instances: []` (empty), plus
   `depth_scale_to_m` for the NYUDv2 unit, and empty/zero `object_masks` and
   `ground_masks`. Produce `train.txt` / `test.txt` from the official NYUDv2
   split. (This mirrors what `annotate.py` writes, minus instances.)
2. **Train Phase 1 with DA-V2 init** (wire `--da-v2-weights` as in Stage B):
   ```bash
   python train.py --model-config instancedepth/configs/instance_depth.yaml \
       --data-root nyud_custom --phase 1 --image-size 476 630 \
       --da-v2-weights /path/to/depth_anything_v2_vitl.pth \
       --batch-size 8 --out runs/nyud_phase1
   ```
   (`476×630` is divisible by 14 and close to NYUDv2's ~480×640; or use `518×686`.)
3. **Evaluate HDI output:**
   ```bash
   python evaluate.py --model-config instancedepth/configs/instance_depth.yaml \
       --data-root nyud_custom --checkpoint runs/nyud_phase1/ckpt_final.pth \
       --split test --image-size 476 630 --use-init-depth --out-json results_nyud.json
   ```
The paper notes the gains here are marginal vs. dynamic scenes (indoor depth is
already well-layered), which is the expected outcome.

---

# Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: instancedepth` | You're not in `~/InstanceDepthRepo`; `cd` there |
| Assertion: H,W must be divisible by 14 | Pass `--image-size 504 896` (or any ÷14 size) |
| `mat1 and mat2 shapes cannot be multiplied` in phase 2 | matcher patch missing — re-apply `patched/matcher.py` |
| Holistic depth collapses toward 0 / σ very low | Eq. 3 sign patch missing — verify `symmetric_range_error` |
| `CUDA out of memory` | `--batch-size 1`, or `--image-size 378 672` |
| Data engine finds no depth / empty annotations | Folder names in `gid_custom.yaml` (`left_depth`/`left_depth_np`) |
| Depth values ~1000× off | Pin `depth.unit: "mm"` or `"m"` in `gid_custom.yaml` |
| No instances annotated | Lower `sam3.min_object_score`; check prompts match your scenes |
| SAM3 output key mismatch | Edit `data_engine/sam3_engine.py::_frame_payload` |
| Phase-3 `obj=0 dist=0` forever | Instance head not confident yet; let Phase 2 train longer / verify masks |

---

# One-page command summary

```bash
# setup
bash setup_repo.sh ~/flat_files ~/instancedepth_delivery ~/InstanceDepthRepo && cd ~/InstanceDepthRepo
# (edit gid_custom.yaml: layout folder names + sam3 paths)

# data (Stage A)
python -m instancedepth.data_engine.run_generate --config instancedepth/configs/gid_custom.yaml

# train (Stage B)
python train.py --model-config instancedepth/configs/instance_depth.yaml --data-root gid_custom \
    --phase 1 --image-size 504 896 --batch-size 4 --out runs/phase1
python train.py --model-config instancedepth/configs/instance_depth.yaml --data-root gid_custom \
    --phase 2 --image-size 504 896 --batch-size 2 --init-from runs/phase1/ckpt_final.pth --out runs/phase2
python train.py --model-config instancedepth/configs/instance_depth.yaml --data-root gid_custom \
    --phase 3 --image-size 504 896 --batch-size 2 --init-from runs/phase2/ckpt_final.pth --out runs/phase3

# evaluate (Stage C) + HDI-only ablation (Table 4 +H)
python evaluate.py --model-config instancedepth/configs/instance_depth.yaml --data-root gid_custom \
    --checkpoint runs/phase3/ckpt_final.pth --split test --image-size 504 896 --out-json results_full.json
python evaluate.py --model-config instancedepth/configs/instance_depth.yaml --data-root gid_custom \
    --checkpoint runs/phase1/ckpt_final.pth --split test --image-size 504 896 --use-init-depth --out-json results_H.json

# inference (Stage E)
python infer.py --model-config instancedepth/configs/instance_depth.yaml \
    --checkpoint runs/phase3/ckpt_final.pth --input /path/to/frames --out-dir predictions \
    --image-size 504 896 --save-color --save-masks
```
