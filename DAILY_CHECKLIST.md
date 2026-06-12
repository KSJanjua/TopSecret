# InstanceDepth — Day-to-Day Execution Checklist

Use this while actively running the pipeline. Print it out or keep in another terminal tab.

---

## Pre-Execution Checklist

```bash
# 1. Activate environment
source ~/intern_storage/Ayush/.venv/bin/activate
which python  # should show .../venv/bin/python

# 2. Navigate to repo
cd ~/intern_storage/Ayush/InstanceDepthRepo

# 3. Verify SAM3 is installed
python -c "import sam3; print('SAM3 OK')"

# 4. Verify dataset exists
ls ~/intern_storage/Ayush/InstanceDepth/Dataset/Batch_1/ | head -3

# 5. Check config
grep -A 3 "dataset_root:" instancedepth/configs/gid_custom.yaml
```

If any step fails, STOP and debug before continuing.

---

## Data Annotation Workflow

### One-Time Setup
```bash
# Check if preview folder is enabled (optional but recommended)
grep "preview_dir:" instancedepth/configs/gid_custom.yaml
# If null, change to "preview" to see colored overlays
```

### Dry Run (validate setup, no annotation yet)
```bash
python -m instancedepth.data_engine.run_generate \
    --config instancedepth/configs/gid_custom.yaml \
    --only "Batch_1/20260120_094343" \
    --limit 1

# ✅ Success: gid_custom/Batch_1/20260120_094343/ appears
# ✅ Check: ls gid_custom/Batch_1/20260120_094343/preview/
# ✅ Open: gid_custom/Batch_1/20260120_094343/preview/*.jpg
```

**If annotations.json is present but you see masks, you're good.**

### Batch Annotation (one sequence at a time on limited GPU)

Option A: One specific sequence
```bash
python -m instancedepth.data_engine.run_generate \
    --config instancedepth/configs/gid_custom.yaml \
    --only "Batch_2/20260129_100833"

# Ctrl-C anytime → safe to resume; --skip-existing won't re-do it
```

Option B: Next unannotated sequence (safest for interrupted GPU)
```bash
# Run whenever GPU is free; stops if done
python -m instancedepth.data_engine.run_generate \
    --config instancedepth/configs/gid_custom.yaml \
    --skip-existing --limit 1

# Repeat this command as many times as needed
```

Option C: Whole batch at once
```bash
python -m instancedepth.data_engine.run_generate \
    --config instancedepth/configs/gid_custom.yaml \
    --only "Batch_3" --skip-existing
```

### After Every Sequence: Spot Check
```bash
# Pick a random frame from the sequence you just annotated
ls gid_custom/Batch_X/timestamp/preview/ | head -1 | xargs -I{} \
  file gid_custom/Batch_X/timestamp/preview/{}

# Open the .jpg — people should be colored consistently across frames
# (same person = same color)

# If overlays look bad, lower min_object_score and re-run with --only
```

### Once All 65 Sequences Are Annotated
```bash
python -m instancedepth.data_engine.run_generate \
    --config instancedepth/configs/gid_custom.yaml \
    --finalize

# ✅ Builds train.txt (80% = ~52 sequences)
# ✅ Builds test.txt (20% = ~13 sequences)
# ✅ Builds meta.json (statistics)

ls gid_custom/train.txt gid_custom/test.txt gid_custom/meta.json
```

---

## Training Workflow

### Phase 1: Global Depth Pretraining

**Duration:** ~2 days on A100 (adjust batch-size if OOM)
```bash
python train.py \
    --model-config instancedepth/configs/instance_depth.yaml \
    --data-root gid_custom \
    --phase 1 \
    --batch-size 4 \
    --out runs/phase1

# Watch for: loss should trend downward (sanity check)
# Every 5k steps: checkpoint saved
# Can interrupt and resume with --resume runs/phase1/ckpt_050000.pth
```

**Checkpoint locations:** `runs/phase1/ckpt_005000.pth`, `ckpt_010000.pth`, ..., `ckpt_final.pth`

**If OOM:**
```bash
# Try smaller batch
--batch-size 2
# or smaller image
--image-size 392 392
```

### Phase 2: Instance Depth Specialization

**Duration:** ~1 day
```bash
python train.py \
    --model-config instancedepth/configs/instance_depth.yaml \
    --data-root gid_custom \
    --phase 2 \
    --init-from runs/phase1/ckpt_final.pth \
    --batch-size 4 \
    --out runs/phase2

# Watch for: "trainable tensors: X / Y" should show fewer trainable params
#            (encoder is frozen in phase 2)
```

### Phase 3: Occlusion-Aware Refinement

**Duration:** ~1 day
```bash
python train.py \
    --model-config instancedepth/configs/instance_depth.yaml \
    --data-root gid_custom \
    --phase 3 \
    --init-from runs/phase2/ckpt_final.pth \
    --batch-size 4 \
    --out runs/phase3

# Watch for: "obj=0 dist=0" early on is NORMAL
#            These losses activate only when occlusion pairs pass selection thresholds
# Later: obj and dist should become non-zero
```

### Resume Training After Interruption
```bash
# Find latest checkpoint in the phase you're on
ls -la runs/phase2/ | grep ckpt_

# Resume with --resume
python train.py \
    --model-config instancedepth/configs/instance_depth.yaml \
    --data-root gid_custom \
    --phase 2 \
    --resume runs/phase2/ckpt_050000.pth \
    --batch-size 4 \
    --out runs/phase2

# Script continues from that step
```

---

## Evaluation Workflow

### After Phase 3 Complete

```bash
# Full model evaluation (phase 3 final checkpoint)
python evaluate.py \
    --model-config instancedepth/configs/instance_depth.yaml \
    --data-root gid_custom \
    --checkpoint runs/phase3/ckpt_final.pth \
    --split test \
    --out-json results_phase3.json

# Prints metrics: RMS, REL, RMSlog, Log10, σ₁, σ₂, σ₃
# Saves to results_phase3.json
```

### Ablation: Phase 1 Only (HDI baseline)

```bash
python evaluate.py \
    --model-config instancedepth/configs/instance_depth.yaml \
    --data-root gid_custom \
    --checkpoint runs/phase1/ckpt_final.pth \
    --split test \
    --use-init-depth \
    --out-json results_hdi_only.json
```

### Compare Your Results to Paper

Open the paper's Table 2 and compare:
```
Your results:     RMS=X.XX  REL=Y.YY  RMSlog=Z.ZZ  σ₁=A.AA
Paper (NYU):      RMS=0.49  REL=0.19  RMSlog=0.26  σ₁=0.92

(Your dataset and task are different, so exact match unlikely,
 but order-of-magnitude should be comparable)
```

---

## Common Issues & Quick Fixes

| Issue | Symptom | Fix |
|---|---|---|
| venv not activated | `ModuleNotFoundError: sam3` | `source ~/.venv/bin/activate` |
| OOM during training | `CUDA out of memory` | `--batch-size 2` or `--image-size 392 392` |
| OOM during annotation | SAM3 crash | Already handled by single-sequence runs; unlikely unless sequences are huge |
| Data not found | `FileNotFoundError: /path/to/Dataset` | Check `dataset_root:` in config; must be absolute path |
| Wrong depth units detected | `scale=0.001` but data is meters | Set `depth.unit: "m"` in config explicitly |
| preview/*.jpg not created | No preview folder | Set `preview_dir: "preview"` in config (not null) |
| Matching keeps failing | matcher.py crashes on GT masks | Use updated code from /mnt/user-data/outputs (patched matcher) |
| Phase 2/3 losses stay 0 | obj=0, dist=0, won't improve | Normal early on; wait for pairs to activate. If never improves, lower thresholds in config |

---

## Before You Talk to Your Mentor

Prepare these files to show:

```bash
# 1. Example annotation output
ls gid_custom/Batch_2/20260129_100833/

# 2. Mask visualizations
ls gid_custom/Batch_2/20260129_100833/preview/ | head -3

# 3. Annotations.json structure
head -50 gid_custom/Batch_2/20260129_100833/annotations.json

# 4. Documentation
cat COMPLETE_MENTOR_GUIDE.md
cat MENTOR_PRESENTATION_SLIDES.md

# 5. Code structure
ls -la instancedepth/data_engine/*.py
```

---

## Logging to File (Optional But Recommended)

To keep a record of training progress:

```bash
# Phase 1
python train.py ... --phase 1 ... 2>&1 | tee logs/phase1.log

# Phase 2
python train.py ... --phase 2 ... 2>&1 | tee logs/phase2.log

# Phase 3
python train.py ... --phase 3 ... 2>&1 | tee logs/phase3.log

# Later, review:
tail -20 logs/phase3.log
grep "step\|loss" logs/phase3.log | tail -30
```

---

## Checkpoints Directory Structure (After Training)

```
runs/
├── phase1/
│   ├── ckpt_005000.pth
│   ├── ckpt_010000.pth
│   ├── ... (every 5k iters)
│   └── ckpt_final.pth
├── phase2/
│   ├── ckpt_005000.pth
│   ├── ...
│   └── ckpt_final.pth
└── phase3/
    ├── ckpt_005000.pth
    ├── ...
    └── ckpt_final.pth  ← USE THIS FOR EVALUATION
```

---

## Disk Space Estimate

| Component | Size | Notes |
|---|---|---|
| RGB frames (65 seqs × ~200 frames × ~2MB) | ~30 GB | Your raw data |
| Depth maps (npy + png) | ~15 GB | Your raw data |
| Generated annotations (masks, depths, meta) | ~35 GB | gid_custom/ folder |
| Training checkpoints (3 phases) | ~10 GB | Can delete intermediate phases |
| **Total** | **~90 GB** | Make sure you have it |

---

## Mental Checklist for "Is Everything Working?"

- [ ] discover.py found all 65 sequences (check log: "Discovered 65 sequences")
- [ ] Depth unit auto-detected (check log: "median raw=... -> scale=...")
- [ ] SAM3 ran three times per sequence (person, floor, ground)
- [ ] Instances found per sequence (track count >= 1)
- [ ] Preview overlays show people in consistent colors across frames
- [ ] object_masks are uint16 (viewer shows black, but `cv2.imread(..., IMREAD_UNCHANGED)` reads values)
- [ ] annotations.json has instance counts matching object_masks unique values
- [ ] train.txt / test.txt split 80/20 of video count
- [ ] Phase 1 loss trends downward
- [ ] Phase 2 matcher doesn't crash (GT masks resized internally)
- [ ] Phase 3 fine-tunes with lower learning rate (1e-6)
- [ ] Evaluation produces six metrics on test split

If all ✅, you're golden.

---

## Final Reminders

1. **One sequence at a time** — use `--only` or `--limit 1` to avoid GPU exhaustion
2. **Always check previews** — colored overlays are your ground-truth validator
3. **Finalize once, at the end** — don't run --finalize between individual sequences
4. **Phase order matters** — 1 → 2 → 3 only; freezing is per-phase
5. **Resume is safe** — `--resume` on any phase picks up where you left off
6. **Your GPU RAM is your constraint** — `--batch-size` and `--image-size` are your knobs

---

## Where to Find Help

| Question | File/Resource |
|---|---|
| "What does discover.py do?" | COMPLETE_MENTOR_GUIDE.md, Part 3 |
| "Why frame-number pairing?" | MENTOR_PRESENTATION_SLIDES.md, Slide 4 |
| "How to resume training?" | This file, "Resume Training After Interruption" |
| "What if I hit OOM?" | This file, "Common Issues" |
| "Show me the architecture" | PATCHES.md, "Code Audit" |
| "Paper vs. our code" | COMPLETE_MENTOR_GUIDE.md, Part 1 |

---

Good luck! You've got this. 🚀
