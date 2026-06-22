# InstanceDepth Reproduction — Forensic Audit (No Code Changes)

**Scope:** full pipeline audit against *Liang et al., "Instance-Level Video Depth in Groups Beyond Occlusions", ICCV 2025*, using the paper text + every implementation file in the project. No code is changed here; this is the investigation you asked to approve before edits.

**Reading of the paper that grounds everything below (verified against the PDF):**

- Two stages: (4.1) **Holistic Depth Initialization (HDI)** with Eqs. 1–4; (4.2.1) **Instance Depth Layer Prediction** (Mask2Former head → mask, class, depth-layer; bipartite cost Eqs. 5–7); (4.2.2) **Occlusion-Aware Depth Refinement** (filter cls>0.9 & mask>0.8, overlap IoU>0.1, nearest-depth guest; ROIAlign F_obj∈R^{2×C×Hp×Wp} + geom priors; Φ_o MLP Eqs. 8–9; losses SigLog + L_dist Eqs. 10–12).
- Three training phases: P1 55k @1e-5 (train backbone+HDI); P2 25k @1e-5 (freeze depth encoder, train instance decoder); P3 25k @1e-6 (freeze instance decoder, fine-tune encoder+decoder+Φ_o).
- GID data: SAM with **bbox prompts for objects + dot prompts for ground**; **DEVA** for tracking; **multiple object categories** (humans, basketballs, rackets, animals); real sensor depth (RealSense D455 / Azure Kinect); range 0.01–10 m.

**One-line diagnosis:** the depth side (HDI) is roughly working; the *instance* side is mis-specified for your data in two compounding ways — **(1) the model has no real notion of "what is a foreground instance"** (single class `person`, no semantic negatives, no Mask2Former deep supervision), and **(2) the ground-truth it learns from is SAM3-quality with no gate** (incomplete limbs, merged people, many invalid depth layers). Phase 3 then faithfully amplifies both problems.

---

## PART 1 — Paper Faithfulness Audit

Status legend: **PS** = Paper Specified, **SI** = Strongly Inferred, **RA** = Reasonable Assumption.

| Module | Paper section | Impl. status | Potential issues found |
|---|---|---|---|
| DINOv2 backbone + DPT reassemble (`dinov2_dpt.py`) | §4.3 ("pretrained DINOv2"); baseline = DA-V2+DPT (§5.3) | SI (reassemble not described) | timm `get_intermediate_layers(reshape=False, norm=True)` must strip CLS/register tokens for the `reshape(b,embed,hg,wg)` to be valid; OK if it does. Layers [11,17,23] of 24 is a reasonable but unverified choice. Low risk. |
| Depth Range Feature Decoder (`range_decoder.py`, `patch_attention.py`) | §4.1, Fig. 5 (patch-conv→FC→patch-attn, P=4/8/16) | PS order/patches; RA internals | "Patch attention" mechanics undefined → implemented as windowless MHSA over PxP tokens then upsample. Defensible. Low risk. |
| HDI heads S/D/C (`heads.py`) | §4.1 Eq. 1; "lightweight conv net" | PS heads exist; SI S=softmax; RA Φ_d=2×1×1 conv | S as softmax forces R_i=Σ(C·S)∈[0,1] (see HDI below). Medium. |
| HDI iterative update Eqs. 1–4 (`holistic_depth.py`, `instance_depth.py`) | §4.1 Eqs. 1–4 | PS equations; SI #steps=#scales; **RA sign fix** | **Author already found that literal Eq. 3 (`E=2(R−1)·step`) can only *decrease* depth** and added `symmetric_range_error=True` (`E=(2R−1)·step`). This is a sensible, documented deviation. Keep it. |
| Instance head shell (`instance_head.py`) | §4.2.1, Fig. 7 | PS outputs; SI mask=⟨emb,feat⟩, class=Linear, depth=MLP; RA sigmoid·max_depth | Docstring promises `aux_outputs` but **none are produced** → no deep supervision (critical, see P3/P6). |
| Pixel decoder (`pixel_decoder.py`) | §4.2.1 (Mask2Former) | SI (FPN instead of MSDeformAttn) | Valid M2F config; acceptable. Low risk. |
| Transformer decoder / masked attn (`transformer_decoder.py`) | §4.2.1 (Mask2Former) | SI | **Masked-attention logits use raw `query·mask_features`, but the prediction head uses `mask_embed(query)·mask_features`** — two different projections; M2F uses the *same* embedding for both. Medium. **No per-layer outputs → no aux loss.** Critical. |
| Query Fusion (`query_fusion.py`) | §4.2.1, Fig. 7 ("query fusion") | PS existence; RA mechanics (gated cross-stream MLP) | Mechanics fully invented (paper doesn't define). Acceptable, low risk. |
| Hungarian matcher (`matcher.py`) | §4.2.1 Eqs. 5–7 | PS cost terms, L_d=smooth-L1; SI BCE+dice (full mask) | **Depth cost is raw metres (0–10) while class∈[−1,0], dice∈[0,1], mask~O(1)** → depth can dominate/destabilize matching. Medium. |
| Instance set criterion (`instance_losses.py`) | §4.2.1 | PS L_m/L_c/L_d; SI no-object=0.1 | No-object/index convention is **correct**. But: no point-sampling/per-mask normalization (M2F uses it); GT masks downsampled with **nearest** → thin limbs vanish in the *target*. **No aux losses.** Medium–High. |
| Occlusion pair selection (`pair_selection.py`) | §4.2.2 ¶1 | PS thresholds/IoU/nearest-guest; RA "cls conf"=max fg-softmax, "mask conf"=mean prob in mask | Logic is faithful. The *failure* is upstream (class head can't reject non-persons), not here. Medium. |
| ROIAlign extractor (`roi_extract.py`) | §4.2.2 Eq. 8 inputs | PS F_obj/G_obj; RA Hp=7, coord grid, "global depth"=init depth | **`spatial_scale` mapping is correct.** But **the "normalized coordinate" grid is identical for every ROI (0..1 within the box)** → carries zero absolute-position info; the paper almost certainly means image-normalized coords. Medium. |
| Relation reasoning Φ_o (`relation_reason.py`) | §4.2.2 Eqs. 8–9 | PS eqs; RA D_bar=D_obj | Eq. 9 with `D_bar=D_obj` gives D̂∈[0, 2·D_obj] → **±100% swing per step**, far larger than the ±1-bucket correction in Eq. 3. Faithful to the literal text but likely destabilizing. Medium. |
| Refinement losses (`refine_losses.py`) | §4.2.2 Eqs. 10–12 | PS SigLog + L_dist | SigLog on a handful of instance scalars is high-variance; otherwise faithful. Low–Medium. |
| Refine compositing (`occlusion_refine._composite`) | (implied) | RA | Adds a **constant Δ over the whole instance mask** (a layer shift). For false-positive masks this shifts large background regions → measured σ1 drop. Medium–High. |
| Three-phase trainer (`train.py`) | §4.3 | PS phases/iters/LRs/losses; RA AdamW/wd/clip/dense-P3 | Config loss weights are **not wired** (uses defaults); see P6. Medium. |
| Data engine: masks/tracking (`sam3_engine.py`, `identity.py`, `annotate.py`) | §3 | RA (SAM3 replaces SAM+DEVA) | Largest source of GT-quality problems (see P2/P3). High. |

---

## PART 2 — Dataset Adaptation Audit (GID → your RGB-D)

Your pipeline diverges from GID in ways that directly produce the symptoms.

**2.1 Categories collapsed to one (`gid_custom.yaml: object_prompts: ["person"]`, `instance_depth.yaml: num_classes: 1`).**
GID has multiple object categories. You annotate and train only `person`. Consequences:
- The classifier has exactly two logits `[person, no-object]`. It is **never shown a sofa or a car labeled "not a person."** The only negatives during training are *random unmatched queries*. So the head learns "**salient blob ⇒ person**," which is exactly why sofas/cars become "person" instances on your indoor/street footage. This is the single biggest driver of the Phase-3 false positives.

**2.2 SAM3 concept-video replaces SAM(box)+DEVA.**
- SAM3 "person" concept segmentation tends to (a) **merge adjacent same-concept people** into one mask and (b) **drop occluded limbs**. GID avoided this with *per-object box prompts* (one object at a time) + DEVA tracking. Your merged/incomplete GT teaches the model merged/incomplete instances → P2 symptoms #2–#5 are partly **baked into the labels**.
- One session per concept means cross-person identity is entirely SAM3's; `identity.py` only *re-links across temporal gaps* and never **splits a merged track** or **fixes an ID swap that happens without a gap**.

**2.3 Ground masks from text prompts, not dot prompts.**
GID used human dot prompts on the floor. You use SAM3 text `"floor"/"ground"` and `_ground_union` ORs all of them, then subtract objects. Quality is unverified; over- or under-segmented ground only matters if you use the ground channel for supervision/eval (currently optional), so this is lower priority — but worth a visual check.

**2.4 Depth source: `left_filled` (hole-filled) + auto unit detection.**
- `depth_io.detect_unit_scale` calls anything with median raw value >80 "millimetres." **If `left_filled` is actually an 8-bit colourized/normalized depth PNG, or in cm, this mis-scales everything** (e.g. a 0–255 viz with median 120 → "0.12 m"). Verify the real encoding of `left_filled` and `left_filled_np` before anything else — a unit error would silently corrupt every depth metric and depth-layer GT.
- "Filled" depth interpolates across holes — including **inside occlusion boundaries**, which is the exact region the paper's refinement targets. GID used raw sensor depth. Your depth-layer GT (mean depth in mask) is therefore biased by interpolated values.

**2.5 Many instances have invalid (0) depth layers.**
Your own `help.py`/`help_depth.py` were written to chase this: small/distant people are unrangeable, so `depth_layer_m = 0`, and the dataset drops them (`require_valid_depth_layer`). Effects: fewer instances supervised in P2's L_d, and in P3 those instances can't be refine-supervised (`build_refine_targets` marks them invalid) → at inference their `pred_depth` is unconstrained.

**2.6 Resolution mismatch.**
`train.py`/`evaluate.py` default to **518×518 (square)**, but `infer_video.py` and `help_occlusion.py` use **504×896 (16:9)**, and `infer_video.target_size` comments call 504×896 "the training size." Square-squishing a 16:9 frame distorts human aspect ratios (contributing to odd/incomplete masks) and, more importantly, **train/eval/infer must use the same geometry**. Pin this down — the eval JSONs were produced at 518×518, which may not match how you trained or infer.

---

## PART 3 — Phase 2 Investigation (Instance Depth Layer Prediction)

For each component: purpose → expected → observed → causes (ranked).

**A. SAM annotation pipeline (GT mask + ID generation).**
*Purpose:* produce per-frame instance masks + consistent IDs that stand in for GID. *Expected:* clean per-person masks with full limbs and stable IDs. *Observed (your symptoms #1–#6):* incomplete masks, merged people, ID instability. *Causes:* (1) **SAM3 concept mode merges/under-segments people** [most likely]; (2) **no mask quality gate / no morphological repair** in `sam3_engine`/`annotate`; (3) `_flatten_id_map` "nearest-depth-wins" reassigns shared pixels between two people whose depth layers are close or `inf` → boundary bleed; (4) `repair_identities` `reid_iou=0.5` can merge two *different* people across a gap; (5) it cannot fix concurrent ID swaps.

**B. Mask2Former training (the model side).**
*Purpose:* learn N instance queries → masks/class/depth. *Expected:* crisp masks, calibrated class, separated instances. *Observed:* poor/merged masks regardless of GT. *Causes, ranked:* (1) **No deep supervision** — `instance_head`/`transformer_decoder` emit no per-layer predictions and `instance_losses` computes no aux loss. M2F's convergence and instance separation depend on this; without it masks are weak and queries collapse [most likely, model-side]. (2) **Masked-attention/prediction projection mismatch** (raw query vs `mask_embed`) destabilizes the mask-conditioned attention. (3) **Nearest-downsampled GT masks** drop thin limbs from the target itself. (4) BCE+dice without point sampling/normalization under-weights thin structures. (5) Aspect distortion from 518² squishing.

**C. Query matching / Hungarian assignment (`matcher.py`).**
*Purpose:* one-to-one pred↔GT. *Expected:* stable assignment dominated by mask+class. *Observed:* plausibly unstable early matching. *Causes:* **depth cost in raw metres** can dominate when depths differ (a 6 m gap = cost 6, dwarfing class∈[−1,0]); early in training pred depths are arbitrary, so matching is partly driven by noise. Normalize depth (÷max_depth) before it enters the cost. Medium.

**D. Depth-layer prediction & GT (`heads`/`annotate._depth_layer`).**
*Purpose:* scalar mean depth per instance. *Expected:* matches mean sensor depth in mask. *Observed:* many zeros / biased values. *Causes:* invalid sensor depth (2.5), filled-depth bias (2.4), and masks that don't match the true person silhouette (A).

**E. Tracking IDs — conceptual clarification.**
The **model is per-frame**; it predicts no track IDs. "Tracking consistency unstable" at inference is the instance head **flickering frame-to-frame** (expected for a per-frame model with weak masks) *plus* GT ID instability from (A). The paper's IDs exist only to make GT temporally consistent. Don't expect temporal stability from the current architecture; it has no temporal module.

---

## PART 4 — Phase 3 Investigation (Occlusion-Aware Refinement)

**Why sofas/cars/background become "instances":**
1. **Root cause: single foreground class with no semantic negatives** (Part 2.1). The class head outputs high `person` probability on any salient blob; sofas and cars are salient blobs.
2. **No deep supervision** → the class head is poorly calibrated and over-confident, so it clears even the strict 0.9 gate.
3. **Inference threshold is 0.5, not 0.9** (`infer_video.py: --score-thresh 0.5`, `--mask-thresh 0.5`). The masks you *see drawn* on sofas/cars are gated at 0.5 — far below the paper's 0.9 — so the overlay floods with false positives even before refinement.
4. **Domain gap:** your footage contains object classes absent from a person-only annotation; the model has never learned to treat them as background.

So `pair_selection.py` is not buggy — it is correctly forwarding garbage. At **training** time `build_refine_targets` *does* drop these false positives (no GT match → `valid=False`), which is why training looks "fine" but **inference** doesn't: there is no equivalent gate at inference beyond the (mis-trained) class score.

**Why occlusion reasoning / refinement is worse than expected (and the metrics agree):**
- `eval_phase3.json` vs `eval_phase1.json`: RMS 0.4236→**0.4202** (tiny gain) but **σ1 0.9345→0.9258** and REL 0.0749→**0.0755** (both worse). Refinement is **net-harmful on accuracy**.
- Causes, ranked: (1) **Constant-Δ compositing over the whole mask** (`_composite`) — for a false-positive sofa/car mask passing 0.9, a wrong constant is added to a large region, hurting σ1 [most likely for the σ1 drop]. (2) **Eq. 9 with D_bar=D_obj allows ±100% swings** → overcorrection. (3) **Coordinate prior carries no position** (`_normalized_coord_grid` identical per ROI) → Φ_o can't exploit vertical position, the strongest monocular depth cue → weak, near-random corrections. (4) **Unreliable `pred_depth` drives guest selection** ("nearest in depth") → wrong guest. (5) **Invalid GT depth layers** (2.5) mean the few real pairs are under-supervised. (6) Your own `help_occlusion.py` confirms refinement fires on only a small % of frames (overlapping confident people), so it can't help much globally but *can* hurt where it fires.

---

## PART 5 — Training Diagnostics

| Signal | Expected | Observed (from your JSONs / scripts) | Likely root cause |
|---|---|---|---|
| Phase-1 depth (RMS/REL/σ1) | ≈ paper Baseline+H (RMS 0.419, σ1 0.978) | RMS 0.424, REL 0.075, **σ1 0.935** | HDI structurally OK; σ1 gap from depth-GT quality (filled/holey, invalid instances), possible unit-scale risk (2.4), 1-m vs 2-m partition (P6), resolution. |
| Phase-2 metrics | should expose **instance** quality | **Identical to Phase 1** (depth frozen in P2; no instance metric computed) | **No instance metric exists** (`evaluate.py` only does dense depth). You are blind to mask/AP/IoU/tracking quality numerically. |
| Phase-3 metrics | should improve over P1 (paper +H+I: RMS 0.397, σ1 0.983) | RMS slightly better, **σ1 and REL worse** | Refinement degrades accuracy: false-positive masks + constant-Δ composite + weak Φ_o (Part 4). |
| Instance seg quality (qualitative) | crisp, separated persons | incomplete/merged | GT quality + no deep supervision (Part 3). |
| Tracking quality | n/a for a per-frame model | flicker | No temporal module; GT IDs unstable (Part 3E). |
| Occlusion-pair coverage | enough pairs to matter | small % of frames (your script) | Few overlapping high-confidence persons; refinement under-exercised. |

**What to add before retraining (diagnostics, not fixes):** mask AP / mean-IoU vs GT, a "false-positive instances on non-person pixels" count at inference, per-instance depth-layer error, and a histogram of `pred_logits` person-prob on background regions. Without these you cannot tell a fix from a regression.

---

## PART 6 — Code Audit (Prioritized Bug List)

Severity: 🔴 correctness/most-impactful · 🟠 important · 🟡 minor/quality.

1. 🔴 **No deep supervision (aux losses).** `instance_head.forward` returns no per-layer outputs; `transformer_decoder` discards intermediate query states; `instance_losses` has no aux loop. The single most important model-side defect for mask quality/instance separation.
2. 🔴 **`evaluate.py:118` ignores the region mask.** `--mask objects` builds `region` (lines 109–117) then calls `depth_metrics(pred[b], gt[b], max_d=max_depth)` **without `region=region`**. Person-restricted metrics silently equal full-frame metrics. All your JSONs are full-frame regardless of the flag.
3. 🔴 **Single foreground class / no negatives** (`num_classes:1`, `object_prompts:["person"]`). Not a "bug" in syntax but the root design mismatch driving sofa/car false positives.
4. 🟠 **Inference gate ≠ paper gate.** `infer_video.py` defaults `--score-thresh 0.5`/`--mask-thresh 0.5`; paper uses 0.9/0.8. Overlays show many false positives by construction.
5. 🟠 **Masked-attention projection mismatch** (`transformer_decoder._attn_mask` uses raw `query`; head uses `mask_embed(query)`). Inconsistent mask conditioning.
6. 🟠 **Depth cost unnormalized in matching** (`matcher.py:90`, raw metres) and in **`instance_losses` L_d** (smooth-L1 on metres with `w_depth=1`). Scale-imbalanced vs class/dice. Normalize by `max_depth`.
7. 🟠 **`_normalized_coord_grid` is identical for every ROI** (`roi_extract.py`) → no absolute position signal to Φ_o. Likely contradicts the paper's "normalized coordinates."
8. 🟠 **Constant-Δ composite over full mask** (`occlusion_refine._composite`) propagates false-positive masks into the dense map → measured σ1 drop.
9. 🟠 **Config loss weights not wired into training.** `train.py:122–123` build `HungarianMatcher()` and `InstanceSetCriterion(num_classes=…)` with **defaults**; `ref_crit = RefinementCriterion()` default. Any `instance_loss`/`refine_loss` block in YAML is ignored. (`build_instance_criterion`/`build_refinement_criterion` exist but are unused by `train.py`.)
10. 🟠 **Partition inconsistency.** Integrated `instance_depth.yaml` uses `partition_meters: 1.0` (rd=10), but `hdi_vitl.yaml` and paper Table 5 favor **2.0** (rd=5). The integrated model trains at the worse setting.
11. 🟠 **Depth unit auto-detect risk** (`depth_io.detect_unit_scale`, median>80⇒mm). Mis-scales if `left_filled` is 8-bit/cm/normalized. Verify and pin `depth.unit`.
12. 🟡 **GT masks downsampled with `nearest`** in matcher & criterion (`F.interpolate(..., mode="nearest")` at 259²) — thin limbs disappear from the target. Bilinear+threshold preserves more.
13. 🟡 **`_flatten_id_map` nearest-depth-wins** bleeds shared pixels between near-equal/`inf`-depth people → boundary errors in stored ID maps and thus GT.
14. 🟡 **`repair_identities` `reid_iou=0.5`** may merge distinct people across a gap; can't split concurrent merges or fix gapless ID swaps.
15. 🟡 **Eq. 9 swing range** (`D_bar=D_obj` ⇒ D̂∈[0, 2·D_obj]). Faithful to the literal text but a very large per-step correction; candidate for instability.
16. 🟡 **Dataset doesn't clamp `min_depth`** (`gid_dataset` zeroes `<0`/`>max` only) while `annotate` zeroed `<0.01`. Tiny (0,0.01) values survive on the dataset side — negligible but inconsistent.
17. 🟡 **`discover._pair_by_stem_or_index`** can return a partial/mismatched RGB↔depth map if stems coincidentally match for some frames and frame numbers aren't unique; relies on warnings.
18. 🟡 **timm token assumption.** `dinov2_dpt` reshape assumes `get_intermediate_layers` returns exactly `hg*wg` tokens (no register tokens). Version-dependent; would error loudly if wrong.
19. 🟡 **SigLog on instance scalars** is high-variance with few pairs (`refine_losses`).
20. 🟡 **No mask-area/no-positive guard around invalid depth** propagating into `e_obj` clamp paths — currently safe but fragile.

---

## PART 7 — Root Cause Analysis (Top 20, ranked)

Format: **#. Cause — Probability · Evidence · Impact · Fix difficulty.**

1. **Single foreground class + no semantic negatives** — *Very High* · `num_classes:1`, `object_prompts:["person"]`, sofa/car false positives · *Severe* (drives all P3 false positives) · *Medium* (re-annotate with more concepts / add negative supervision).
2. **No Mask2Former deep supervision** — *Very High* · head/decoder/criterion produce/consume no aux outputs · *Severe* (poor & merged masks) · *Medium*.
3. **SAM3 GT quality (merged people / missing limbs / no gate)** — *Very High* · concept-video behavior + no repair in `annotate` · *Severe* (GIGO into P2) · *Medium–High*.
4. **Inference threshold 0.5 (not 0.9/0.8)** — *High* · `infer_video.py` defaults · *High* (visualized false positives) · *Trivial*.
5. **Constant-Δ composite spreads false positives into dense depth** — *High* · `_composite` + σ1 0.935→0.926 · *High* · *Medium*.
6. **Depth-layer GT invalid for many instances** — *High* · your `help.py`/`help_depth.py` · *High* (weak L_d/refine supervision) · *Medium* (depth completion / robust stat).
7. **Coordinate prior carries no position** — *High* · identical per-ROI grid in `roi_extract` · *High* (weak occlusion reasoning) · *Easy*.
8. **Unnormalized depth cost in matching** — *Medium–High* · raw metres in `matcher`/`L_d` · *Medium* (unstable early matching) · *Easy*.
9. **Filled (interpolated) depth at occlusion boundaries** — *Medium–High* · `left_filled` naming · *Medium* (biases exactly the refined region) · *Medium*.
10. **Train/eval/infer resolution mismatch (518² vs 504×896)** — *Medium–High* · `train`/`evaluate` 518², `infer`/`help_occlusion` 504×896 · *Medium* · *Easy* (standardize).
11. **Depth unit mis-scale risk** — *Medium* (conditional, but catastrophic if true) · `detect_unit_scale` heuristic · *Severe-if-triggered* · *Trivial* (verify+pin).
12. **Masked-attn vs prediction projection mismatch** — *Medium* · `_attn_mask` raw query · *Medium* · *Easy*.
13. **No instance/tracking metric → blind evaluation** — *Medium* · `evaluate.py` depth-only; P2≈P1 · *Medium* (can't measure fixes) · *Easy*.
14. **Eq. 9 ±100% correction range** — *Medium* · `D_bar=D_obj` · *Medium* (overcorrection) · *Easy* (try bounded scale).
15. **Identity merge/split limitations** — *Medium* · `identity.repair_identities` gap-only · *Medium* (ID instability) · *Medium*.
16. **Nearest-downsampled GT masks** — *Medium* · matcher/criterion interpolate · *Medium* (limbs lost in target) · *Easy*.
17. **`evaluate.py` region bug** — *Medium* · line 118 · *Medium* (mismeasured, hides P2 quality) · *Trivial*.
18. **Config loss weights ignored by trainer** — *Low–Medium* · `train.py` uses defaults · *Low–Medium* (can't tune) · *Trivial*.
19. **1-m partition (rd=10) vs Table-5-best 2-m** — *Low–Medium* · config mismatch · *Low* · *Trivial*.
20. **Per-frame model expected to be temporally stable** — *Low (misexpectation, not a bug)* · no temporal module · *Low* · *N/A* (set expectations or add a tracker post-hoc).

---

## PART 8 — Fix Plan (no code; ordered, with rationale + verification)

Work in this order — earlier items unblock measurement of later ones.

**Tier 0 — Stop measuring blind (do first; cheap).**
- **F0.1 Fix `evaluate.py:118`** to pass `region=region`. *Why:* you currently cannot measure person-region depth at all. *Verify:* `--mask objects` and `--mask all` should now differ.
- **F0.2 Add instance metrics** (mask mean-IoU / AP vs GT) and an **inference false-positive counter** (instances whose mask lies on non-person GT pixels). *Why:* P2≈P1 means you have no number for the actual problem. *Verify:* numbers move when masks visibly change.
- **F0.3 Verify depth units.** Print raw-value histograms for a few `left_filled`/`left_filled_np` files; confirm metres, then **pin `depth.unit`** instead of "auto." *Why:* a unit error would invalidate everything. *Verify:* known objects land at plausible metres; σ1 sanity.
- **F0.4 Standardize resolution** across train/eval/infer (pick one, e.g. 504×896 to match your footage, or 518² — but one). *Verify:* eval at the training resolution; no aspect squish.

**Tier 1 — Make "instance" mean something (largest expected gain).**
- **F1.1 Reintroduce real categories OR add semantic negatives.** Either re-run the data engine with the object concepts your scenes contain (e.g. `person`, plus whatever else is foreground) and raise `num_classes` to match, **or** add an explicit background/negative-instance supervision so sofas/cars are learned as not-foreground. *Why:* this is the root cause of P3 false positives; no downstream tweak removes them while the model only knows "person vs nothing." *Verify:* inference false-positive count on sofas/cars drops sharply at the paper's 0.9 gate.
- **F1.2 Add Mask2Former deep supervision.** Emit per-decoder-layer (mask, class, depth) and apply the matched criterion at each layer (standard M2F aux loss). *Why:* this is what makes M2F converge and separate instances; its absence explains poor/merged masks even with perfect GT. *Verify:* mask IoU and instance count-vs-GT improve; fewer merged people.
- **F1.3 Restore inference gate to 0.9/0.8** in `infer_video.py` to match §4.2.2. *Why:* trivial, removes the bulk of visualized false masks immediately. *Verify:* overlays show far fewer non-person masks.
- **F1.4 Make masked attention use the same `mask_embed`** the head uses. *Why:* consistent mask conditioning is core to M2F. *Verify:* attention maps align with predicted masks; training stabilizes.

**Tier 2 — Improve the labels (GIGO).**
- **F2.1 Add a mask quality gate / light morphology** in `annotate` (drop specks, fill 1-px holes, optionally reject masks with implausible aspect/area), and consider **box-prompted per-object SAM3** to reduce person-merging. *Verify:* preview overlays; fewer merged/limbless GT masks.
- **F2.2 Handle invalid depth layers** rather than dropping (depth completion or robust median over valid pixels, with a confidence flag). *Verify:* `help.py` invalid-% drops; more instances enter L_d/refine.
- **F2.3 Switch GT-mask resize to bilinear+threshold** in matcher/criterion. *Verify:* thin-limb recall in the loss target improves.
- **F2.4 Tighten identity merge** (`reid_iou` up, add a same-frame split/duplicate check). *Verify:* fewer wrong re-links in logs; ID maps cleaner.

**Tier 3 — Make refinement help instead of hurt.**
- **F3.1 Replace the per-ROI 0..1 grid with image-normalized coordinates** of each ROI. *Why:* gives Φ_o the vertical-position cue the paper intends. *Verify:* P3 σ1 ≥ P1 σ1 on overlapping-pair frames.
- **F3.2 Normalize depth in matching cost and L_d** (÷max_depth). *Verify:* matching no longer flips on large depth gaps; earlier mask convergence.
- **F3.3 Gate the composite to matched/high-confidence instances only**, and consider clamping Eq. 9's correction range (test `D_bar` < `D_obj`). *Why:* stops false positives from shifting dense depth and curbs ±100% swings. *Verify:* P3 σ1/REL improve vs P1 (currently they regress).
- **F3.4 Use the 2-m partition** (rd=5) in the integrated config per Table 5. *Verify:* P1 REL/σ1 nudge toward the paper.

**Tier 4 — Hygiene.**
- **F4.1 Wire config loss weights** through `train.py` (use the existing `build_instance_criterion`/`build_refinement_criterion`). *Verify:* changing YAML weights changes logged losses.
- **F4.2 Set expectations on tracking:** the model is per-frame; if you need temporal stability, add a post-hoc tracker (e.g. IoU/DEVA on predicted masks) rather than expecting the head to be stable.

**Suggested validation loop after each tier:** re-run P2 with the new instance metrics on a fixed 200–300-frame slice; require mask-IoU↑ and false-positive-count↓ before touching P3; then require **P3 σ1 ≥ P1 σ1** (today it is lower) as the gate that refinement is finally net-positive.

---

### What is already correct (don't "fix" these)
- HDI Eq. 3 sign handling (`symmetric_range_error`) — your documented patch is the right call.
- No-object class index/convention across matcher, criterion, pair-selection, and inference — consistent.
- `build_refine_targets` correctly drops unmatched (false-positive) pairs at **training** time.
- ROIAlign `spatial_scale` mapping across feature/mask/depth maps — correct.
- Video-level 20% split, batch stratification, and the depth-decoder/patch-attention structure — faithful.

**Bottom line:** the depth backbone is fine; fix the *instance* definition (Tier 1) and the *labels* (Tier 2) before spending effort on refinement (Tier 3). The metrics and your own debug scripts already point at the same conclusion.
