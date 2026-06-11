# Code Audit — uploaded InstanceDepth implementation vs. the ICCV 2025 paper

Verdict per file is one of: ✅ faithful, ⚠️ deviation/ambiguity to be aware of,
🐛 bug (fix provided).

## 🐛 Bugs (fixes included in this delivery)

### 1. `matcher.py` — GT/pred mask resolution mismatch (crash)
`HungarianMatcher` flattens `pred_masks` at (Hf, Wf) and GT masks at whatever
resolution they arrive in, then matrix-multiplies them. With image-resolution
GT masks (the natural dataset output, and what `InstanceSetCriterion` already
handles by resizing), matching crashes:

    RuntimeError: mat1 and mat2 shapes cannot be multiplied

**Fix** (applied in `patched/matcher.py`): nearest-resize GT masks to the
prediction's mask resolution before computing BCE/dice costs — the dense
analog of Mask2Former's common-point-set sampling. Verified against the
generated dataset end-to-end.

## ⚠️ Faithfulness deviations / ambiguities

### 2. `holistic_depth.py` — Eq. 3 correction is one-sided (cannot increase depth)
The code implements Eq. 2 as `R = Σ_rd (sigmoid(C) · softmax(S))`, so
R ∈ [0, 1], and Eq. 3 `E = 2·(R − 1)·(MAX_d/rd)` therefore lies in
**[−2·MAX_d/rd, 0]**: every refinement step can only *decrease* depth. The
paper calls E a "relative depth error" and the clear intent (mirrored by the
symmetric Eq. 9, `(E·2 − 1) ∈ [−1, 1]`) is a *signed* correction. The printed
Eq. 3 is only symmetric if R ∈ [0, 2], which the printed Eq. 2 cannot produce
under any standard normalization of S and C.

This is an ambiguity in the paper, not strictly a bug in the code — but the
literal implementation biases the iterative refinement downward and partially
defeats Eqs. 1–4. **Recommended, config-gated change** in
`HolisticDepthInitialization._refine` / `InstanceDepth._hdi_from_feats`:

```python
# cfg flag: hdi.symmetric_range_error: bool = True
if self.cfg.symmetric_range_error:
    e = (2.0 * r - 1.0) * self.range_step      # E ∈ [−step, +step], Eq.9-style
else:
    e = 2.0 * (r - 1.0) * self.range_step      # literal Eq. 3 (one-sided)
```

Keep the literal form available for ablation; default to symmetric. Label:
[Reasonable Assumption] forced by an internal inconsistency in the paper.

### 3. `holistic_depth.py` — refinement reuses ONE fused feature, paper suggests per-level features
The paper: "Let F_i represent the depth range features at the **i-th level**…
The depth is then refined iteratively at multiple **segmentation levels**…
repeated across all depth levels." The most faithful reading runs Eqs. 1–4
once per decoder scale (1/8 → 1/4 → 1/2) with that scale's fused feature
F_i, recomputing S_i each level. The uploaded code instead computes one fused
map and iterates Eqs. 1–4 on it `num_refine_steps=3` times. Both are
defensible (the paper never defines "level" formally); the per-level variant
matches the text more closely and gives the confidence net genuinely
multi-scale evidence. Suggested refactor: expose the intermediate fused maps
`g8, g4, g2` from `DepthRangeFeatureDecoder` and run one `_refine` per scale
(upsampling the running depth between scales). Mark [Strongly Inferred].

### 4. `instance_head.py` — no deep supervision (aux losses)
The docstring promises `aux_outputs` per decoder layer; none are produced.
Mask2Former's training recipe applies the matching loss after every decoder
layer, which matters for convergence with 9 layers. Not paper-contradicting
(the paper is silent), but worth restoring if phase-2 convergence is slow.

### 5. `matcher.py` — depth cost uses L1, paper says smoothed-L1
For matching costs the two are near-identical (and DETR-style matchers
routinely use L1), but for literal fidelity swap `torch.cdist(p=1)` for an
elementwise `F.smooth_l1_loss(..., reduction="none", beta=1.0)` matrix.
Cosmetic; left unchanged.

### 6. `dinov2_dpt.py` — bilinear resample instead of DPT's learned (de)convolutions
DPT's reassemble uses strided/transposed convs per scale; the code uses
`F.interpolate` + 3×3 conv. Functionally close, fewer params; flagged
[Reasonable Assumption] in the file, which is accurate. Also note DPT-L
conventionally hooks 4 layers (4, 11, 17, 23); the code hooks 3 because
Fig. 5 shows exactly three scales — defensible.

## ✅ Verified faithful (against the extracted paper text + Figs. 4, 5, 7)

| File | Checks |
|---|---|
| `patch_attention.py` | Fig. 5 order patch-conv→FC→attention; sizes 4/8/16 on 1/8,1/4,1/2 (all scales tokenize to H/32 — internally consistent) |
| `range_decoder.py` | coarse→fine additive fusion = Fig. 5 ⊕ symbols |
| `heads.py` | S softmax over rd, D sigmoid·MAX_d, Φd = Eq. 1 |
| `query_fusion.py`, `transformer_decoder.py`, `pixel_decoder.py` | Mask2Former mechanics + Fig. 7 dual query streams + fusion; FPN-vs-MSDeformAttn assumption correctly labeled |
| `instance_losses.py` | L_m BCE+dice, L_c CE w/ no-object 0.1, L_d smooth-L1 (Eqs. 5–7) |
| `pair_selection.py` | cls>0.9, mask>0.8, IoU>0.1, nearest-depth guest — exactly Sec. 4.2.2 |
| `roi_extract.py` | F_obj ∈ R^{2×C×Hp×Wp}; G_obj = mask logits + normalized coords + global depth, via ROIAlign — matches text |
| `relation_reason.py` | Eqs. 8–9 verbatim |
| `refine_losses.py` | Eqs. 10–12 verbatim (per-pair Ldist averaged — sensible) |
| `instance_depth.py` | phase schedule = Sec. 4.3 (train range → freeze encoder/train instance → freeze instance/finetune encoder+decoder); shared backbone is the right engineering reading |
| configs | rd = 10 m / 2 m = 5 matches Table 5 best (2-meter partitioning); MAX_d = 10 m matches Fig. 2 |

## Missing pieces (now partially delivered)

- **Dataset/annotation pipeline** — delivered (`data_engine/`, `data/`).
- **Training loop for the 3-phase schedule (55k/25k/25k iters, LR 1e-5/1e-5/1e-6)** — next step.
- **Eval metrics (RMS, REL, RMSlog, Log10, σ1–3)** — next step.
- **Depth-Anything-V2 encoder initialization** (paper inits from DA-V2 for the
  NYUDv2 experiment; `checkpoint.load_pretrained` already supports the remap).
