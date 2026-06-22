# How Your InstanceDepth Code Works — Phase 2 & Phase 3 (with the math)

This walks through the data as it flows through your **current** code (post-fixes: deep supervision, unified `_predictor`, `num_queries=32`, phase-3 = `L_ref` only). Shapes use your real training size **H×W = 504×896**, DINOv2 ViT-L/14, `hidden_dim C = 256`, `num_queries N = 32`, `num_classes K = 1` (so `K+1 = 2`: `[person, no-object]`), `max_depth = 10`.

Notation: `B` = batch, `(…)` = tensor shape. "logits" = pre-sigmoid/softmax scores. ⊙ = elementwise multiply, ⟨·,·⟩ = dot product.

---

## 0. Shared setup (what Phase 2 receives)

The backbone runs once (`instance_depth.py: feats = self.backbone(rgb)`):

- Input `rgb (B,3,504,896)` → DINOv2 patches on a `36×64` grid → DPT "reassemble" produces three maps, coarse→fine:
  - `f8 (B,256,63,112)`  (stride 8)
  - `f4 (B,256,126,224)` (stride 4)
  - `f2 (B,256,252,448)` (stride 2)

Phase 1 (HDI) consumes these to produce `init_depth (B,1,504,896)` — a coarse metric depth map. **Phase 2 reuses the *same* `[f8,f4,f2]`.** In phase 2 the backbone + HDI are frozen, so these features are fixed inputs.

> HDI in one line (context only — Eqs. 1–4): a range decoder makes features `F`; heads give a depth-range distribution `S` (softmax over `rd` bins) and a first depth `D`; a confidence net `C=σ(Φ_d(F,D))`; then iterate `R=Σ(C⊙S)`, `E=(2R−1)·step`, `D←D+E` (your symmetric form). Output = `init_depth`.

---

# PHASE 2 — Instance Depth Layer Prediction (§4.2.1)

**Goal:** predict, for up to `N=32` instance slots ("queries"), three things — a **mask** `Msk`, a **class** `Cls`, and a single **depth layer** `Dep` (the instance's *average* depth, a scalar). This is a Mask2Former segmentation head with an extra depth-layer output.

## Step 1 — Pixel decoder: turn 3 features into mask features + memory
`instance_head.py → pixel_decoder.py`

An FPN fuses `[f8,f4,f2]` top-down (upsample coarser, add finer, 3×3 conv) and produces:

- `mask_features (B,256,252,448)` — the **finest** map; this is the per-pixel embedding masks are read from. Call its size `Hf×Wf = 252×448`.
- `ms_feats` — a list of maps fed to the decoder's cross-attention (your config uses all three: `[63×112, 126×224, 252×448]`).

Each `ms_feat` gets a fixed 2-D sinusoidal positional embedding `ms_pos` (so attention knows *where* each pixel is). Sinusoidal PE for a position and channel `k`: alternating `sin(pos·ω_k)`, `cos(pos·ω_k)` with `ω_k = 1/10000^(k/quarter)` — standard transformer position encoding, no learning.

## Step 2 — The queries
`instance_head.py`

Three learned embedding tables of size `(N,C)=(32,256)`:
- `mask_query_feat` — drives masks + class.
- `depth_query_feat` — drives the depth layer.
- `query_pos` — a positional embedding shared by both streams.

Tile to batch and **concatenate the two streams** along the query axis:
```
mask_q  (32,B,256)
depth_q (32,B,256)
cat_q = [mask_q ; depth_q]  -> (64,B,256)      # 2N queries go through the decoder together
cat_pos = [query_pos ; query_pos] -> (64,B,256)
```
Intuition: each of the 32 "slots" will try to claim one object. Running both streams together lets self-attention share information between a slot's mask-half and depth-half.

## Step 3 — `_predictor`: query → (mask logits, class logits)
`instance_head._predictor`

Given any query state `q (2N,B,C)` and `mask_features (B,C,Hf,Wf)`:
```
q   = LayerNorm(q).permute -> (B, 2N, C)
class_logits = Linear_{C->K+1}(q)              -> (B, 2N, 2)
mask_emb     = MLP_{C->C->C->C}(q)             -> (B, 2N, C)
mask_logits  = einsum("bnc,bchw->bnhw", mask_emb, mask_features)  -> (B, 2N, Hf, Wf)
```
The mask is literally a **dot product** between each query's embedding and every pixel embedding: `mask_logits[b,n,h,w] = ⟨ mask_emb[b,n], mask_features[b,:,h,w] ⟩`. A high dot product ⇒ that pixel belongs to query `n`.

**Why this function matters (your fix #2):** the *same* `_predictor` is used both to build the per-layer attention mask *and* to produce the final output, so "where the query looks" and "what the query predicts" are consistent.

## Step 4 — The decoder loop with masked attention + deep supervision
`transformer_decoder.py`

Before any layer, predict from the initial queries → **prediction 0**. Then for each of the 9 layers (cycling `idx = i mod 3` over the three `ms_feats`, so scales repeat 1/8→1/4→1/2 three times):

1. **Build the attention mask from the current mask prediction** (`_attn_mask_from_logits`):
   ```
   ml   = interpolate(mask_logits) to this feat's (h,w)
   attn = (sigmoid(ml) < 0.5)      # True = BLOCK this pixel
   ```
   So a query may only attend to pixels where its predicted mask probability ≥ 0.5 — it focuses on *its own object*. Safety: if a query would block **every** pixel, that row is reset to "attend everywhere" (otherwise `softmax(all −∞)=NaN`). The mask is `.detach()`ed (it steers attention but isn't itself differentiated here).

2. **One decoder layer** (`_DecoderLayer`):
   - Masked **cross-attention**: queries (`+query_pos`) attend to memory pixels (`+memory_pos`), restricted by `attn_mask`; residual + LayerNorm.
   - **Self-attention** among the 64 queries; residual + LayerNorm. (This is how slots "negotiate" so two slots don't grab the same object.)
   - **FFN** (Linear→ReLU→Linear); residual + LayerNorm.

3. **Predict again** from the updated queries → prediction `i+1`.

Result: `predictions` = a list of **10** `(mask_logits, class_logits)` (initial + 9 layers), plus the final query state `final_q (64,B,256)`.

**Deep supervision (your fix #1):** every one of those 10 predictions is supervised by the loss (Step 6's aux loop). Without it, only the last layer gets gradient and masks stay blurry/jittery — this was the main Phase-2 quality problem.

## Step 5 — Depth layer via query fusion
`instance_head.py → query_fusion.py`

Split the final queries back into the two streams (first 32 = mask, last 32 = depth) and fuse:
```
cat = [depth_q ; mask_q]                  # (32,B,512)
delta = MLP(cat); gate = σ(Linear(cat))   # gate in (0,1)
fused = LayerNorm(depth_q + gate ⊙ delta) # (32,B,256)
```
This is a **gated update**: the depth stream stays primary but is conditioned on the mask stream, tying each instance's depth to *its own segmented region*. Then:
```
depth_layer = sigmoid(MLP_{C->C->1}(fused)) * max_depth   -> (B,32,1), in [0,10]
```
A single scalar per slot — the instance's average depth (paper's `Dep_i`). Note depth is computed **once, at the final layer** (it's a per-instance scalar; deep supervision is for mask+class only).

## Step 6 — Outputs
```
pred_masks   (B,32,252,448)   # final layer, mask stream (first N of the 64)
pred_logits  (B,32,2)
pred_depth   (B,32,1)
mask_features(B,256,252,448)  # reused by Phase 3
aux_outputs  list of 9 × {pred_masks (B,32,252,448), pred_logits (B,32,2)}  # for deep supervision
```

---

## Step 7 — Hungarian matching (Eqs. 5–7)
`matcher.py`

The model emits 32 predictions in no particular order; GT has `G` people (typically 2–3). We need a **one-to-one** assignment so each prediction is trained toward at most one GT. We build a cost matrix `cost (32, G)` where `cost[n,g]` = "how bad is it to call prediction `n` the GT person `g`," then pick the assignment minimizing total cost.

Per image, three cost terms (computed at mask resolution; GT masks `nearest`-resized from 504×896 down to 252×448):

**(a) Class cost** — push matched predictions toward high probability of the GT label:
```
prob = softmax(pred_logits)            # (32, 2)
cost_class[n,g] = − prob[n, label_g]   # lower (more negative) = better
```

**(b) Mask BCE cost** (`_bce_cost`), evaluated over all `P = 252·448` pixels. The trick computes, for every (prediction, GT) pair at once:
```
pos = BCE(pred_logit, target=1)   neg = BCE(pred_logit, target=0)   # per pixel
cost_mask[n,g] = ( Σ_pixels pos[n]·tgt[g] + neg[n]·(1−tgt[g]) ) / P
```
i.e. pay `pos` where the GT mask is 1 and `neg` where it's 0. Low when the predicted mask matches GT `g`.

**(c) Dice cost** (`_dice_cost`) — overlap-based, scale-robust:
```
p = sigmoid(pred); dice = (2·⟨p,t⟩ + 1) / (Σp + Σt + 1);  cost_dice = 1 − dice
```
Dice ≈ 1 (cost ≈ 0) when the predicted and GT masks overlap well.

**(d) Depth cost** — only when `pred_depth` is present (the **main** output, not aux layers; your matcher fix):
```
cost_depth[n,g] = | pred_depth[n] − depth_g |     # L1 between depth layers
```

Total and solve:
```
cost = w_mask·cost_mask + w_dice·cost_dice + w_class·cost_class (+ w_depth·cost_depth)
       (defaults 5,        5,                2,                    1)
row, col = linear_sum_assignment(cost)   # Hungarian algorithm
```
`linear_sum_assignment` returns the matching `(prediction_idx, gt_idx)` with the **minimum total cost** over all one-to-one pairings (it's the classic assignment problem; SciPy solves it in ≈O((N+G)³)). For aux layers `pred_depth` is absent, so the depth term is dropped — exactly what your matcher now handles.

---

## Step 8 — The set loss (with deep supervision)
`instance_losses.py`

For **one** prediction layer (`_layer_loss`), given the matching `indices`:

**Classification — over ALL 32 queries.** Build per-query targets: matched queries get their GT label, every unmatched query gets the **no-object** class (index `K = 1`):
```
target_classes[n] = label_{g(n)}  if n matched else no_object
loss_class = CrossEntropy(pred_logits, target_classes, weight=[1.0, 0.1])
```
The class weight `[1.0, 0.1]` down-weights "no-object" because ~29 of 32 queries are unmatched each step; without it the model would just predict "no-object" for everything. (Cross-entropy here = `−Σ w_c · y_c · log softmax(logits)_c`.)

**Mask — on matched pairs only.** Concatenate the matched predicted masks `pm` and GT masks `tm` (nearest-resized to 252×448):
```
loss_mask = BCEWithLogits(pm, tm)                  # per-pixel classification
loss_dice = mean( 1 − (2⟨σ(pm),tm⟩+1)/(Σσ(pm)+Σtm+1) )   # overlap
```

**Depth layer — matched pairs, FINAL layer only** (`with_depth=True`):
```
valid = isfinite(td) & (td>0) & isfinite(pd)       # skip the ~19% invalid GT layers
loss_depth = SmoothL1(pd[valid], td[valid], beta=1.0)
```
Smooth-L1 (Huber) is `0.5·e²` for `|e|<β` and `β·(|e|−0.5β)` otherwise — quadratic near zero (precise) but linear for big errors (robust to a far-off depth). The `valid` guard is important: a single invalid (0) GT depth makes the smooth-L1 *backward* NaN, so those are excluded and the graph is kept alive with `pd.sum()*0` if none are valid.

**Putting layers together** (`forward`):
```
total = w_class·L_class + w_mask·L_mask + w_dice·L_dice + w_depth·L_depth      # MAIN output
for each aux layer:                                                            # 9 of them
    aux_idx = matcher(aux, targets)          # match this layer independently
    total += w_class·L_class + w_mask·L_mask + w_dice·L_dice   # NO depth term on aux
```
So the gradient reaches **every** decoder layer. Logged numbers (`loss_class/mask/dice/depth`) are the main-output components; `loss_total` is the full sum that's back-propagated.

**Phase-2 training** (`train.py`, `phase==2`): backbone+HDI frozen; `out = model(rgb, run_instance=True, run_refine=False)`; `indices = matcher(out, targets)`; `loss = inst_crit(out, targets, indices)["loss_total"]`; AdamW step. Only the instance head learns.

---

# PHASE 3 — Occlusion-Aware Depth Refinement (§4.2.2)

**Goal:** where instances overlap, their depth layers can be wrong (occlusion ambiguity). Phase 3 looks at **pairs** of overlapping instances and predicts a correction to each one's depth layer so the pair is geometrically consistent, then writes the corrected layers back into the depth map. The instance head is **frozen** here; only Φ_o (+ the depth encoder/decoder) learn.

Input: the Phase-2 dict (`pred_logits/masks/depth/mask_features`) + `init_depth`.

## Step 1 — Select occlusion pairs
`pair_selection.select_occlusion_pairs`, per image.

**Filter to confident instances.**
```
cls_conf = max over foreground classes of softmax(pred_logits)   # = softmax[:,0] here (person prob)
mask probs mp = sigmoid(pred_masks); binm = mp ≥ 0.5
mask_conf = mean of mp inside binm                               # avg confidence within the mask
keep = (cls_conf > 0.9) & (mask_conf > 0.8)                      # paper's thresholds
```
Only instances the model is sure about survive (this is the gate that should reject background — and why a calibrated class head matters).

**Pair survivors by overlap, choose the nearest-depth guest.** Among the `M` survivors compute the IoU matrix on their binary masks:
```
inter = bm @ bmᵀ ; union = areaᵢ + areaⱼ − inter ; IoU = inter/union  (diagonal zeroed)
depth_dist[i,j] = | dep_i − dep_j |
overlap = IoU > 0.1
for each main a with ≥1 overlap:  guest = argmin over overlaps of depth_dist[a, ·]
```
Each **main** instance gets one **guest** = the overlapping instance closest to it in depth. Output `pairs (P,2)` of `(main_idx, guest_idx)`. If <2 survivors or no overlaps → no pairs (refinement is a no-op; on low-occlusion data this is most frames — expected).

## Step 2 — Boxes from masks
`pair_selection.masks_to_boxes`

For every query, the tight bounding box of its binarized mask, in **feature coordinates** (252×448): `box = [x_min, y_min, x_max+1, y_max+1]` (empty mask → a 1×1 box so ROIAlign stays defined).

## Step 3 — ROIAlign: crop features + geometric priors
`roi_extract.ROIPairExtractor`

For each pair we extract, for **both** instances (main & guest), fixed `7×7` crops. The pair layout `(P,2,…)` is flattened to `2P` ROIs tagged with their batch index.

**F_obj — depth features** (`depth_feats = mask_features`, `(B,256,252,448)`):
```
scale_feat = 448 / 448 = 1.0
F_obj = roi_align(depth_feats, rois, output=7, spatial_scale=1.0, aligned=True)  -> (P,2,256,7,7)
```
`spatial_scale` rescales the boxes (given in `Wf=448` coords) onto the target map. Since `mask_features` is also 448 wide, scale = 1.

**G_obj — geometric priors** `(P,2,4,7,7)`, channels = mask(1) + coords(2) + depth(1):
- **mask logits** of each instance, ROIAligned (scale 1.0): the shape prior.
- **coordinate grid** (2 channels, x and y in `[0,1]` over the 7×7 patch).
- **global depth** = `init_depth (B,1,504,896)` ROIAligned with `scale_depth = 896/448 = 2.0`: the HDI depth under each box.

```
G_obj = concat([mask_pooled(1), coord(2), depth_pooled(1)])  -> (P,2,4,7,7)
```

## Step 4 — Φ_o relation reasoning (Eqs. 8–9)
`relation_reason.RelationReasoning`

Flatten and concatenate features for each instance in the pair, run a 3-layer MLP that outputs **one scalar per instance**:
```
x = concat([F_obj, G_obj], channels) -> (P, 2, (256+4)·7·7 = 12740)
E_obj = sigmoid( MLP_{12740->256->256->1}(x) )      -> (P,2), in (0,1)     # Eq. 8
```
Then the refinement update, with `D_obj` = the instance's current depth layer and `D_bar = D_obj`:
```
D_hat = (2·E_obj − 1) · D_bar + D_obj      -> (P,2)                         # Eq. 9
      = (2·E_obj) · D_obj                  (since D_bar = D_obj)
```
Read it as: `E=0.5` ⇒ no change; `E→1` ⇒ depth doubled; `E→0` ⇒ depth → 0. `(2E−1) ∈ [−1,1]` is a **signed relative correction**, scaled by `D_bar`. (As the audit noted, with `D_bar=D_obj` the correction range is ±100% — faithful to the literal equation, and a knob worth watching.) `D_hat` is clamped to `[0,10]`.

## Step 5 — Composite back into the depth map
`occlusion_refine._composite` (no-grad; bookkeeping only)

For each refined instance, add the **constant layer shift** `Δ = D_hat − D_obj` to every pixel inside that instance's (upsampled) binary mask of `init_depth`:
```
refined_depth = init_depth.clone()
for each refined instance: refined_depth[ mask region ] += (D_hat − D_obj)
```
So `refined_depth` is `init_depth` with whole instance regions shifted by their corrected layer. The trainable signal does **not** flow through this composite — it flows through `D_hat` in the loss (next step). `refined_depth` is what `evaluate.py` scores.

## Step 6 — Build the supervision targets
`gid_dataset.build_refine_targets` (no-grad)

The pairs are between **predicted** query indices, but the loss needs **GT** depth layers. Use the Phase-2 Hungarian `indices` as a lookup `predicted_query → GT depth`:
```
for each pair (q_main, q_guest):
    if BOTH queries were matched to a GT person:
        dt[p] = [GT_depth(q_main), GT_depth(q_guest)] ; valid[p]=True
    else: valid[p]=False        # a false-positive instance in the pair -> dropped
```
This is the key training-time guard: pairs containing an unmatched (false-positive) instance are excluded, so refinement is trained only on real overlaps.

## Step 7 — Refinement losses (Eqs. 10–12)
`refine_losses.RefinementCriterion`, evaluated on `d_hat[valid]` vs `dt[valid]`.

**L_obj — scale-invariant log loss (SigLog, Eq. 10).** With `g = log(D_hat) − log(DT)` over all valid instances and `vf = 0.85`:
```
L_obj = sqrt( mean(g²) − vf · (mean g)² )
```
The `mean(g²)` term penalizes per-instance log error; subtracting `vf·(mean g)²` **forgives a shared global scale offset** (if every instance is off by the same factor, `mean g ≠ 0` cancels part of it). This focuses the loss on getting *relative* depths right — exactly the occlusion concern.

**L_dist — pairwise relative-depth consistency (Eq. 11).** Per pair (`i=main, j=guest`):
```
L_dist = mean | (D_hat_i − D_hat_j)² − (DT_i − DT_j)² |
```
It forces the **squared depth gap** between main and guest to match the GT gap. Squaring makes it scale-sensitive and emphasizes pairs with large true separation (occluder clearly in front of occludee).

**Total (Eq. 12):**
```
L_ref = λ_obj · L_obj + λ_dist · L_dist        (defaults 1.0, 0.5)
```

**Phase-3 training** (`train.py`, `phase==3`): the **instance head is frozen**; encoder+decoder+Φ_o are trainable. Each step:
```
out = model(rgb, run_instance=True, run_refine=True)
indices = matcher(out, targets)                         # to build refine targets
dt, valid = build_refine_targets(meta.pair_query_idx, meta.batch_index, indices, targets)
loss = ref_crit(out["d_hat"][valid], dt[valid])["loss_ref"]   # L_ref ONLY (your fix)
```
The dense SigLog term is **off** (`--dense-weight-phase3 0`): on low-occlusion data it would dominate and push the shared backbone, dragging the frozen masks onto background — the regression you removed. Gradient path: `L_ref → D_hat → E_obj → Φ_o`, and through `G_obj`'s `init_depth` crop into the (fine-tuned) HDI. If a frame has no pairs, `loss = 0·init_depth.sum()` (a valid zero that keeps the optimizer happy).

---

## How the two phases connect to the metrics
`evaluate.py` now reports two blocks:
- **`all`** — every valid GT pixel (paper Table 2 setting). Barely moves between phases because Phase 2 doesn't change depth and Phase 3 only edits instance regions on the few overlapping frames.
- **`objects`** — only pixels inside GT person masks. This is the block that should improve from Phase 1 → 3, because that's the only region the instance head and refinement touch. (Watching only `all` is why your three phases previously looked identical.)

`analyze_depth_range.py` is your data-driven check for the *next* decision: your ZED depth runs to ~45 m with people at 0–4 m, and the 10 m clamp is dropping ~19% of instances to depth-0. It recomputes true per-instance layers from raw depth and shows, per candidate `max_depth`, how many instances you'd recover — i.e. how high `max_depth` (and therefore `rd = max_depth/partition`) must go for your scene.

---

### Quick reference — the equations
- **Matching (5–7):** `min Σ [ w_m·L_mask + w_c·L_cls + w_d·L_depth ]`, solved by Hungarian.
- **Refinement error (8):** `E_obj = σ(Φ_o([F_obj, G_obj]))`.
- **Refinement update (9):** `D_hat = (2E_obj − 1)·D_bar + D_obj` (here `D_bar = D_obj`).
- **Object loss (10):** `L_obj = SigLog(D_hat, DT) = sqrt(mean g² − 0.85·(mean g)²)`, `g = log D_hat − log DT`.
- **Distance loss (11):** `L_dist = mean |(D_hat_i − D_hat_j)² − (DT_i − DT_j)²|`.
- **Total (12):** `L_ref = 1.0·L_obj + 0.5·L_dist`.
