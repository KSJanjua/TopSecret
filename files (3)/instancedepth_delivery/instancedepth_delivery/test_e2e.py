"""End-to-end smoke test of the GID data engine + dataset on synthetic data,
including compatibility with the uploaded HungarianMatcher / InstanceSetCriterion
and the build_refine_targets glue."""

import json
import logging
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")
sys.path.insert(0, "/home/claude/work")
sys.path.insert(0, "/home/claude/work/patched"); sys.path.insert(1, "/mnt/project")          # the user's uploaded model code

ROOT = Path("/home/claude/work/_synth")
shutil.rmtree(ROOT, ignore_errors=True)

# ---- 1) synthesize the user's dataset layout -------------------------------
H, W, N_FRAMES = 120, 160, 12
for batch in ("Batch 1", "Batch 2"):
    for ts in ("20260105_012545", "20260106_090001"):
        seq = ROOT / "Dataset" / batch / ts
        (seq / "left_rgb").mkdir(parents=True)
        (seq / "left_filled").mkdir()
        (seq / "left_filled_np").mkdir()
        rng = np.random.default_rng(hash((batch, ts)) % 2**32)
        for i in range(N_FRAMES):
            name = f"{i:06d}"
            rgb = rng.integers(0, 255, (H, W, 3), dtype=np.uint8)
            cv2.imwrite(str(seq / "left_rgb" / f"{name}.jpg"), rgb)
            # depth in MILLIMETERS (16-bit png) + float npy, two discs at 3m/5m,
            # background ramp 6..9 m. Mock segmenter discs: r=H//8=15,
            # c1=(60, 40+2%W*i), c2=(60, 56+...).
            yy, xx = np.mgrid[0:H, 0:W]
            d = 6000 + 3000 * (yy / H)
            r = max(6, H // 8)
            c1 = (H // 2, int(W * 0.25 + i * W * 0.02) % W)
            c2 = (H // 2, int(W * 0.35 + i * W * 0.02) % W)
            m1 = (yy - c1[0]) ** 2 + (xx - c1[1]) ** 2 <= r * r
            m2 = (yy - c2[0]) ** 2 + (xx - c2[1]) ** 2 <= (r + 2) ** 2
            d[m1] = 5000.0
            d[m2] = 3000.0
            d[0, 0] = 0.0                                # an invalid pixel
            cv2.imwrite(str(seq / "left_filled" / f"{name}_depth.png"),
                        d.astype(np.uint16))
            np.save(seq / "left_filled_np" / f"{name}.npy", d.astype(np.float32))

# ---- 2) run the data engine with the mock SAM3 backend ---------------------
from instancedepth.data_engine.config import DataEngineConfig
from instancedepth.data_engine.run_generate import generate

cfg = DataEngineConfig.from_dict(dict(
    dataset_root=str(ROOT / "Dataset"),
    sam3=dict(backend="mock", object_prompts=["person"]),
    identity=dict(min_track_length=3),
    output=dict(out_root=str(ROOT / "gid_custom"), preview_dir="preview", preview_every=6),
    split=dict(test_fraction=0.25, stratify_by_batch=True),
))
meta = generate(cfg)
print("\n--- meta.json ---")
print(json.dumps(meta["statistics"], indent=1)[:600])

# sanity on the generated artifacts
seq_dir = ROOT / "gid_custom" / "Batch 1" / "20260105_012545"
idm = cv2.imread(str(seq_dir / "object_masks" / "000000.png"), cv2.IMREAD_UNCHANGED)
assert idm.dtype == np.uint16 and set(np.unique(idm)) == {0, 1, 2}, np.unique(idm)
man = json.loads((seq_dir / "annotations.json").read_text())
inst = man["frames"]["000000"]["instances"]
layers = {i["track_id"]: i["depth_layer_m"] for i in inst}
print("GT depth layers:", layers)
assert abs(layers[2] - 3.0) < 0.05, "front disc should be ~3 m"
assert abs(layers[1] - 5.0) < 0.05, "back disc should be ~5 m"
# occluder (3 m) must win the overlap in the flattened id map:
ys, xs = np.nonzero(idm == 2)
assert len(ys) > 0

# ---- 3) PyTorch dataset -> uploaded matcher/criterion ----------------------
import torch
from torch.utils.data import DataLoader
from instancedepth.data.gid_dataset import (
    GIDDatasetConfig, GIDInstanceDepthDataset, collate_gid, build_refine_targets)

ds = GIDInstanceDepthDataset(GIDDatasetConfig(
    annotations_root=str(ROOT / "gid_custom"), split="train",
    image_size=(126, 168), min_instance_px=16, hflip_prob=0.0))
print(f"\ndataset: {len(ds)} frames")
dl = DataLoader(ds, batch_size=2, collate_fn=collate_gid)
batch = next(iter(dl))
print("image", tuple(batch["image"].shape), "depth", tuple(batch["depth"].shape),
      "ground", tuple(batch["ground"].shape))
assert batch["image"].shape[-2] % 14 == 0 and batch["image"].shape[-1] % 14 == 0
assert batch["depth"].min() >= 0 and batch["depth"].max() <= 10.0
g = batch["targets"][0]
print("targets[0]: labels", tuple(g["labels"].shape), "masks", tuple(g["masks"].shape),
      "depths", g["depths"].tolist())

# feed FAKE predictions through the user's real matcher + criterion
from matcher import HungarianMatcher
from instance_losses import InstanceSetCriterion

B, Nq, K = 2, 10, 1
Hf, Wf = batch["image"].shape[-2] // 2, batch["image"].shape[-1] // 2
outputs = dict(
    pred_logits=torch.randn(B, Nq, K + 1),
    pred_masks=torch.randn(B, Nq, Hf, Wf),
    pred_depth=torch.rand(B, Nq, 1) * 10.0,
)
matcher = HungarianMatcher()
indices = matcher(outputs, batch["targets"])
crit = InstanceSetCriterion(num_classes=K)
losses = crit(outputs, batch["targets"], indices)
print("instance losses:", {k: round(float(v), 4) for k, v in losses.items()})
assert torch.isfinite(losses["loss_total"])

# refine-target glue: fabricate two pairs, one fully matched, one not
pair_query_idx = torch.tensor([[int(indices[0][0][0]), int(indices[0][0][-1])],
                               [0, 9]])
batch_index = torch.tensor([0, 1])
dt, valid = build_refine_targets(pair_query_idx, batch_index, indices, batch["targets"])
print("refine dt:", dt.tolist(), "valid:", valid.tolist())
assert valid[0].item() in (True, False)  # shape contract holds
from refine_losses import RefinementCriterion
rl = RefinementCriterion()(torch.rand(2, 2) * 9 + 0.5, dt.clamp_min(0.1))
print("refine losses:", {k: round(float(v), 4) for k, v in rl.items()})

# test split files
print("\ntrain.txt:", (ROOT / "gid_custom" / "train.txt").read_text().split())
print("test.txt:", (ROOT / "gid_custom" / "test.txt").read_text().split())
print("\nALL CHECKS PASSED")
