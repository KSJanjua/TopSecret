"""Identify which checkpoint tensors don't match the eval config.

The Phase-2 eval printed "Skipping 5 keys with shape mismatches", meaning 5
parameters were left RANDOM during evaluation -> the reported AP/recall is
measured on a partially-random model and cannot be trusted yet.

This prints exactly which tensors mismatch and what config the checkpoint was
trained with, so you can re-evaluate with the matching config (no retraining).

Torch + yaml only; no model build, no backbone download, no GPU.

    python inspect_checkpoint_shapes.py runs/phase2/ckpt_final.pth \
        instancedepth/configs/instance_depth.yaml
"""

import sys

import torch
import yaml

ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "runs/phase2/ckpt_final.pth"
cfg_path = sys.argv[2] if len(sys.argv) > 2 else "instancedepth/configs/instance_depth.yaml"

ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
state = ck.get("model", ck)
cfg = yaml.safe_load(open(cfg_path))
ic = cfg.get("instance", {})

print(f"checkpoint : {ckpt_path}")
print(f"  phase={ck.get('phase')}  step={ck.get('step')}  tensors={len(state)}")


def shp(k):
    return tuple(state[k].shape) if k in state else "ABSENT"


print("\ninstance-head tensors stored in the checkpoint:")
for name in ["mask_query_feat.weight", "depth_query_feat.weight", "query_pos.weight",
             "class_head.weight", "class_head.bias"]:
    print(f"  instance_head.{name:24s} {shp('instance_head.' + name)}")

mq = state.get("instance_head.mask_query_feat.weight")
ch = state.get("instance_head.class_head.weight")
print("\ninferred config the CHECKPOINT was trained with:")
if mq is not None:
    print(f"  num_queries = {mq.shape[0]}")
if ch is not None:
    print(f"  num_classes = {ch.shape[0] - 1}")
print(f"\nconfig used for EVAL ({cfg_path}):")
print(f"  num_queries = {ic.get('num_queries')}")
print(f"  num_classes = {ic.get('num_classes')}")

# Surface every config-shaped tensor (instance head + HDI range heads) so a
# num_ranges mismatch is visible too.
print("\nall config-shaped tensors (name -> shape):")
for k in sorted(state):
    if any(s in k for s in ("query", "class_head", "range", "seg", "confidence")):
        print(f"  {k}: {tuple(state[k].shape)}")

mismatch = []
if mq is not None and ic.get("num_queries") is not None and mq.shape[0] != ic["num_queries"]:
    mismatch.append(f"num_queries: checkpoint={mq.shape[0]} vs yaml={ic['num_queries']}")
if ch is not None and ic.get("num_classes") is not None and (ch.shape[0] - 1) != ic["num_classes"]:
    mismatch.append(f"num_classes: checkpoint={ch.shape[0]-1} vs yaml={ic['num_classes']}")
print("\n=> VERDICT:",
      "instance head MISMATCH -> re-eval with the matching config (no retrain): " + "; ".join(mismatch)
      if mismatch else
      "instance-head num_queries/num_classes MATCH the yaml; the 5 skipped keys are "
      "elsewhere (likely HDI range heads). Run the eval again and confirm the instance "
      "head loaded fully -- if so, the near-zero AP is real (model/capacity).")
