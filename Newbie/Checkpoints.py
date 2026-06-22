"""Checkpoint save/load and partial weight-loading utilities.

    save_checkpoint   : model + optimizer + meta -> .pth
    load_checkpoint   : restore into model (+ optimizer), tolerant of missing keys
                        AND of shape-mismatched keys (robust partial load)
    load_pretrained   : load a sub-state (e.g. a Depth-Anything-V2 encoder, or a
                        phase-1 HDI checkpoint into the integrated model) with
                        prefix remapping and shape filtering.

Designed for the Sec. 4.3 multi-phase workflow where each phase resumes from the
previous phase's weights.

WHAT CHANGED vs. the previous version
-------------------------------------
`load_checkpoint` now filters the incoming state dict to keys that exist in the
model WITH A MATCHING SHAPE before calling `load_state_dict`. A shape mismatch
otherwise raises even under `strict=False`. This makes phase->phase transitions
that legitimately resize a sub-module load cleanly -- e.g. going phase 1 -> phase
2 after changing `num_queries` (100 -> 32): the instance-head query embeddings no
longer fit, so they keep their fresh initialization (correct, since phase 1 never
trains the instance head), while the backbone + HDI depth weights load normally.
Skipped keys are logged and returned, so the transition is transparent rather
than a crash.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

log = logging.getLogger("checkpoint")


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: int = 0,
    step: int = 0,
    phase: int = 1,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    ckpt: Dict[str, Any] = {
        "model": model.state_dict(),
        "epoch": epoch,
        "step": step,
        "phase": phase,
    }
    if optimizer is not None:
        ckpt["optimizer"] = optimizer.state_dict()
    if extra:
        ckpt["extra"] = extra
    torch.save(ckpt, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    strict: bool = False,
    map_location: str = "cpu",
) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    state = ckpt.get("model", ckpt)
    own = model.state_dict()

    # keep only keys present in the model with a matching shape
    filtered: Dict[str, torch.Tensor] = {}
    shape_skipped: List[str] = []
    for k, v in state.items():
        if k in own and own[k].shape == v.shape:
            filtered[k] = v
        elif k in own:                       # present but wrong shape -> keep fresh init
            shape_skipped.append(k)

    if strict and (shape_skipped or len(filtered) < len(own)):
        raise RuntimeError(
            f"strict load requested but cannot fully load; "
            f"shape-mismatched keys: {shape_skipped}")

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if shape_skipped:
        log.warning("load_checkpoint: %d key(s) skipped on shape mismatch "
                    "(kept fresh init): %s", len(shape_skipped),
                    ", ".join(shape_skipped[:6]) + (" ..." if len(shape_skipped) > 6 else ""))

    # optimizer state is only meaningful for a same-architecture resume; restore
    # defensively (a param-group change after an arch edit would otherwise crash)
    if optimizer is not None and "optimizer" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except (ValueError, KeyError) as e:
            log.warning("load_checkpoint: optimizer state not restored (%s)", e)

    return {
        "missing": list(missing),
        "unexpected": list(unexpected),
        "shape_skipped": shape_skipped,
        "epoch": ckpt.get("epoch", 0),
        "step": ckpt.get("step", 0),
        "phase": ckpt.get("phase", 1),
    }


@torch.no_grad()
def load_pretrained(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
    src_prefix: str = "",
    dst_prefix: str = "",
    verbose: bool = True,
) -> Dict[str, int]:
    """Load matching keys from `state_dict` into `model`, filtering by shape.

    src_prefix/dst_prefix allow remapping (e.g. load a standalone backbone
    state into model.backbone.*). Returns counts of loaded / skipped keys.
    """
    own = model.state_dict()
    loaded, skipped = 0, 0
    new_state = {}
    for k, v in state_dict.items():
        if src_prefix and not k.startswith(src_prefix):
            continue
        nk = dst_prefix + k[len(src_prefix):] if src_prefix else dst_prefix + k
        if nk in own and own[nk].shape == v.shape:
            new_state[nk] = v
            loaded += 1
        else:
            skipped += 1
    own.update(new_state)
    model.load_state_dict(own, strict=False)
    if verbose:
        print(f"[load_pretrained] loaded {loaded}, skipped {skipped}")
    return {"loaded": loaded, "skipped": skipped}


python train.py --model-config instancedepth/configs/instance_depth.yaml --data-root gid_custom \
    --phase 2 --init-from runs/phase1/ckpt_final.pth --out runs/phase2_v2
