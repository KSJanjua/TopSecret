"""Evaluation on the GID-style test split (paper Sec. 5.1 / Table 2 metrics).

Metrics (computed on valid GT pixels, i.e. gt in (min_d, max_depth]):
    RMS     sqrt(mean((d - d*)^2))
    REL     mean(|d - d*| / d*)
    RMSlog  sqrt(mean((log d - log d*)^2))
    Log10   mean(|log10 d - log10 d*|)
    sigma_i fraction with max(d/d*, d*/d) < 1.25^i, i = 1,2,3

WHAT CHANGED vs. the previous version
-------------------------------------
1. BUG FIX: the old code computed a person-region mask when `--mask objects`
   was passed but then called `depth_metrics(pred, gt, ...)` WITHOUT passing
   `region=`, so the flag was silently ignored and every run measured the
   full frame. The region is now passed through.
2. This script now ALWAYS reports two metric blocks in a single run:
       "all"      -> dense, every valid GT pixel (paper Table 2 setting)
       "objects"  -> only pixels inside the GT instance (person) masks
   The "objects" block is the one that actually moves between phases, because
   the instance head (phase 2) and occlusion refinement (phase 3) only touch
   instance regions. Watching only the dense block is why the three phases
   looked identical. The `--mask` flag is removed in favour of this dual report.

[Paper Specified] the metric set; per-frame evaluation, averaged over frames.
[Reasonable Assumption] predictions are taken from `refined_depth` (the final
output; equals init_depth when no occlusion pairs fire); frame-mean averaging;
the "objects" region is the union of GT instance masks for the frame.

Usage
-----
    python evaluate.py --model-config instancedepth/configs/instance_depth.yaml \
        --data-root gid_custom --checkpoint runs/phase3/ckpt_final.pth \
        [--split test] [--use-init-depth]
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from instancedepth.build import build_instance_depth_from_yaml
from instancedepth.data.gid_dataset import (
    GIDDatasetConfig, GIDInstanceDepthDataset, collate_gid)
from instancedepth.utils.checkpoint import load_checkpoint

log = logging.getLogger("eval")


@torch.no_grad()
def depth_metrics(pred: torch.Tensor, gt: torch.Tensor,
                  min_d: float = 0.01, max_d: float = 10.0,
                  region: Optional[torch.Tensor] = None) -> Optional[dict]:
    """pred, gt: (1, H, W) metric depth for ONE frame.

    region: optional (1, H, W) or (H, W) bool mask; when given, metrics are
    restricted to pixels inside it (e.g. the union of GT person masks).
    """
    valid = (gt > min_d) & (gt <= max_d)
    if region is not None:
        valid = valid & region.reshape(valid.shape).bool()
    if valid.sum() == 0:
        return None
    p = pred[valid].clamp(min=min_d)
    g = gt[valid]
    thresh = torch.maximum(p / g, g / p)
    return dict(
        RMS=float(torch.sqrt(((p - g) ** 2).mean())),
        REL=float(((p - g).abs() / g).mean()),
        RMSlog=float(torch.sqrt(((p.log() - g.log()) ** 2).mean())),
        Log10=float((p.log10() - g.log10()).abs().mean()),
        sigma1=float((thresh < 1.25).float().mean()),
        sigma2=float((thresh < 1.25 ** 2).float().mean()),
        sigma3=float((thresh < 1.25 ** 3).float().mean()),
    )


def person_region(target: dict, gt_hw, device) -> Optional[torch.Tensor]:
    """Union of the frame's GT instance masks, resized to GT depth resolution.

    Returns a (1, H, W) bool mask, or None when the frame has no instances.
    """
    masks = target.get("masks")
    if masks is None or masks.numel() == 0:
        return None
    region = masks.to(device).any(0, keepdim=True).float()          # (1, Hm, Wm)
    if region.shape[-2:] != gt_hw:
        region = F.interpolate(region.unsqueeze(0), size=gt_hw,
                               mode="nearest").squeeze(0)
    return region > 0.5


def main() -> None:
    ap = argparse.ArgumentParser(description="InstanceDepth evaluation")
    ap.add_argument("--model-config", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--image-size", type=int, nargs=2, default=(518, 518))
    ap.add_argument("--use-init-depth", action="store_true",
                    help="evaluate Stage-1 holistic output instead of refined_depth")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    model = build_instance_depth_from_yaml(args.model_config).to(args.device)
    load_checkpoint(args.checkpoint, model)
    model.eval()

    import yaml
    with open(args.model_config) as f:
        max_depth = float(yaml.safe_load(f).get("max_depth", 10.0))

    ds = GIDInstanceDepthDataset(GIDDatasetConfig(
        annotations_root=args.data_root, split=args.split,
        image_size=tuple(args.image_size), max_depth=max_depth, hflip_prob=0.0))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate_gid)
    log.info("evaluating %d frames (%s split)", len(ds), args.split)

    sums_all: dict = defaultdict(float)
    sums_obj: dict = defaultdict(float)
    n_all = n_obj = 0

    run_inst = not args.use_init_depth
    run_ref = not args.use_init_depth

    with torch.inference_mode():
        for batch in loader:
            rgb = batch["image"].to(args.device)
            gt = batch["depth"].to(args.device)
            out = model(rgb, run_instance=run_inst, run_refine=run_ref)
            pred = out["init_depth"] if args.use_init_depth else out["refined_depth"]

            for b in range(rgb.shape[0]):
                # dense metric, every valid pixel (paper Table 2 setting)
                m_all = depth_metrics(pred[b], gt[b], max_d=max_depth)
                if m_all is not None:
                    for k, v in m_all.items():
                        sums_all[k] += v
                    n_all += 1

                # instance-region metric (the one phases 2/3 actually affect)
                region = person_region(batch["targets"][b], gt[b].shape[-2:], args.device)
                if region is not None:
                    m_obj = depth_metrics(pred[b], gt[b], max_d=max_depth, region=region)
                    if m_obj is not None:
                        for k, v in m_obj.items():
                            sums_obj[k] += v
                        n_obj += 1

    result = {
        "all": {k: round(v / max(n_all, 1), 4) for k, v in sums_all.items()},
        "objects": {k: round(v / max(n_obj, 1), 4) for k, v in sums_obj.items()},
        "num_frames_all": n_all,
        "num_frames_objects": n_obj,
        "depth_source": "init_depth" if args.use_init_depth else "refined_depth",
    }
    log.info("results: %s", json.dumps(result))
    print(json.dumps(result, indent=1))
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
