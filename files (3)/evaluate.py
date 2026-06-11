"""Evaluation on the GID-style test split (paper Sec. 5.1 / Table 2 metrics).

Metrics (computed on valid GT pixels, i.e. gt > 0, within [min_d, max_depth]):
    RMS     sqrt(mean((d - d*)^2))
    REL     mean(|d - d*| / d*)
    RMSlog  sqrt(mean((log d - log d*)^2))
    Log10   mean(|log10 d - log10 d*|)
    sigma_i fraction with max(d/d*, d*/d) < 1.25^i, i = 1,2,3

[Paper Specified] the metric set; evaluation per frame, averaged over frames.
[Reasonable Assumption] predictions are taken from `refined_depth` (the final
output; equals init_depth when no occlusion pairs fire); frame-mean averaging.

Usage
-----
    python evaluate.py --model-config instancedepth/configs/instance_depth.yaml \
        --data-root gid_custom --checkpoint runs/phase3/ckpt_final.pth \
        [--split test] [--use-init-depth] [--mask ground|objects|all]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from instancedepth.build import build_instance_depth_from_yaml
from instancedepth.data.gid_dataset import (
    GIDDatasetConfig, GIDInstanceDepthDataset, collate_gid)
from instancedepth.utils.checkpoint import load_checkpoint

log = logging.getLogger("eval")


@torch.no_grad()
def depth_metrics(pred: torch.Tensor, gt: torch.Tensor,
                  min_d: float = 0.01, max_d: float = 10.0) -> dict | None:
    """pred, gt: (1, H, W) metric depth for ONE frame."""
    valid = (gt > min_d) & (gt <= max_d)
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
                    help="evaluate Stage-1 output instead of refined_depth")
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

    sums: dict = defaultdict(float)
    n = 0
    with torch.inference_mode():
        for batch in loader:
            rgb = batch["image"].to(args.device)
            gt = batch["depth"].to(args.device)
            out = model(rgb, run_instance=not args.use_init_depth,
                        run_refine=not args.use_init_depth)
            pred = out["init_depth"] if args.use_init_depth else out["refined_depth"]
            for b in range(rgb.shape[0]):
                m = depth_metrics(pred[b], gt[b], max_d=max_depth)
                if m is None:
                    continue
                for k, v in m.items():
                    sums[k] += v
                n += 1

    result = {k: round(v / max(n, 1), 4) for k, v in sums.items()}
    result["num_frames"] = n
    log.info("results: %s", json.dumps(result))
    print(json.dumps(result, indent=1))
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
