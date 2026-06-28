"""InstanceDepth 3-phase trainer.

Training script for the InstanceDepth model (Sec. 4.3 of the paper).
Supports three progressive training phases:
  Phase 1: Holistic Depth Initialization (55k iters, lr=1e-5)
  Phase 2: Instance-Aware Depth Rectification (25k iters, lr=1e-5)
  Phase 3: Occlusion-Aware Joint Refinement (25k iters, lr=1e-6)

Usage
-----
    # Phase 1
    python train.py --model-config instancedepth/configs/instance_depth.yaml \\
        --data-root gid_custom --phase 1 --out runs/phase1

    # Phase 2 (init from phase 1)
    python train.py --model-config instancedepth/configs/instance_depth.yaml \\
        --data-root gid_custom --phase 2 --init-from runs/phase1/ckpt_final.pth --out runs/phase2

    # Phase 3 (init from phase 2)
    python train.py --model-config instancedepth/configs/instance_depth.yaml \\
        --data-root gid_custom --phase 3 --init-from runs/phase2/ckpt_final.pth --out runs/phase3
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from instancedepth.build import build_instance_depth_from_yaml
from instancedepth.data.gid_dataset import (
    GIDDatasetConfig, GIDInstanceDepthDataset, build_refine_targets, collate_gid)
from instancedepth.losses.hdi_losses import SigLogLoss, range_segmentation_loss
from instancedepth.losses.instance_losses import InstanceSetCriterion
from instancedepth.losses.refine_losses import RefinementCriterion
from instancedepth.models.instance.matcher import HungarianMatcher
from instancedepth.utils.checkpoint import load_checkpoint, save_checkpoint
from instancedepth.utils.instance_metrics import mean_matched_iou

log = logging.getLogger("train")

PAPER_DEFAULTS = {1: dict(iters=55_000, lr=1e-5),
                  2: dict(iters=25_000, lr=1e-5),
                  3: dict(iters=25_000, lr=1e-6)}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="InstanceDepth 3-phase trainer")
    ap.add_argument("--model-config", required=True)
    ap.add_argument("--data-root", required=True, help="data engine out_root")
    ap.add_argument("--phase", type=int, choices=(1, 2, 3), required=True)
    ap.add_argument("--iters", type=int, default=None, help="default: paper value")
    ap.add_argument("--lr", type=float, default=None, help="default: paper value")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--image-size", type=int, nargs=2, default=(504, 896),
                    help="H W, both divisible by 14; your ZED training size")
    ap.add_argument("--init-from", default=None, help="previous phase checkpoint")
    ap.add_argument("--resume", default=None, help="resume same-phase checkpoint")
    ap.add_argument("--out", required=True)
    ap.add_argument("--save-every", type=int, default=5_000)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--clip-grad", type=float, default=1.0)
    ap.add_argument("--amp", action="store_true",
                    help="enable bf16 autocast (CUDA only). bf16 needs no grad scaler "
                         "and keeps the log-based depth losses (SigLog/range) numerically "
                         "stable; saves memory/time without changing the model.")
    ap.add_argument("--dense-weight-phase3", type=float, default=1.0,
                    help="weight of the dense SigLog anchor on refined_depth in phase 3. "
                         "Anchors the fine-tuned depth encoder/decoder (Sec. 4.3) so it "
                         "cannot drift while the refinement (Eqs. 10-12) trains. Keep > 0.")
    ap.add_argument("--tb-logdir", default=None, help="default: <out>/tb")
    ap.add_argument("--overfit", type=int, default=0,
                    help="DEBUG: train on first N non-empty frames, no flip, mask+class only.")
    ap.add_argument("--unfreeze-backbone", action="store_true",
                    help="DEBUG: also train the backbone (removes frozen-feature confound).")
    ap.add_argument("--ov-wdepth", type=float, default=0.0,
                    help="depth-layer loss weight during overfit; set 1.0 to mirror real training")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    args.iters = args.iters or PAPER_DEFAULTS[args.phase]["iters"]
    args.lr = args.lr or PAPER_DEFAULTS[args.phase]["lr"]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    writer = SummaryWriter(args.tb_logdir or str(out_dir / "tb"))

    # ---- model ------------------------------------------------------------
    model = build_instance_depth_from_yaml(args.model_config).to(args.device)
    if args.init_from:
        info = load_checkpoint(args.init_from, model)
        log.info("init from %s (missing=%d unexpected=%d)",
                 args.init_from, len(info["missing"]), len(info["unexpected"]))
    model.set_phase(args.phase)                     # Sec. 4.3 freezing
    if args.unfreeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad_(True)
        log.info("[debug] backbone UNFROZEN")
    model.train()

    with open(args.model_config) as f:
        mc = yaml.safe_load(f)
    max_depth = float(mc.get("max_depth", 10.0))
    num_ranges = model.num_ranges

    # ---- data -------------------------------------------------------------
    ds = GIDInstanceDepthDataset(GIDDatasetConfig(
        annotations_root=args.data_root, split="train",
        image_size=tuple(args.image_size), max_depth=max_depth,
        hflip_prob=0.0 if args.overfit else 0.5))   # no flip when overfitting
    if args.overfit:                                 # first N frames with >=1 annotated instance
        keep = [i for i, (man, fk) in enumerate(ds.index)
                if len(man["frames"][fk]["instances"]) >= 1][:args.overfit]
        ds = torch.utils.data.Subset(ds, keep)
        log.info("[overfit] %d frames, indices=%s", len(keep), keep)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, collate_fn=collate_gid,
                        drop_last=not bool(args.overfit),
                        pin_memory=args.device.startswith("cuda"))
    log.info("phase %d | %d train frames | %d iters | lr %.1e | image %s",
             args.phase, len(ds), args.iters, args.lr, tuple(args.image_size))

    # ---- losses -----------------------------------------------------------
    siglog = SigLogLoss().to(args.device)
    matcher = HungarianMatcher(w_depth=args.ov_wdepth if args.overfit else 1.0)
    inst_crit = InstanceSetCriterion(
        num_classes=mc.get("instance", {}).get("num_classes", 1),
        w_depth=args.ov_wdepth if args.overfit else 1.0).to(args.device)
    ref_crit = RefinementCriterion().to(args.device)

    # ---- optimizer over trainable params only -----------------------------
    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    log.info("trainable tensors: %d / %d",
             len(params), sum(1 for _ in model.parameters()))

    dev_type = "cuda" if args.device.startswith("cuda") else "cpu"
    use_amp = args.amp and dev_type == "cuda"            # bf16 autocast (no grad scaler)
    if use_amp:
        log.info("AMP: bf16 autocast enabled")

    start_step = 0
    if args.resume:
        info = load_checkpoint(args.resume, model, optim)
        start_step = info["step"]
        log.info("resumed at step %d", start_step)

    # ---- loop -------------------------------------------------------------
    def _infinite(dl):
        while True:
            for b in dl:
                yield b
    data_iter = _infinite(loader)
    t0 = time.time()
    for step in range(start_step, args.iters):
        batch = next(data_iter)
        rgb = batch["image"].to(args.device, non_blocking=True)
        gt_depth = batch["depth"].to(args.device, non_blocking=True)
        targets = [{k: v.to(args.device) for k, v in t.items()}
                   for t in batch["targets"]]

        run_instance = args.phase >= 2
        run_refine = args.phase == 3
        # Forward + loss under bf16 autocast (no-op when use_amp is False). autocast's
        # op policy keeps log/exp/softmax/*_loss in fp32, so SigLog/range/BCE stay safe.
        with torch.autocast(device_type=dev_type, dtype=torch.bfloat16, enabled=use_amp):
            out = model(rgb, run_instance=run_instance, run_refine=run_refine)

            logs = {}
            if args.phase == 1:
                l_sig = siglog(out["init_depth"], gt_depth)
                l_rng = range_segmentation_loss(out["range_logits"], gt_depth,
                                                max_depth, num_ranges)
                loss = l_sig + l_rng
                logs = dict(siglog=l_sig, range=l_rng)

            elif args.phase == 2:
                indices = matcher(out, targets)
                li = inst_crit(out, targets, indices)
                loss = li["loss_total"]
                logs = {k: v for k, v in li.items() if k != "loss_total"}

            else:  # phase 3 -- refinement (Eqs. 10-12) + anchored depth fine-tuning (Sec. 4.3)
                indices = matcher(out, targets)
                meta = out.get("refine_meta")
                if meta is not None and meta["pair_query_idx"].numel() > 0:
                    dt, valid = build_refine_targets(
                        meta["pair_query_idx"], meta["batch_index"], indices, targets)
                    lr_ = ref_crit(out["d_hat"][valid], dt[valid])
                else:
                    z = out["init_depth"].sum() * 0.0
                    lr_ = dict(loss_obj=z, loss_dist=z, loss_ref=z)
                # Dense anchor + refinement-map supervision on the (now differentiable)
                # refined depth: trains the HDI (prevents drift) AND Phi_o (improves the
                # dense map). range CE keeps the range-seg head anchored too. When no
                # pairs fire, refined_depth == init_depth, so this still anchors the HDI.
                l_dense = siglog(out["refined_depth"], gt_depth)
                l_rng = range_segmentation_loss(out["range_logits"], gt_depth,
                                                max_depth, num_ranges)
                loss = lr_["loss_ref"] + args.dense_weight_phase3 * l_dense + l_rng
                logs = dict(obj=lr_["loss_obj"], dist=lr_["loss_dist"],
                            dense=l_dense, range=l_rng)

        # backward/step run OUTSIDE autocast in fp32 (standard AMP recipe).
        optim.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, args.clip_grad)
        if torch.isfinite(grad_norm):
            optim.step()
        else:
            log.warning("step %d: non-finite grad norm (%.3e) -- update skipped",
                        step, float(grad_norm))

        if step % args.log_every == 0 or step == args.iters - 1:
            extras = " ".join(f"{k}={float(v):.4f}" for k, v in logs.items())
            log.info("step %6d/%d loss=%.4f %s (%.2fs/it)",
                     step, args.iters, float(loss), extras,
                     (time.time() - t0) / max(step - start_step + 1, 1))
            # ---- live TensorBoard scalars ----
            writer.add_scalar("train/loss", float(loss), step)
            writer.add_scalar("train/grad_norm", float(grad_norm), step)
            writer.add_scalar("train/lr", optim.param_groups[0]["lr"], step)
            for k, v in logs.items():
                writer.add_scalar(f"train/{k}", float(v), step)
            if args.phase == 2:
                with torch.no_grad():
                    pm, pl = out["pred_masks"], out["pred_logits"]
                    ph, pw = pm.shape[-2:]
                    ious = []
                    for bi in range(rgb.shape[0]):
                        gm = targets[bi]["masks"]
                        if gm.numel() == 0:
                            continue
                        fg = pl[bi].softmax(-1)[:, :-1].max(-1).values > 0.5
                        if fg.sum() == 0:
                            ious.append(0.0); continue
                        pred_b = pm[bi][fg].sigmoid() > 0.5
                        gm_r = torch.nn.functional.interpolate(
                            gm.unsqueeze(1).float(), size=(ph, pw), mode="nearest").squeeze(1) > 0.5
                        ious.append(mean_matched_iou(pred_b.cpu().numpy(), gm_r.cpu().numpy()))
                    if ious:
                        miou = sum(ious) / len(ious)
                        writer.add_scalar("train/mask_mIoU_matched", miou, step)
                        log.info("        mask_mIoU_matched=%.3f", miou)

        if (step + 1) % args.save_every == 0:
            save_checkpoint(str(out_dir / f"ckpt_{step + 1:06d}.pth"),
                            model, optim, step=step + 1, phase=args.phase)

    save_checkpoint(str(out_dir / "ckpt_final.pth"), model, optim,
                    step=args.iters, phase=args.phase)
    writer.close()
    log.info("phase %d done -> %s", args.phase, out_dir / "ckpt_final.pth")


if __name__ == "__main__":
    main()
