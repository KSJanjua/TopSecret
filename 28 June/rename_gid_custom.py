#!/usr/bin/env python3
"""Rename gid_custom mask files + rewrite annotations.json to the <NNN> convention.

Your source RGB/depth files were renamed to the frame number (e.g.
``zed_..._204_left_rgb.png`` -> ``204.png``, ``..._204_..._np.npy`` -> ``204.npy``).
This script makes the generated ``gid_custom`` tree match that convention:

  per sequence (``<root>/<batch>/<sequence>/``):
    * renames files in   object_masks/   ground_masks/   object_masks_colored/
        <old_stem>.<ext>  ->  <NNN>.<ext>          (extension preserved per file)
    * rewrites annotations.json so that
        - frame keys                 <old_stem>      ->  <NNN>
        - rgb / depth_npy / depth_png basename       ->  <NNN>.<orig-ext>  (dir kept)
        - object_mask / ground_mask  basename        ->  <NNN>.png         (dir kept)
        - any other path field whose stem == old_stem (e.g. a colored field)

The frame number is the LAST integer in the old frame name (the rgb stem) -- the
same rule ``data_engine/discover._frame_number`` already uses -- so RGB, depth and
masks stay aligned to the identical physical frame.

SAFE BY DEFAULT: prints a plan and changes NOTHING. Pass --apply to write.
IDEMPOTENT: a sequence whose manifest keys are already <NNN> is detected and skipped,
so re-running is harmless. The first run backs up each manifest to annotations.json.bak.

Usage
-----
    python rename_gid_custom.py --root gid_custom            # dry run (preview)
    python rename_gid_custom.py --root gid_custom --apply    # actually rename + rewrite
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

log = logging.getLogger("rename_gid_custom")

MASK_FOLDERS = ("object_masks", "ground_masks", "object_masks_colored")
SOURCE_FIELDS = ("rgb", "depth_npy", "depth_png")     # different stem than the key -> set by number
_INT = re.compile(r"(\d+)")


def frame_number(stem: str) -> Optional[int]:
    """Last integer in the stem = frame index (e.g. 'zed_..._204_left_rgb' -> 204)."""
    nums = _INT.findall(stem)
    return int(nums[-1]) if nums else None


def _new_name(frame_no: int, pad: int, suffix: str) -> str:
    return f"{frame_no:0{pad}d}{suffix}"


def _already_converted(frames: Dict[str, dict]) -> bool:
    """True if every frame key is a pure (already-renamed) integer string."""
    return bool(frames) and all(k.isdigit() for k in frames)


def plan_sequence(seq_dir: Path, pad: int) -> Optional[dict]:
    """Build a rename/rewrite plan for one sequence, or None to skip.

    Returns a dict with: manifest path, new manifest dict, and a list of
    (src_path, dst_path) file renames. Raises ValueError on an unsafe mapping.
    """
    ann_path = seq_dir / "annotations.json"
    if not ann_path.is_file():
        return None
    manifest = json.loads(ann_path.read_text())
    frames: Dict[str, dict] = manifest.get("frames", {})
    if not frames:
        log.info("[%s] no frames in manifest -> skip", seq_dir)
        return None
    if _already_converted(frames):
        log.info("[%s] already converted (keys are <NNN>) -> skip", seq_dir)
        return None

    # ---- map every old stem -> frame number, and verify uniqueness ----------
    stem_to_no: Dict[str, int] = {}
    for stem in frames:
        n = frame_number(stem)
        if n is None:
            raise ValueError(f"{seq_dir}: frame key '{stem}' has no integer -> cannot map")
        stem_to_no[stem] = n
    if len(set(stem_to_no.values())) != len(stem_to_no):
        dupes = sorted({n for n in stem_to_no.values()
                        if list(stem_to_no.values()).count(n) > 1})
        raise ValueError(f"{seq_dir}: non-unique frame numbers {dupes[:10]} -> refusing")

    # ---- physical file renames in the mask folders --------------------------
    renames: List[Tuple[Path, Path]] = []
    for folder in MASK_FOLDERS:
        fdir = seq_dir / folder
        if not fdir.is_dir():
            continue
        on_disk: Dict[str, List[Path]] = {}
        for p in fdir.iterdir():
            if p.is_file():
                on_disk.setdefault(p.stem, []).append(p)
        for stem, n in stem_to_no.items():
            for src in on_disk.get(stem, []):
                dst = src.with_name(_new_name(n, pad, src.suffix))
                if dst == src:
                    continue
                if dst.exists():
                    raise ValueError(f"{dst} already exists -> refusing to clobber")
                renames.append((src, dst))

    # ---- rewrite the manifest (keys + path fields) --------------------------
    new_frames: Dict[str, dict] = {}
    for stem in sorted(frames, key=lambda s: stem_to_no[s]):
        n = stem_to_no[stem]
        rec = dict(frames[stem])                       # shallow copy; instances kept as-is
        for field in SOURCE_FIELDS:
            v = rec.get(field)
            if isinstance(v, str) and v:
                rec[field] = str(Path(v).with_name(_new_name(n, pad, Path(v).suffix)))
        # any remaining path field whose basename is the old stem (object_mask,
        # ground_mask, a colored field, ...) -> rename by number, keep its dir+ext.
        for k, v in list(rec.items()):
            if k in SOURCE_FIELDS or k == "instances":
                continue
            if isinstance(v, str) and v and Path(v).stem == stem:
                rec[k] = str(Path(v).with_name(_new_name(n, pad, Path(v).suffix)))
        new_frames[_new_name(n, pad, "")] = rec        # key = '<NNN>' (no extension)

    new_manifest = dict(manifest)
    new_manifest["frames"] = new_frames
    return {"ann_path": ann_path, "manifest": new_manifest, "renames": renames}


def apply_plan(plan: dict) -> None:
    # back up the manifest once (preserve the very first original)
    bak = plan["ann_path"].with_suffix(".json.bak")
    if not bak.exists():
        shutil.copy2(plan["ann_path"], bak)
    for src, dst in plan["renames"]:
        src.rename(dst)
    tmp = plan["ann_path"].with_suffix(".json.tmp")
    tmp.write_text(json.dumps(plan["manifest"], indent=1))
    tmp.replace(plan["ann_path"])


def iter_sequences(root: Path):
    """Yield every sequence dir (one that contains annotations.json) under root."""
    for ann in sorted(root.rglob("annotations.json")):
        yield ann.parent


def verify_paths(root: Path) -> int:
    """Check that every rgb/depth/mask path in every manifest resolves on disk.

    Returns the number of missing paths (0 = healthy). This catches a wrong
    extension assumption on the source files (e.g. rgb was .jpg, not .png).
    """
    checked = missing = 0
    for ann in sorted(root.rglob("annotations.json")):
        m = json.loads(ann.read_text())
        for key, rec in m.get("frames", {}).items():
            for field in ("rgb", "depth_npy", "depth_png", "object_mask", "ground_mask"):
                p = rec.get(field)
                if isinstance(p, str) and p:
                    checked += 1
                    if not Path(p).exists():
                        missing += 1
                        log.warning("MISSING %s [%s/%s] -> %s",
                                    field, ann.parent.name, key, p)
    log.info("verify: %d path(s) checked, %d missing", checked, missing)
    return missing


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="gid_custom output root")
    ap.add_argument("--pad", type=int, default=3, help="zero-pad width for <NNN> (default 3)")
    ap.add_argument("--apply", action="store_true",
                    help="actually rename + rewrite (default: dry run / preview only)")
    ap.add_argument("--verify-only", action="store_true",
                    help="don't rename anything; just check that all manifest paths exist")
    ap.add_argument("--verbose", action="store_true", help="print every file rename")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"root not found: {root}")

    if args.verify_only:
        raise SystemExit(0 if verify_paths(root) == 0 else 1)

    seqs = list(iter_sequences(root))
    log.info("found %d sequence(s) under %s", len(seqs), root)
    tot_files = tot_seqs = skipped = errors = 0

    for seq_dir in seqs:
        try:
            plan = plan_sequence(seq_dir, args.pad)
        except ValueError as e:
            log.error("SKIP %s", e)
            errors += 1
            continue
        if plan is None:
            skipped += 1
            continue
        tot_seqs += 1
        tot_files += len(plan["renames"])
        if args.verbose or not args.apply:
            for src, dst in plan["renames"][: (10 ** 9 if args.verbose else 4)]:
                log.info("  %s/%s -> %s", seq_dir.name, src.parent.name + "/" + src.name, dst.name)
            extra = len(plan["renames"]) - (len(plan["renames"]) if args.verbose else 4)
            if not args.verbose and extra > 0:
                log.info("  %s ... (+%d more files)", seq_dir.name, extra)
        if args.apply:
            apply_plan(plan)

    mode = "APPLIED" if args.apply else "DRY RUN (no changes written)"
    log.info("%s | sequences changed=%d  files renamed=%d  skipped=%d  errors=%d",
             mode, tot_seqs, tot_files, skipped, errors)
    if args.apply:
        miss = verify_paths(root)
        if miss:
            log.error("%d path(s) do not resolve -- check the source file extensions "
                      "(see --verify-only).", miss)
    elif tot_seqs:
        log.info("re-run with --apply to perform the above changes")


if __name__ == "__main__":
    main()
