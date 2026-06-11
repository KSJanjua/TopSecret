"""SAM3 video segmentation+tracking backends.

Replaces the paper's two-stage pipeline (SAM [27] mask extraction + DEVA [13]
identity tracking, Sec. 3 "Mask Generation" / "Tracking Identity Generation")
with SAM3 promptable-concept video segmentation, which produces instance masks
AND temporally consistent object IDs in a single pass.

[Reasonable Assumption] Backend choice and exact API plumbing. Two real
backends are provided:

  * "native" — github.com/facebookresearch/sam3 session API:
        predictor = build_sam3_video_predictor(...)
        handle_request({"type": "start_session", "resource_path": <jpeg dir>})
        handle_request({"type": "add_prompt", "session_id", "frame_index", "text"})
        <propagate>  -> per-frame {obj_id: mask}
    NOTE: one SAM3 session tracks ONE text concept. Multi-category datasets
    therefore run one session per prompt; merging happens in identity.py.
  * "hf" — HuggingFace transformers Sam3VideoModel / Sam3VideoProcessor.

A "mock" backend (deterministic moving blobs) lets the whole engine run in CI
without GPU/weights.

Every backend returns the same normalized structure:

    frames_out: Dict[int, Dict[int, MaskObs]]
        frame_index -> { local_obj_id -> MaskObs(mask: bool (H,W), score: float) }
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .config import SAM3Config

log = logging.getLogger("data_engine.sam3")

FramesOut = Dict[int, Dict[int, "MaskObs"]]


@dataclass
class MaskObs:
    mask: np.ndarray            # bool (H, W) at full image resolution
    score: float = 1.0


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _to_bool_mask(m, hw: Tuple[int, int]) -> np.ndarray:
    """Accept torch/np, (H,W)/(1,H,W), logits/probs/bool -> bool (H,W)."""
    try:
        import torch

        if isinstance(m, torch.Tensor):
            m = m.detach().float().cpu().numpy()
    except ImportError:
        pass
    m = np.asarray(m)
    m = np.squeeze(m)
    if m.ndim != 2:
        raise ValueError(f"mask has unexpected shape {m.shape}")
    if m.dtype == np.bool_:
        out = m
    elif m.min() < 0.0 or m.max() > 1.0:    # logits
        out = m > 0.0
    else:                                    # probabilities or already {0,1}
        out = m > 0.5
    if out.shape != hw:                      # resize low-res masks to image size
        import cv2

        out = cv2.resize(out.astype(np.uint8), (hw[1], hw[0]),
                         interpolation=cv2.INTER_NEAREST).astype(bool)
    return out


def _area_ok(mask: np.ndarray, min_frac: float) -> bool:
    return mask.sum() >= min_frac * mask.size


class VideoSegmenter:
    """Backend interface."""

    def track_concept(
        self, rgb_paths: Sequence[Path], hw: Tuple[int, int], prompt: str
    ) -> FramesOut:
        raise NotImplementedError

    def close(self) -> None:  # noqa: B027
        pass


# --------------------------------------------------------------------------- #
# native facebookresearch/sam3 backend
# --------------------------------------------------------------------------- #
class NativeSAM3Segmenter(VideoSegmenter):
    """Session-based API of the official SAM3 repository."""

    def __init__(self, cfg: SAM3Config) -> None:
        from sam3.model_builder import build_sam3_video_predictor

        kwargs = {}
        if cfg.checkpoint_path:
            kwargs["checkpoint_path"] = cfg.checkpoint_path
        self.predictor = build_sam3_video_predictor(**kwargs)
        self.cfg = cfg
        self._tmp: Optional[Path] = None

    # SAM3 start_session accepts a JPEG folder or an MP4 file. If the RGB
    # frames are not all .jpg, stage them into a temporary JPEG folder.
    def _stage_frames(self, rgb_paths: Sequence[Path]) -> Path:
        if all(p.suffix.lower() in (".jpg", ".jpeg") for p in rgb_paths):
            parents = {p.parent for p in rgb_paths}
            if len(parents) == 1:
                d = parents.pop()
                jpgs = [q for q in d.iterdir()
                        if q.is_file() and q.suffix.lower() in (".jpg", ".jpeg")]
                if len(jpgs) == len(rgb_paths):   # dir contains exactly our frames
                    return d
        import cv2

        self._tmp = Path(tempfile.mkdtemp(prefix="sam3_frames_"))
        for i, p in enumerate(rgb_paths):
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            cv2.imwrite(str(self._tmp / f"{i:06d}.jpg"), img,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
        return self._tmp

    def _iter_propagation(self, session_id) -> Iterable[dict]:
        """Yield per-frame output dicts, tolerating minor repo API drift."""
        req = dict(type="propagate_in_video", session_id=session_id,
                   start_frame_index=0)
        if hasattr(self.predictor, "handle_stream_request"):
            yield from self.predictor.handle_stream_request(request=req)
            return
        resp = self.predictor.handle_request(request=req)  # non-streaming fallback
        outs = resp.get("outputs", resp)
        if isinstance(outs, dict):
            for fi in sorted(outs):
                yield {"frame_index": fi, **outs[fi]} if isinstance(outs[fi], dict) \
                    else {"frame_index": fi, "results": outs[fi]}
        else:
            yield from outs

    @staticmethod
    def _frame_payload(out: dict) -> Tuple[int, List[int], List, List[float]]:
        fi = out.get("frame_index", out.get("frame_idx"))
        body = out.get("outputs", out)
        ids = body.get("out_obj_ids", body.get("obj_ids", body.get("object_ids", [])))
        masks = body.get("out_binary_masks",
                         body.get("out_mask_logits", body.get("masks", [])))
        scores = body.get("out_probs", body.get("scores", [1.0] * len(ids)))
        return int(fi), list(ids), list(masks), [float(s) for s in scores]

    def track_concept(self, rgb_paths, hw, prompt) -> FramesOut:
        frames_dir = self._stage_frames(rgb_paths)
        resp = self.predictor.handle_request(
            request=dict(type="start_session", resource_path=str(frames_dir))
        )
        session_id = resp["session_id"]
        try:
            first = self.predictor.handle_request(
                request=dict(type="add_prompt", session_id=session_id,
                             frame_index=self.cfg.prompt_frame_index, text=prompt)
            )
            frames_out: FramesOut = {}
            stream: List[dict] = list(self._iter_propagation(session_id))
            if not stream and "outputs" in first:      # single-frame degenerate case
                stream = [{"frame_index": self.cfg.prompt_frame_index,
                           "outputs": first["outputs"]}]
            for out in stream:
                fi, ids, masks, scores = self._frame_payload(out)
                per: Dict[int, MaskObs] = {}
                for oid, m, s in zip(ids, masks, scores):
                    if s < self.cfg.min_object_score:
                        continue
                    bm = _to_bool_mask(m, hw)
                    if _area_ok(bm, self.cfg.min_mask_area_frac):
                        per[int(oid)] = MaskObs(bm, s)
                frames_out[fi] = per
            return frames_out
        finally:
            try:
                self.predictor.handle_request(
                    request=dict(type="close_session", session_id=session_id)
                )
            except Exception:                          # noqa: BLE001
                log.debug("close_session failed (non-fatal)")

    def close(self) -> None:
        if self._tmp is not None:
            shutil.rmtree(self._tmp, ignore_errors=True)
            self._tmp = None


# --------------------------------------------------------------------------- #
# HuggingFace transformers backend
# --------------------------------------------------------------------------- #
class HFSAM3Segmenter(VideoSegmenter):
    def __init__(self, cfg: SAM3Config) -> None:
        import torch
        from transformers import Sam3VideoModel, Sam3VideoProcessor

        self.device = cfg.device if torch.cuda.is_available() else "cpu"
        self.model = Sam3VideoModel.from_pretrained(cfg.hf_model_id).to(self.device)
        self.model.eval()
        self.processor = Sam3VideoProcessor.from_pretrained(cfg.hf_model_id)
        self.cfg = cfg

    def track_concept(self, rgb_paths, hw, prompt) -> FramesOut:
        import cv2
        import torch

        video = [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in rgb_paths]
        session = self.processor.init_video_session(
            video=video, inference_device=self.device
        )
        self.processor.add_text_prompt(session, prompt)

        frames_out: FramesOut = {}
        with torch.inference_mode():
            for model_out in self.model.propagate_in_video_iterator(
                inference_session=session
            ):
                proc = self.processor.postprocess_outputs(session, model_out)
                fi = int(getattr(model_out, "frame_idx", len(frames_out)))
                ids = proc.get("object_ids", proc.get("obj_ids", []))
                masks = proc.get("masks", [])
                scores = proc.get("scores", [1.0] * len(ids))
                per: Dict[int, MaskObs] = {}
                for oid, m, s in zip(ids, masks, scores):
                    s = float(s)
                    if s < self.cfg.min_object_score:
                        continue
                    bm = _to_bool_mask(m, hw)
                    if _area_ok(bm, self.cfg.min_mask_area_frac):
                        per[int(oid)] = MaskObs(bm, s)
                frames_out[fi] = per
        return frames_out


# --------------------------------------------------------------------------- #
# mock backend (tests / dry runs)
# --------------------------------------------------------------------------- #
class MockSegmenter(VideoSegmenter):
    """Two deterministic moving discs per concept; disc 2 occludes disc 1."""

    def __init__(self, cfg: SAM3Config) -> None:
        self.cfg = cfg

    def track_concept(self, rgb_paths, hw, prompt) -> FramesOut:
        h, w = hw
        yy, xx = np.mgrid[0:h, 0:w]
        frames_out: FramesOut = {}
        is_ground = prompt in self.cfg.ground_prompts
        for fi in range(len(rgb_paths)):
            per: Dict[int, MaskObs] = {}
            if is_ground:
                per[1] = MaskObs((yy > int(0.7 * h)), 0.95)
            else:
                r = max(6, h // 8)
                c1 = (h // 2, int(w * 0.25 + fi * w * 0.02) % w)
                c2 = (h // 2, int(w * 0.35 + fi * w * 0.02) % w)
                m1 = (yy - c1[0]) ** 2 + (xx - c1[1]) ** 2 <= r * r
                m2 = (yy - c2[0]) ** 2 + (xx - c2[1]) ** 2 <= (r + 2) ** 2
                m1 = m1 & ~m2                            # 2 occludes 1
                per[1] = MaskObs(m1, 0.97)
                per[2] = MaskObs(m2, 0.99)
            frames_out[fi] = per
        return frames_out


def build_segmenter(cfg: SAM3Config) -> VideoSegmenter:
    if cfg.backend == "native":
        return NativeSAM3Segmenter(cfg)
    if cfg.backend == "hf":
        return HFSAM3Segmenter(cfg)
    if cfg.backend == "mock":
        return MockSegmenter(cfg)
    raise ValueError(f"unknown SAM3 backend '{cfg.backend}'")
