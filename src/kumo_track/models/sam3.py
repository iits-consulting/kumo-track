import math
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor

from kumo_track.base import BaseDetector, DetectionResult
from kumo_track.masks import _mask_to_corners
from kumo_track.tiled import tiled_detect

_DEFAULT_MODEL = "facebook/sam3"
_DEFAULT_PROMPTS = ["object"]

# Bound the memory of the full-resolution instance masks. SAM3 returns one
# HxW int64 mask per detected instance; at a low score floor a single 4K frame
# can yield 100+ instances, which interpolated to full resolution exceeds GPU
# memory (and silently spills to slow shared system memory on WSL). We cap the
# *total* mask pixel count (instances x ds_h x ds_w) by downscaling the
# segmentation target size, then rescale the fitted box corners back to full
# resolution. Small detection counts stay at full resolution, so band frames
# (few instances) keep exact boxes; only crowded frames are downscaled, where
# the objects are large and the loss of box precision is negligible.
_DEFAULT_MASK_PIXEL_BUDGET = 2.5e8


def _xyxy_to_corners(box: list[float]) -> list[list[float]]:
    """Convert [x1, y1, x2, y2] to 4-corner format [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]."""
    x1, y1, x2, y2 = box
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


class SAM3Detector(BaseDetector):
    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        text_prompts: list[str] | None = None,
        threshold: float = 0.3,
        device: str | None = None,
        box_mode: str = "obb",
        mask_threshold: float = 0.5,
        mask_pixel_budget: float | None = _DEFAULT_MASK_PIXEL_BUDGET,
        tile_cols: int = 1,
        tile_rows: int = 1,
        tile_overlap: float = 0.2,
        tile_full_frame: bool = False,
        nms_iou: float = 0.5,
    ):
        if box_mode not in ("obb", "aabb"):
            raise ValueError(f"box_mode must be 'obb' or 'aabb', got {box_mode!r}")
        self.model_name = model_name
        self.text_prompts = text_prompts or _DEFAULT_PROMPTS
        self.threshold = threshold
        self.box_mode = box_mode
        self.mask_threshold = mask_threshold
        self.mask_pixel_budget = mask_pixel_budget
        # Tiled (sliding-window) inference. SAM3 resizes every frame to 1008x1008
        # internally, so on a 3-4K frame a thin object is downsampled ~3x and its
        # mask comes out coarse. A tile_cols x tile_rows grid (>1x1) keeps objects
        # near native size; per-tile rotated boxes are merged back with polygon NMS.
        # 1x1 is the plain full-frame path.
        self.tile_cols = tile_cols
        self.tile_rows = tile_rows
        self.tile_overlap = tile_overlap
        self.tile_full_frame = tile_full_frame
        self.nms_iou = nms_iou
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.processor = Sam3Processor.from_pretrained(model_name)
        self.model = Sam3Model.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def detect(self, image_path: str | Path) -> list[DetectionResult]:
        if self.tile_cols > 1 or self.tile_rows > 1:
            return tiled_detect(
                self, image_path, self.tile_cols, self.tile_rows,
                overlap=self.tile_overlap, nms_iou=self.nms_iou,
                full_frame=self.tile_full_frame,
            )
        return self.detect_image(Image.open(image_path).convert("RGB"))

    def detect_image(self, image: Image.Image) -> list[DetectionResult]:
        """Run inference on an in-memory RGB image (used for full frames and tiles)."""
        h, w = image.height, image.width

        results: list[DetectionResult] = []

        for prompt in self.text_prompts:
            inputs = self.processor(images=image, text=prompt, return_tensors="pt")
            inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}

            with torch.inference_mode():
                outputs = self.model(**inputs)

            if self.box_mode == "aabb":
                results.extend(self._detect_aabb(outputs, prompt, h, w))
            else:
                results.extend(self._detect_obb(outputs, prompt, h, w))

        return results

    def _detect_aabb(self, outputs, prompt: str, h: int, w: int) -> list[DetectionResult]:
        """Axis-aligned boxes from `post_process_object_detection` (masks discarded)."""
        detections = self.processor.post_process_object_detection(
            outputs, threshold=self.threshold, target_sizes=[(h, w)]
        )[0]
        return [
            DetectionResult(corners=_xyxy_to_corners(box), score=round(score, 4), label=prompt)
            for score, box in zip(detections["scores"].tolist(), detections["boxes"].tolist())
        ]

    def _detect_obb(self, outputs, prompt: str, h: int, w: int) -> list[DetectionResult]:
        """Rotated boxes fit to SAM3's instance masks via `cv2.minAreaRect`."""
        ds_h, ds_w = self._mask_target_size(outputs, h, w)
        scale_x, scale_y = w / ds_w, h / ds_h

        seg = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=self.threshold,
            mask_threshold=self.mask_threshold,
            target_sizes=[(ds_h, ds_w)],
        )[0]

        results: list[DetectionResult] = []
        masks = seg["masks"].cpu().numpy()
        for score, mask in zip(seg["scores"].tolist(), masks):
            corners = _mask_to_corners(mask, scale_x, scale_y)
            if corners is None:  # empty/degenerate mask
                continue
            results.append(DetectionResult(corners=corners, score=round(score, 4), label=prompt))
        return results

    def _mask_target_size(self, outputs, h: int, w: int) -> tuple[int, int]:
        """Pick a (possibly downscaled) mask resolution that bounds total mask pixels.

        Counts the instances that survive the score threshold and shrinks the
        segmentation target size so `n_keep * ds_h * ds_w <= mask_pixel_budget`,
        preserving aspect ratio. Returns full resolution when the budget is
        disabled or already satisfied.
        """
        if not self.mask_pixel_budget:
            return h, w
        scores = outputs.pred_logits.sigmoid()
        if getattr(outputs, "presence_logits", None) is not None:
            scores = scores * outputs.presence_logits.sigmoid()
        n_keep = int((scores[0] > self.threshold).sum())
        if n_keep <= 0:
            return h, w
        scale = math.sqrt(self.mask_pixel_budget / (n_keep * h * w))
        if scale >= 1.0:
            return h, w
        return max(1, round(h * scale)), max(1, round(w * scale))
