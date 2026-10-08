from pathlib import Path

import torch
from PIL import Image
from transformers import Owlv2ForObjectDetection, Owlv2Processor

from kumo_track.base import BaseDetector, DetectionResult
from kumo_track.tiled import tiled_detect

_DEFAULT_MODEL = "google/owlv2-large-patch14-ensemble"
_DEFAULT_PROMPTS = ["object"]


def _xyxy_to_corners(box: list[float]) -> list[list[float]]:
    x1, y1, x2, y2 = box
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


class OWLv2Detector(BaseDetector):
    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        text_prompts: list[str] | None = None,
        threshold: float = 0.1,
        device: str | None = None,
        tile_cols: int = 1,
        tile_rows: int = 1,
        tile_overlap: float = 0.2,
        tile_full_frame: bool = False,
        nms_iou: float = 0.5,
    ):
        self.model_name = model_name
        self.text_prompts = text_prompts or _DEFAULT_PROMPTS
        self.threshold = threshold
        # Tiled (sliding-window) inference. OWLv2's image processor resizes every
        # frame to a fixed 1008x1008 square, so on a 3-4K frame a ~130px object
        # shrinks to ~35px and the patch14 stride erases it. A tile_cols x tile_rows
        # grid (>1x1) keeps objects near native size; per-tile boxes are merged back
        # with rotated-box NMS. 1x1 is the plain full-frame path.
        self.tile_cols = tile_cols
        self.tile_rows = tile_rows
        self.tile_overlap = tile_overlap
        self.tile_full_frame = tile_full_frame
        self.nms_iou = nms_iou
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.processor = Owlv2Processor.from_pretrained(model_name)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_name).to(self.device)
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
        """Run inference on an in-memory RGB image (used for full frames and tiles).

        Exposed separately from `detect` so `kumo_track.tiled.tiled_detect` can run
        the model on overlapping crops: the processor resizes every frame to a
        1008x1008 square, so on a 3-4K frame a ~130px object shrinks to ~35px
        and the backbone stride erases it. Cropping first keeps objects near their
        native size, which is the difference between ~0 and useful recall here.
        """
        h, w = image.height, image.width

        # Query ONE prompt per forward pass (like SAM3 / MM-Grounding-DINO) and
        # label every box with that prompt. OWLv2 returns a `labels` index into the
        # query list and a matching `text_labels` string, but prompting a single
        # phrase makes the class unambiguous so we can attribute every box directly.
        results: list[DetectionResult] = []
        for prompt in self.text_prompts:
            inputs = self.processor(text=[[prompt]], images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}

            with torch.inference_mode():
                outputs = self.model(**inputs)

            detections = self.processor.post_process_grounded_object_detection(
                outputs,
                threshold=self.threshold,
                target_sizes=[(h, w)],
            )[0]

            for box, score in zip(detections["boxes"], detections["scores"]):
                x1, y1, x2, y2 = box.tolist()
                # clip to image bounds
                x1, x2 = max(0.0, x1), min(float(w), x2)
                y1, y2 = max(0.0, y1), min(float(h), y2)
                # skip degenerate boxes (nearly zero area)
                if (x2 - x1) < 2.0 or (y2 - y1) < 2.0:
                    continue
                results.append(DetectionResult(
                    corners=_xyxy_to_corners([x1, y1, x2, y2]),
                    score=round(score.item(), 4),
                    label=prompt,
                ))
        return results
