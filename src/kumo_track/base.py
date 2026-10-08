from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DetectionResult:
    corners: list[list[float]]  # [[x1,y1],[x2,y1],[x2,y2],[x1,y2]] top-left → clockwise
    score: float
    label: str


class BaseDetector(ABC):
    @abstractmethod
    def detect(self, image_path: str | Path) -> list[DetectionResult]:
        """Run inference on a single image and return all detected objects."""
        ...

    def unload(self) -> None:
        """Remove model from GPU memory. Override if subclass needs custom cleanup."""
        import gc
        import torch
        if hasattr(self, "model"):
            self.model.cpu()
            del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
