# SAM3 and OWLv2 both need `transformers` (5.x for the SAM3 video tracker). Each
# import is guarded and resolves to None when its dependency isn't installed, so
# importing the package never hard-fails on a missing optional dependency.
try:
    from kumo_track.models.owlv2 import OWLv2Detector
except ImportError:
    OWLv2Detector = None
try:
    from kumo_track.models.sam3 import SAM3Detector
except ImportError:
    SAM3Detector = None

__all__ = ["SAM3Detector", "OWLv2Detector"]
