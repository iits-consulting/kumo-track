"""Mock of the hosted SAM3 inference service for local development — same
endpoints and response shapes, no GPU, no cost.

Simulates the scale-to-zero lifecycle so the app's cold-start handling and the
frontend's "SAM3 waking up…" pill can be exercised:

- The service starts **asleep**. While asleep, every endpoint (including /health)
  returns 503 and all sessions are dropped (they are ephemeral on the real service).
- The first /health poll while asleep starts a wake-up; /health keeps returning 503
  until ``MOCK_COLD_START_S`` (default 20) have elapsed, then 200.
- After ``MOCK_IDLE_S`` (default 120) without any request, it falls asleep again.

Fits are deterministic fakes: a box prompt is echoed back, a point prompt becomes a
box around the point, and propagation drifts each seed by 2 px per frame.

Run (from the repo root, any free port):

    uv run uvicorn scripts.mock_sam3_service:app --port 9911

then point the app at it:

    SAM3_URL=http://127.0.0.1:9911 uv run uvicorn ... (or `poe serve`)
"""

import io
import json
import os
import time
import uuid

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from kumo_track.annotate import rle
from kumo_track.masks import mask_target_size

COLD_START_S = float(os.environ.get("MOCK_COLD_START_S", "20"))
IDLE_S = float(os.environ.get("MOCK_IDLE_S", "120"))

app = FastAPI(title="Mock SAM3 service")

state = {
    "waking_since": None,   # health polling started while asleep
    "last_activity": None,  # None = asleep (starts asleep, like scaled-to-zero)
    "sessions": {},         # sid -> {n_frames, fps, width, height, stride}
}


def _awake() -> bool:
    last = state["last_activity"]
    if last is not None and time.monotonic() - last > IDLE_S:
        state["last_activity"] = None
        state["sessions"].clear()  # replica gone → sessions are lost
        print(f"[mock-sam3] idle {IDLE_S:.0f}s → asleep (sessions dropped)")
    return state["last_activity"] is not None


@app.get("/health")
def health():
    if _awake():
        state["last_activity"] = time.monotonic()
        return {"status": "ok", "device": "mock", "model": "mock/sam3",
                "tracker_loaded": True, "detector_loaded": True,
                "sessions": len(state["sessions"])}
    now = time.monotonic()
    if state["waking_since"] is None:
        state["waking_since"] = now
        print(f"[mock-sam3] waking up ({COLD_START_S:.0f}s cold start)…")
    if now - state["waking_since"] < COLD_START_S:
        raise HTTPException(503, "replica not ready (cold start)")
    state["waking_since"] = None
    state["last_activity"] = now
    print("[mock-sam3] ready")
    return {"status": "ok", "device": "mock", "model": "mock/sam3",
            "tracker_loaded": False, "detector_loaded": False, "sessions": 0}


def _touch():
    """Non-health endpoints 503 while asleep — the real replica isn't there."""
    if not _awake():
        raise HTTPException(503, "replica not ready (cold start)")
    state["last_activity"] = time.monotonic()


def _fit(box, w, h, score=None):
    x1, y1, x2, y2 = (max(0.0, box[0]), max(0.0, box[1]),
                      min(float(w), box[2]), min(float(h), box[3]))
    corners = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    ds_h, ds_w = mask_target_size(h, w)
    binary = rle.rasterize_polygon(corners, ds_h, ds_w, h, w)
    return {"corners": corners, "polygon": corners,
            "mask_rle": rle.encode(binary) if binary.any() else None, "score": score}


def _box_of(fit) -> list[float]:
    xs = [p[0] for p in fit["corners"]]
    ys = [p[1] for p in fit["corners"]]
    return [min(xs), min(ys), max(xs), max(ys)]


@app.post("/detect")
async def detect(image: UploadFile = File(...), prompts: str = Form("object"),
                 threshold: float = Form(0.3), box_mode: str = Form("obb")):
    _touch()
    data = await image.read()
    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        raise HTTPException(400, "could not decode image")
    h, w = arr.shape[:2]
    # One fake detection in the middle of the frame per prompt.
    label = prompts.split(",")[0].strip() or "object"
    return {"detections": [{
        "corners": [[w * 0.375, h * 0.375], [w * 0.625, h * 0.375],
                    [w * 0.625, h * 0.625], [w * 0.375, h * 0.625]],
        "score": 0.85, "label": label,
    }]}


@app.post("/sessions")
async def open_session(file: UploadFile = File(...), stride: int = Form(5)):
    _touch()
    data = await file.read()
    # Probe the clip via a temp file (VideoCapture can't read from memory).
    tmp = f"/tmp/mock-sam3-{uuid.uuid4().hex}"
    with open(tmp, "wb") as f:
        f.write(data)
    cap = cv2.VideoCapture(tmp)
    n_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    info = {
        "n_frames": max(1, (n_src + stride - 1) // stride),
        "fps": cap.get(cv2.CAP_PROP_FPS) or 25.0,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640,
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480,
        "stride": stride,
    }
    cap.release()
    os.unlink(tmp)
    sid = uuid.uuid4().hex
    state["sessions"][sid] = info
    print(f"[mock-sam3] session {sid[:8]} opened: {info}")
    return {"session_id": sid, **info, "is_image": False, "is_folder": False}


@app.delete("/sessions/{sid}")
def delete_session(sid: str):
    _touch()
    if state["sessions"].pop(sid, None) is None:
        raise HTTPException(404, "session not found")
    return {"ok": True}


class SegmentReq(BaseModel):
    frame_idx: int
    obj_id: int
    box: list[float] | None = None
    points: list[list[float]] | None = None
    labels: list[int] | None = None


@app.post("/sessions/{sid}/segment")
def segment(sid: str, req: SegmentReq):
    _touch()
    sess = state["sessions"].get(sid)
    if sess is None:
        raise HTTPException(404, "session not found")
    w, h = sess["width"], sess["height"]
    if req.box:
        box = req.box
    elif req.points:
        # A box around the positive points (fixed pad), like a lazy segmenter.
        pos = [p for p, l in zip(req.points, req.labels or []) if l == 1] or req.points
        xs = [p[0] for p in pos]
        ys = [p[1] for p in pos]
        pad = min(w, h) * 0.08
        box = [min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad]
    else:
        return {"obj_id": req.obj_id, "frame_idx": req.frame_idx, "fit": None}
    return {"obj_id": req.obj_id, "frame_idx": req.frame_idx,
            "fit": _fit(box, w, h, score=0.9)}


class PropagateReq(BaseModel):
    start_frame_idx: int
    reverse: bool = False
    max_steps: int | None = None
    seeds: dict[str, dict]


@app.post("/sessions/{sid}/propagate")
def propagate(sid: str, req: PropagateReq):
    _touch()
    sess = state["sessions"].get(sid)
    if sess is None:
        raise HTTPException(404, "session not found")
    if not req.seeds:
        raise HTTPException(400, "seeds must not be empty")
    w, h, n = sess["width"], sess["height"], sess["n_frames"]

    def stream():
        step = -1 if req.reverse else 1
        boxes = {oid: _box_of(fit) for oid, fit in req.seeds.items()}
        yield json.dumps({"frame_idx": req.start_frame_idx, "objects": {
            oid: _fit(b, w, h) for oid, b in boxes.items()}}) + "\n"
        i, emitted = req.start_frame_idx + step, 0
        while 0 <= i < n and (req.max_steps is None or emitted < req.max_steps):
            time.sleep(0.05)  # feel like inference; keeps Stop testable
            objects = {}
            for oid in boxes:
                b = boxes[oid]
                boxes[oid] = [b[0] + 2 * step, b[1] + step, b[2] + 2 * step, b[3] + step]
                objects[oid] = _fit(boxes[oid], w, h)
            yield json.dumps({"frame_idx": i, "objects": objects}) + "\n"
            i += step
            emitted += 1

    return StreamingResponse(stream(), media_type="application/x-ndjson")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("MOCK_PORT", "9911")))
