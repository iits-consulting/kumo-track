"""HTTP client for the hosted SAM3 inference service + drop-in replacements.

When ``SAM3_URL`` is set, the annotation app routes all SAM 3 work to the remote
service instead of loading SAM 3 in-process. The service mirrors kumo-track's
internal contracts: ``POST /detect`` mirrors
``SAM3Detector`` and the session endpoints mirror ``TrackerManager``. So
:class:`RemoteTrackerManager` and :class:`RemoteSAM3Detector` are drop-in
replacements behind the app's ``tracker_factory`` / detector-loader seams — the
``Fit`` shape (``corners``/``polygon``/``mask_rle``/``score``) is identical.

Cold start: the service scales to zero when idle, so the first request after idle
takes ~2 min while a replica spins up. Every call polls ``GET /health`` first and
retries connection failures / 503 with backoff. Tracking sessions
are ephemeral (lost on idle/restart); recovery is reopen-on-404, which re-uploads
the clip.
"""

import io
import itertools
import json
import os
import time

import httpx

from kumo_track.base import DetectionResult

_FIT_KEYS = ("corners", "polygon", "mask_rle", "score")
_SENTINEL = object()


class SessionNotFound(RuntimeError):
    """The remote tracking session expired or was lost (HTTP 404)."""


def _fit_subset(fit: dict) -> dict:
    """Just the Fit fields the service's propagate seeds understand."""
    return {k: fit.get(k) for k in _FIT_KEYS}


class SAM3Client:
    """Thin, cold-start-aware HTTP client for the SAM3 service."""

    def __init__(self, base_url: str, api_key: str | None = None,
                 timeout: float = 300.0, ready_timeout: float = 300.0,
                 stream_read_timeout: float = 120.0):
        self.base = base_url.rstrip("/")
        self.headers = {"X-API-Key": api_key} if api_key else {}
        self.ready_timeout = ready_timeout
        # Per-read cap for the propagate stream: each FRAME must arrive within
        # this window (the whole stream may run much longer). Never None — a
        # hung service would otherwise block a propagation forever.
        self.stream_read_timeout = stream_read_timeout
        self._client = httpx.Client(timeout=timeout, headers=self.headers)
        self._ready = False

    @classmethod
    def from_env(cls) -> "SAM3Client":
        url = os.environ.get("SAM3_URL")
        if not url:
            raise RuntimeError("SAM3_URL is not set")
        return cls(
            url,
            api_key=os.environ.get("SAM3_API_KEY"),
            timeout=float(os.environ.get("SAM3_TIMEOUT", "300")),
            ready_timeout=float(os.environ.get("SAM3_READY_TIMEOUT", "300")),
            stream_read_timeout=float(os.environ.get("SAM3_STREAM_READ_TIMEOUT", "120")),
        )

    # --- cold start ------------------------------------------------------------

    def wait_until_ready(self, force: bool = False) -> None:
        """Poll ``/health`` until 200 to absorb the scale-from-zero cold start."""
        if self._ready and not force:
            return
        deadline = time.monotonic() + self.ready_timeout
        delay, last = 2.0, None
        while time.monotonic() < deadline:
            try:
                r = httpx.get(f"{self.base}/health", timeout=30)
                if r.status_code == 200:
                    self._ready = True
                    return
                last = f"health {r.status_code}"
            except httpx.HTTPError as exc:
                last = str(exc)
            time.sleep(delay)
            delay = min(delay * 1.5, 15.0)
        raise RuntimeError(f"SAM3 service did not become ready ({last})")

    # --- request plumbing ------------------------------------------------------

    def _send(self, method: str, path: str, **kw) -> httpx.Response:
        """Issue a request, warming once on a cold-start failure (conn error / 503 / 504)."""
        url = f"{self.base}{path}"
        self.wait_until_ready()
        for attempt in range(2):
            try:
                r = self._client.request(method, url, **kw)
            except httpx.HTTPError as exc:
                if attempt == 0:
                    self.wait_until_ready(force=True)
                    continue
                raise RuntimeError(f"SAM3 request failed: {exc}") from exc
            if r.status_code in (503, 504) and attempt == 0:  # replica not ready yet; gateways emit 504 during scale-up
                self.wait_until_ready(force=True)
                continue
            _raise_for_status(r)
            return r
        raise RuntimeError("SAM3 request failed after retry")

    # --- endpoints -------------------------------------------------------------

    def detect(self, image_bytes: bytes, prompts: str, threshold: float,
               box_mode: str = "obb", filename: str = "frame.jpg") -> list[dict]:
        r = self._send(
            "POST", "/detect",
            files={"image": (filename, image_bytes, "image/jpeg")},
            data={"prompts": prompts, "threshold": str(threshold), "box_mode": box_mode},
        )
        return r.json().get("detections", [])

    def open_session(self, file_bytes: bytes, filename: str, stride: int) -> dict:
        r = self._send(
            "POST", "/sessions",
            files={"file": (filename, file_bytes, "application/octet-stream")},
            data={"stride": str(int(stride))},
        )
        return r.json()

    def segment(self, sid: str, frame_idx: int, obj_id: int,
                box=None, points=None, labels=None) -> dict | None:
        body: dict = {"frame_idx": int(frame_idx), "obj_id": int(obj_id)}
        if box is not None:
            body["box"] = [float(v) for v in box]
        if points is not None:
            body["points"] = [[float(x), float(y)] for x, y in points]
        if labels is not None:
            body["labels"] = [int(v) for v in labels]
        r = self._send("POST", f"/sessions/{sid}/segment", json=body)
        return r.json().get("fit")

    def propagate(self, sid: str, start_frame_idx: int, reverse: bool,
                  max_steps: int | None, seeds: dict):
        """Stream propagation as a sync generator of ``{frame_idx, objects}`` records."""
        body = {
            "start_frame_idx": int(start_frame_idx),
            "reverse": bool(reverse),
            "max_steps": None if max_steps is None else int(max_steps),
            "seeds": {str(oid): _fit_subset(fit) for oid, fit in seeds.items()},
        }
        self.wait_until_ready()
        # read= applies PER CHUNK, not to the whole body: the stream may run for
        # hours, but any single frame taking longer than stream_read_timeout
        # means the service hung — surface it instead of blocking forever.
        stream_timeout = httpx.Timeout(
            connect=10.0, read=self.stream_read_timeout, write=60.0, pool=60.0
        )
        try:
            with self._client.stream(
                "POST", f"{self.base}/sessions/{sid}/propagate", json=body,
                timeout=stream_timeout,
            ) as r:
                if r.status_code >= 400:
                    r.read()
                    _raise_for_status(r)
                for line in r.iter_lines():
                    if not line:
                        continue
                    rec = json.loads(line)
                    if "error" in rec:
                        raise RuntimeError(rec["error"])
                    yield rec
        except httpx.ReadTimeout as exc:
            raise RuntimeError(
                f"SAM3 propagate stream stalled (no frame for "
                f"{self.stream_read_timeout:.0f}s)"
            ) from exc

    def delete_session(self, sid: str) -> None:
        try:  # best-effort cleanup; never raise from teardown
            self._client.delete(f"{self.base}/sessions/{sid}", timeout=60)
        except httpx.HTTPError:
            pass

    def close(self) -> None:
        self._client.close()


def _raise_for_status(r: httpx.Response) -> None:
    if r.status_code < 400:
        return
    body = r.text[:200]
    if r.status_code == 404:
        raise SessionNotFound(body)
    if r.status_code == 507:  # inference failed (incl. CUDA OOM)
        raise RuntimeError(f"remote inference failed (507): {body}")
    raise RuntimeError(f"SAM3 service error {r.status_code}: {body}")


class RemoteTrackerManager:
    """Drop-in for ``TrackerManager`` backed by the remote service.

    Same interface ``VideoSession`` / the app calls: ``segment``, ``propagate``,
    ``forget_object``, ``unforget_object``, ``close``. The remote session is opened
    lazily on first use (the clip is uploaded once, then reused) and reopened on a
    404 (expired/lost). ``removed`` objects are filtered out of propagation seeds,
    matching the local tracker.
    """

    def __init__(self, frame_source, client: SAM3Client | None = None):
        self.fs = frame_source
        self.client = client or SAM3Client.from_env()
        self.removed: set[int] = set()
        self._sid: str | None = None

    def _ensure_session(self) -> str:
        if self._sid is None:
            with open(self.fs.path, "rb") as f:
                data = f.read()
            info = self.client.open_session(data, self.fs.name, self.fs.stride)
            self._sid = info["session_id"]
        return self._sid

    def _reopen(self) -> str:
        self._sid = None
        return self._ensure_session()

    def segment(self, frame_idx, obj_id, points=None, labels=None, box=None) -> dict | None:
        sid = self._ensure_session()
        try:
            return self.client.segment(sid, frame_idx, obj_id, box=box, points=points, labels=labels)
        except SessionNotFound:
            sid = self._reopen()
            return self.client.segment(sid, frame_idx, obj_id, box=box, points=points, labels=labels)

    def propagate(self, start_frame_idx, reverse, max_steps, seeds):
        seeds = {oid: ann for oid, ann in seeds.items() if oid not in self.removed and ann}
        if not seeds:
            return
        sid = self._ensure_session()
        try:
            gen = self.client.propagate(sid, start_frame_idx, reverse, max_steps, seeds)
            first = next(gen, _SENTINEL)
        except SessionNotFound:  # session expired before streaming began → reopen once
            sid = self._reopen()
            gen = self.client.propagate(sid, start_frame_idx, reverse, max_steps, seeds)
            first = next(gen, _SENTINEL)
        if first is _SENTINEL:
            return
        for rec in itertools.chain([first], gen):
            fi = rec["frame_idx"]
            if fi == start_frame_idx:
                continue  # start/seed frame — already annotated, don't re-emit
            yield fi, {int(oid): fit for oid, fit in rec["objects"].items()}

    def forget_object(self, obj_id: int) -> None:
        self.removed.add(obj_id)

    def unforget_object(self, obj_id: int) -> None:
        self.removed.discard(obj_id)

    def close(self) -> None:
        if self._sid is not None:
            self.client.delete_session(self._sid)
            self._sid = None


class RemoteSAM3Detector:
    """Drop-in for ``SAM3Detector`` as ``find_all`` uses it: mutable ``text_prompts``
    / ``threshold`` and ``detect_image(pil) -> list[DetectionResult]`` over ``/detect``."""

    def __init__(self, client: SAM3Client | None = None, threshold: float = 0.3,
                 box_mode: str = "aabb", text_prompts: list[str] | None = None):
        self.client = client or SAM3Client.from_env()
        self.threshold = threshold
        self.box_mode = box_mode
        self.text_prompts = list(text_prompts) if text_prompts else ["object"]

    def detect_image(self, image) -> list[DetectionResult]:
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="JPEG")
        prompts = ", ".join(self.text_prompts)
        dets = self.client.detect(buf.getvalue(), prompts, self.threshold, self.box_mode)
        out: list[DetectionResult] = []
        for d in dets:
            corners = d.get("corners")
            if not corners:
                continue
            out.append(DetectionResult(
                corners=corners,
                score=float(d["score"]) if d.get("score") is not None else 0.0,
                label=d.get("label") or prompts,
            ))
        return out
