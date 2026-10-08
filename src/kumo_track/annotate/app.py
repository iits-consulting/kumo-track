"""FastAPI app for SAM3-assisted video annotation with windowed tracking.

Flow: pick a clip → open (no full decode; frames stream on demand) → seed an object
on a frame with points/box → propagate forward/back/whole-clip (windowed, cancellable)
→ correct boxes/masks by hand → export images + labels.json. One SAM3 model loads
once; a bounded per-video LRU keeps several clips open at once (two users on different
clips don't evict each other), while every GPU call stays serialised by one lock. The
DB (SQLite for dev, PostgreSQL in deployment) is the source of truth; videos/exports
live behind a storage abstraction (local FS or Azure Blob); the tracker session is
disposable cache.

``create_app`` is a factory so tests can inject a fake tracker (no GPU/weights) and
skip model loading.
"""

import asyncio
import base64
import gc
import json
import os
import secrets as pysecrets
import threading
import tomllib
from collections import OrderedDict
from collections.abc import Callable
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path

import cv2
import httpx
from PIL import Image as PILImage

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from kumo_track.annotate import db, rle
from kumo_track.annotate.session import VideoSession
from kumo_track.annotate.storage import VIDEO_EXTS, make_storage
from kumo_track.masks import _mask_polygon, _mask_to_corners, mask_target_size

VIDEO_DIR = Path(os.environ.get("VIDEO_DIR", "data/videos"))
# Metadata store: a postgresql:// URL (deployment) or a SQLite path (dev/tests).
DB_URL = os.environ.get("DATABASE_URL") or os.environ.get("ANNOTATION_DB", "outputs/annotations.db")
STATIC = Path(__file__).parent / "static"
DEFAULT_STRIDE = int(os.environ.get("STRIDE", "5"))
# How many open clips (VideoSessions) to keep hot at once. Two users on different
# clips must not evict each other, so keep this ≥ the expected concurrent-clip count.
MAX_ACTIVE_SESSIONS = int(os.environ.get("MAX_ACTIVE_SESSIONS", "4"))
# Auth gate (off for dev/tests). When on, a request without X-Kumo-User is 401.
REQUIRE_AUTH = os.environ.get("REQUIRE_AUTH", "").lower() in ("1", "true", "yes")
# When set, every request (except health probes) must carry this value in
# X-Kumo-Proxy-Auth — proof it came through the login-node gateway. This is the
# actual trust anchor for X-Kumo-User: the ingress IP allow-list alone is not
# (App Service outbound IPs are shared with co-located tenants).
PROXY_SHARED_SECRET = os.environ.get("PROXY_SHARED_SECRET") or None
DEV_USER = os.environ.get("DEV_USER", "dev")
CONFIG_FILE = Path(os.environ.get("CONFIG_FILE", "config.toml"))
EDIT_TOOLS = ("polygon", "brush")


def load_config() -> dict:
    """Read the optional config.toml. Missing file / bad value → safe defaults.

    Only the frontend's manual-edit tool lives here today; the API itself is
    tool-agnostic (both /api/annotations and /api/mask are always served).
    """
    edit_tool = "polygon"
    try:
        data = tomllib.loads(CONFIG_FILE.read_text())
        choice = (data.get("annotation") or {}).get("edit_tool")
        if choice in EDIT_TOOLS:
            edit_tool = choice
        elif choice is not None:
            print(f"[annotate] config: unknown edit_tool {choice!r}, using {edit_tool!r}")
    except FileNotFoundError:
        pass
    except (tomllib.TOMLDecodeError, OSError) as exc:
        print(f"[annotate] config: could not read {CONFIG_FILE} ({exc}); using defaults")
    return {"edit_tool": edit_tool}


# --- request bodies ------------------------------------------------------------


class OpenReq(BaseModel):
    name: str
    stride: int = DEFAULT_STRIDE


class ObjectReq(BaseModel):
    video_id: int
    label: str
    # created_by is intentionally absent: attribution comes from the trusted
    # X-Kumo-User header (see current_user), never from the client body.


class ObjectPatchReq(BaseModel):
    label: str | None = None
    static: bool | None = None
    hidden: bool | None = None
    color: str | None = None


class SegmentReq(BaseModel):
    video_id: int
    frame_idx: int
    obj_id: int
    points: list[list[float]] | None = None
    labels: list[int] | None = None
    box: list[float] | None = None


class PropagateReq(BaseModel):
    video_id: int
    start_frame_idx: int
    direction: str = "fwd"          # "fwd" | "rev"
    n_frames: int | None = None     # None = to clip start/end
    full_clip: bool = False         # bidirectional: rev to 0, then fwd to end
    active_obj: int | None = None   # selected object — always tracks (see below)
    # legacy fields (older clients); honoured if the new ones are absent.
    reverse: bool | None = None
    max_frames: int | None = None


class DeleteReq(BaseModel):
    object_id: int
    frame_idx: int


class ManualReq(BaseModel):
    video_id: int
    object_id: int
    frame_idx: int
    corners: list[list[float]] | None = None
    polygon: list[list[float]] | None = None


class MaskReq(BaseModel):
    video_id: int
    object_id: int
    frame_idx: int
    mask_png: str  # base64 PNG (RGBA); any non-transparent pixel is the object


class FindAllReq(BaseModel):
    video_id: int
    frame_idx: int
    label: str
    query: str | None = None
    threshold: float = 0.50
    detector: str = "sam3"  # "sam3" | "owlv2"


class ExportReq(BaseModel):
    video_id: int
    out_name: str
    quality: int = 95
    include_images: bool = True  # False = labels.json only


def _obb_from_polygon(polygon: list[list[float]]) -> list[list[float]] | None:
    """minAreaRect corners of a polygon — the single polygon→OBB path (manual edits)."""
    import numpy as np

    pts = np.asarray(polygon, dtype=np.float32)
    if pts.shape[0] < 3:
        return None
    rect = cv2.minAreaRect(pts)
    (_, _), (rw, rh), _ = rect
    if rw * rh < 1.0:
        return None
    return cv2.boxPoints(rect).tolist()


def create_app(
    tracker_factory: Callable | None = None,
    load_on_start: bool = True,
    db_path: str | None = None,
    video_dir: Path | None = None,
    config: dict | None = None,
    require_auth: bool | None = None,
    proxy_secret: str | None = None,
) -> FastAPI:
    db_url = db_path or DB_URL
    video_dir = video_dir or VIDEO_DIR
    config = config or load_config()
    require_auth = REQUIRE_AUTH if require_auth is None else require_auth
    # None → env default; "" → explicitly disabled (tests).
    proxy_secret = PROXY_SHARED_SECRET if proxy_secret is None else (proxy_secret or None)
    database = db.Database(db_url)
    store = make_storage(video_dir)
    # Polygons are a polygon-mode editing convenience only; brush mode stores none.
    store_polygon = config["edit_tool"] == "polygon"

    # Remote mode: when SAM3_URL is set, all SAM 3 inference goes to the hosted
    # service (no local torch/weights). One shared, cold-start-aware client backs
    # both the tracker and the detector. Otherwise everything runs locally as before.
    remote = bool(os.environ.get("SAM3_URL"))
    remote_client = None
    if remote:
        from kumo_track.annotate.remote import RemoteTrackerManager, SAM3Client

        remote_client = SAM3Client.from_env()

    # Resolve the tracker: an explicit factory (tests) wins; else remote or local.
    if tracker_factory is None:
        if remote:
            tracker_factory = lambda fs: RemoteTrackerManager(fs, client=remote_client)  # noqa: E731
        else:
            from kumo_track.annotate.tracker import TrackerManager

            tracker_factory = TrackerManager

    # Per-video LRU of open sessions (two users on different clips coexist) guarded
    # by its own lock — NOT gpu_lock, so cache lookups never block on decode or
    # inference. gpu_lock stays a single lock serialising every GPU call (one GPU).
    # `pins` counts in-flight long operations per video (propagate streams,
    # exports, GPU calls): a pinned session is exempt from LRU eviction, so a
    # concurrent open can never close a VideoCapture that is still being read.
    sessions: "OrderedDict[int, VideoSession]" = OrderedDict()
    pins: dict[int, int] = {}
    session_lock = threading.Lock()
    gpu_lock = asyncio.Lock()
    detectors: dict = {"owl": None, "sam3": None}

    def _conn():
        return database.connection()

    # Paths the auth gate must never refuse: orchestrator liveness/readiness probes
    # don't send X-Kumo-User, so with REQUIRE_AUTH=on the container would never
    # become healthy. Kept in sync with the health routes registered below.
    HEALTH_PATHS = ("/api/health", "/healthz")

    def current_user(request: Request) -> str:
        """The acting user from the gateway-injected header (attribution + gate)."""
        if request.url.path in HEALTH_PATHS:
            return DEV_USER  # gate-exempt: probes have no identity header
        if proxy_secret is not None:
            # Only the gateway knows this value; without it X-Kumo-User is just a
            # client-writable header. Constant-time compare, and the same 401 as
            # a missing user so probes can't distinguish which layer refused.
            supplied = request.headers.get("X-Kumo-Proxy-Auth", "")
            if not pysecrets.compare_digest(supplied, proxy_secret):
                raise HTTPException(401, "authentication required")
        user = request.headers.get("X-Kumo-User")
        if not user:
            if require_auth:
                raise HTTPException(401, "authentication required")
            return DEV_USER
        return user

    def _cache_get(video_id: int) -> VideoSession | None:
        with session_lock:
            sess = sessions.get(video_id)
            if sess is not None:
                sessions.move_to_end(video_id)
            return sess

    def _cache_put(video_id: int, new_sess: VideoSession) -> tuple[VideoSession, VideoSession | None]:
        """Insert a freshly-built session, LRU-evicting if over the cap.

        Returns ``(session_to_use, session_to_close)``. If a concurrent open of the
        same clip already cached one, that one is used and ``new_sess`` is returned
        for closing. The evicted LRU victim (if any) is likewise returned to be
        closed by the caller — outside the lock, since close() is slow.

        Pinned sessions are never chosen as the victim: an in-flight propagate
        stream/export would have its VideoCapture closed under it. If everything
        is pinned the cap is temporarily exceeded (bounded by in-flight requests).
        """
        with session_lock:
            if video_id in sessions:  # racing insert won — use theirs, close ours
                sessions.move_to_end(video_id)
                return sessions[video_id], new_sess
            sessions[video_id] = new_sess
            sessions.move_to_end(video_id)
            victim = None
            if len(sessions) > MAX_ACTIVE_SESSIONS:
                for lru_vid in sessions:  # OrderedDict iterates LRU-first
                    if lru_vid != video_id and pins.get(lru_vid, 0) == 0:
                        victim = sessions.pop(lru_vid)
                        break
            return new_sess, victim

    def _decode_session(video_id: int, name: str, stride: int) -> VideoSession:
        """Materialize + decode one clip. Slow (decode/probe) — always call
        outside session_lock, and via run_in_threadpool from async handlers."""
        local_path = store.materialize_video(name)
        return VideoSession(video_id, name, str(local_path), stride, tracker_factory)

    def _cache_put_and_close(video_id: int, new_sess: VideoSession) -> VideoSession:
        """_cache_put, closing whatever it hands back (racing dup or LRU victim).

        Blocking (close() is slow) — call via run_in_threadpool from async handlers.
        """
        sess, to_close = _cache_put(video_id, new_sess)
        if to_close is not None:
            to_close.close()
            gc.collect()  # break PyTorch ref cycles before the next session allocates
        return sess

    def _rebuild_session(video_id: int) -> VideoSession:
        """Rebuild a session lost to LRU eviction or a server restart, from the DB
        row /api/open already wrote — same construction logic as there, just keyed
        by video_id instead of (name, stride). 404 only if the video truly doesn't
        exist. Blocking (decode) — call via run_in_threadpool from async handlers.
        """
        with _conn() as c:
            row = db.video_row(c, video_id)
        if row is None:
            raise HTTPException(404, f"video not found: {video_id}")
        sess = _decode_session(video_id, row["name"], row["stride"])
        return _cache_put_and_close(video_id, sess)

    def _session(video_id: int) -> VideoSession:
        sess = _cache_get(video_id)
        if sess is None:  # evicted or lost to a restart — rebuild, don't 409
            sess = _rebuild_session(video_id)
        return sess

    def _pin(video_id: int) -> VideoSession | None:
        """Atomic lookup + pin. The session cannot be evicted until _unpin."""
        with session_lock:
            sess = sessions.get(video_id)
            if sess is not None:
                sessions.move_to_end(video_id)
                pins[video_id] = pins.get(video_id, 0) + 1
            return sess

    def _unpin(video_id: int) -> None:
        with session_lock:
            n = pins.get(video_id, 0) - 1
            if n > 0:
                pins[video_id] = n
            else:
                pins.pop(video_id, None)

    def _pin_or_rebuild(video_id: int) -> VideoSession:
        """_pin(), rebuilding the session from the DB first if it was evicted or
        lost to a restart. Blocking on the rebuild path (decode) — call via
        run_in_threadpool from async handlers.
        """
        sess = _pin(video_id)
        if sess is not None:
            return sess
        _rebuild_session(video_id)
        sess = _pin(video_id)
        if sess is None:
            # ponytail: rebuild-then-pin is two lock acquisitions, not one atomic
            # op, so a fresh rebuild could theoretically be evicted again before
            # this pin lands. Needs a concurrent evict *and* re-evict in that
            # razor-thin window; a retry loop would close it fully if it ever fires.
            raise HTTPException(409, "video not open — call /api/open first")
        return sess

    @contextmanager
    def _pinned_session(video_id: int):
        """_session(), but pinned for the duration of the block — for handlers
        whose work outlives a quick lookup (GPU calls, exports)."""
        sess = _pin_or_rebuild(video_id)
        try:
            yield sess
        finally:
            _unpin(video_id)

    @asynccontextmanager
    async def _pinned_session_async(video_id: int):
        """_pinned_session, off the event loop — the rebuild path decodes, so
        async handlers (segment, find_all) must not call it inline."""
        sess = await run_in_threadpool(_pin_or_rebuild, video_id)
        try:
            yield sess
        finally:
            _unpin(video_id)

    def _load_owl():
        if detectors["owl"] is None:
            try:
                from kumo_track.models.owlv2 import OWLv2Detector
            except ImportError as exc:
                raise RuntimeError(f"OWLv2 not available — install transformers: {exc}") from exc
            detectors["owl"] = OWLv2Detector(threshold=0.05)
        return detectors["owl"]

    def _load_sam3det():
        if detectors["sam3"] is None:
            if remote:
                from kumo_track.annotate.remote import RemoteSAM3Detector

                detectors["sam3"] = RemoteSAM3Detector(client=remote_client, box_mode="aabb")
            else:
                try:
                    from kumo_track.models.sam3 import SAM3Detector
                except ImportError as exc:
                    raise RuntimeError(f"SAM3Detector not available: {exc}") from exc
                detectors["sam3"] = SAM3Detector(threshold=0.05, box_mode="aabb")
        return detectors["sam3"]

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if load_on_start and not remote:
            # Local mode only: preload the in-process SAM 3 video model. In remote
            # mode the service loads lazily on first inference (warmed per request).
            from kumo_track.annotate.tracker import load_model

            await run_in_threadpool(load_model)
        if load_on_start:
            where = f"remote {os.environ.get('SAM3_URL')}" if remote else "local"
            # Host part only: DATABASE_URL carries the DB password and this line lands in Log Analytics.
            print(f"[annotate] ready ({where}) · VIDEO_DIR={video_dir} · DB={db_url.rsplit('@', 1)[-1]}")
            # The session cache and gpu_lock are per-process: kumo-track must run as
            # exactly ONE replica (see README "Deployment"). Emit the assumption so a
            # stray scale-out is visible in the logs.
            print("[annotate] single-replica mode — do not scale out (per-process session cache + GPU lock)")
        yield
        # Shutdown: release every open clip and the DB pool.
        for sess in sessions.values():
            sess.close()
        database.close()

    # The auth gate runs on every request; handlers needing the value re-declare
    # Depends(current_user) (FastAPI caches it per request).
    app = FastAPI(
        title="SAM3 Video Annotation", lifespan=lifespan,
        dependencies=[Depends(current_user)],
    )
    if STATIC.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

    # --- pages -----------------------------------------------------------------

    # Gate-exempt (see HEALTH_PATHS): orchestrator probes reach these with no
    # X-Kumo-User even when REQUIRE_AUTH=on. Liveness only — no DB/GPU touch.
    @app.get("/api/health")
    @app.get("/healthz")
    def health():
        return {"status": "ok"}

    @app.get("/api/sam3/health")
    def sam3_health():
        """Readiness of the SAM3 inference backing, for the frontend's loading pill.

        Remote mode: one quick probe of the service's /health — a scaled-to-zero
        replica answers with a connection error / 503 until it has spun up
        (~2 min), so anything but a 200 means "loading". The short timeout keeps
        this a snappy poll target (sync def → threadpool, never blocks the loop).
        Local mode: the model loads during startup, so once we serve requests it
        is ready.
        """
        if not remote:
            return {"status": "ready"}
        try:
            r = httpx.get(f"{remote_client.base}/health", timeout=5)
            return {"status": "ready" if r.status_code == 200 else "loading"}
        except httpx.HTTPError:
            return {"status": "loading"}

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/config")
    def get_config():
        return config

    @app.get("/api/videos")
    def list_videos():
        # location is the backend's own store (a dir path locally, a container name
        # on blob) — meaningful under either backend, unlike the raw video_dir.
        return {"videos": store.list_videos(), "location": store.location}

    @app.post("/api/upload")
    async def upload_video(file: UploadFile = File(...)):
        name = Path(file.filename or "").name  # basename only — no path traversal
        if Path(name).suffix.lower() not in VIDEO_EXTS:
            raise HTTPException(400, f"unsupported video type: {name or '(none)'}")
        # Reject a name that already exists: video_id is keyed on (name, stride), so
        # overwriting a clip with different content would silently rebind existing
        # annotations to the new footage. Force a distinct name instead.
        if await run_in_threadpool(store.video_exists, name):
            raise HTTPException(409, f"a video named {name!r} already exists — rename it or delete the existing clip first")
        await run_in_threadpool(store.save_video, name, file.file)
        await file.close()
        return {"name": name}

    def _open_response(vid: int, sess: VideoSession, ann: dict, stride: int) -> dict:
        return {
            "video_id": vid, "n_frames": sess.n_frames, "width": sess.width,
            "height": sess.height, "fps": sess.fps, "source_indices": sess.source_indices,
            "annotations": ann, "truncated_at": sess.truncated_at, "stride": stride,
        }

    @app.post("/api/open")
    async def open_video(req: OpenReq):
        if not store.video_exists(req.name):
            raise HTTPException(404, f"video not found: {req.name}")

        # Fast path: this clip is already open (same or another user) — share it.
        # video_id is fixed by (name, stride), so resolve it and hit the cache
        # before doing any decode work.
        with _conn() as c:
            row = db.video_by_name_stride(c, req.name, req.stride)
        if row is not None:
            sess = _cache_get(row["id"])
            if sess is not None:
                with _conn() as c:
                    ann = db.get_annotations(c, row["id"])
                return _open_response(row["id"], sess, ann, req.stride)

        # Build the session OUTSIDE any lock (decode/probe is slow) so a slow open
        # never blocks other clips' cache lookups or in-flight inference.
        sess = await run_in_threadpool(_decode_session, 0, req.name, req.stride)
        with _conn() as c:
            vid = db.upsert_video(
                c, req.name, sess.fs.path, sess.width, sess.height, sess.fps,
                req.stride, None, sess.n_frames, sess.source_indices,
            )
            sess.video_id = vid
            ann = db.get_annotations(c, vid)

        sess = await run_in_threadpool(_cache_put_and_close, vid, sess)
        return _open_response(vid, sess, ann, req.stride)

    @app.get("/api/frame/{video_id}/{idx}.jpg")
    def get_frame(video_id: int, idx: int, q: int = 80):
        sess = _session(video_id)
        if idx < 0 or idx >= sess.n_frames:
            raise HTTPException(404, "frame out of range")
        return Response(
            content=sess.jpeg(idx, quality=q),
            media_type="image/jpeg",
            # Frames are immutable for a given (video_id, idx): video_id differs per
            # (name, stride), so the browser can cache aggressively.
            headers={"Cache-Control": "public, max-age=31536000, immutable"},
        )

    # --- objects ---------------------------------------------------------------

    @app.post("/api/objects")
    def create_object(req: ObjectReq, user: str = Depends(current_user)):
        # created_by is the trusted header user, never the request body.
        with _conn() as c:
            obj_id = db.create_object(c, req.video_id, req.label, user)
        # SQLite reuses rowids of deleted rows, and a deleted object's id stays
        # tombstoned in the tracker (forget_object) — clear it or the new object
        # silently never tracks.
        sess = _cache_get(req.video_id)
        if sess is not None:
            sess.unforget_object(obj_id)
        return {"obj_id": obj_id}

    @app.patch("/api/objects/{obj_id}")
    def patch_object(obj_id: int, req: ObjectPatchReq):
        with _conn() as c:
            sess = _cache_get(db.object_video_id(c, obj_id))
            if req.label is not None:
                db.rename_object(c, obj_id, req.label)
            if req.static is not None:
                db.set_object_static(c, obj_id, req.static)
                # A static object never tracks; drop it from the live tracker too
                # (and let it back in when made trackable again).
                if sess is not None:
                    if req.static:
                        sess.forget_object(obj_id)
                    else:
                        sess.unforget_object(obj_id)
            # hidden + color are display-only: they never touch the tracker.
            if req.hidden is not None:
                db.set_object_hidden(c, obj_id, req.hidden)
            if req.color is not None:
                db.set_object_color(c, obj_id, req.color)
        return {"ok": True}

    @app.delete("/api/objects/{obj_id}")
    def delete_object(obj_id: int):
        with _conn() as c:
            sess = _cache_get(db.object_video_id(c, obj_id))
            if sess is not None:
                sess.forget_object(obj_id)
            db.delete_object(c, obj_id)
        return {"ok": True}

    # --- segment / propagate ---------------------------------------------------

    @app.post("/api/segment")
    async def segment(req: SegmentReq):
        async with _pinned_session_async(req.video_id) as sess:
            try:
                async with gpu_lock:
                    fit = await run_in_threadpool(
                        sess.segment, req.frame_idx, req.obj_id, req.points, req.labels, req.box
                    )
            except RuntimeError as e:  # includes CUDA OOM
                raise HTTPException(507, f"inference failed: {e}")
            polygon = fit["polygon"] if (fit and store_polygon) else None
            with _conn() as c:
                db.upsert_annotation(
                    c, req.video_id, req.obj_id, req.frame_idx, sess.source_frame(req.frame_idx),
                    fit and fit["corners"], polygon, fit and fit["score"], "seed",
                    mask_rle=fit and fit["mask_rle"],
                )
        return {"obj_id": req.obj_id, "frame_idx": req.frame_idx,
                "corners": fit and fit["corners"], "polygon": polygon}

    @app.post("/api/propagate")
    async def propagate(req: PropagateReq, request: Request):
        # Pinned manually (not via _pinned_session_async): the pin must outlive
        # this handler and cover the whole NDJSON stream — a long propagation must
        # not be evicted by a concurrent open. stream()'s finally releases it.
        sess = await run_in_threadpool(_pin_or_rebuild, req.video_id)
        try:
            reverse = req.reverse if req.reverse is not None else (req.direction == "rev")
            max_steps = req.n_frames if req.n_frames is not None else req.max_frames

            fallback_seed: dict | None = None
            with _conn() as c:
                seeds0 = db.seed_annotations(c, req.video_id, req.start_frame_idx)
                # Static objects never reach SAM3: the box from their annotated frame
                # nearest the start is copied onto every tracked frame instead.
                statics = db.static_annotations(c, req.video_id, req.start_frame_idx)
                # The selected object always tracks: if it has no box on the start
                # frame, copy its box from the nearest annotated frame here (persisted
                # as origin='seed'), so SAM3 can take over from any frame.
                if req.active_obj is not None and req.active_obj not in seeds0 \
                        and req.active_obj not in statics:
                    fallback_seed = db.copy_nearest_annotation(
                        c, req.active_obj, req.start_frame_idx, req.video_id
                    )
                    if fallback_seed:
                        seeds0[req.active_obj] = fallback_seed
            if not seeds0 and not statics:
                raise HTTPException(
                    409,
                    "no box on this frame to track — seed/select an object and segment "
                    "it on this frame first",
                )

            if req.full_clip:
                passes = [(req.start_frame_idx, False, None)]
                if req.start_frame_idx > 0:
                    passes.insert(0, (req.start_frame_idx, True, None))
            else:
                passes = [(req.start_frame_idx, reverse, max_steps)]
        except BaseException:
            _unpin(req.video_id)
            raise

        def _persist(c, manual, frame_idx, boxes) -> str:
            out: dict = {"frame_idx": frame_idx, "boxes": {}}
            for oid, fit in {**boxes, **statics}.items():
                key = (oid, frame_idx)
                if key in manual:  # manual precedence — never overwrite a hand edit
                    out["boxes"][oid] = manual[key]
                    continue
                db.upsert_annotation(
                    c, req.video_id, oid, frame_idx, sess.source_frame(frame_idx),
                    fit and fit["corners"],
                    (fit["polygon"] if (fit and store_polygon) else None),
                    fit and fit["score"], "propagated",
                    mask_rle=fit and fit.get("mask_rle"),
                )
                out["boxes"][oid] = fit
            return json.dumps(out) + "\n"

        def _copy_range(start, rev, steps):
            """Frames a copy-only pass visits (start frame excluded, like SAM3)."""
            if rev:
                lo = 0 if steps is None else max(0, start - steps)
                return range(start - 1, lo - 1, -1)
            hi = sess.n_frames - 1 if steps is None else min(sess.n_frames - 1, start + steps)
            return range(start + 1, hi + 1)

        def _safe_next(gen):
            try:
                return next(gen)
            except StopIteration:
                return (None, None)

        async def stream():
            # Producer/consumer split: gpu_lock is held only while frames are
            # PRODUCED (inference-paced) — never across client-paced yields, so
            # one slow browser can't pin the lock and block every other user's
            # inference. Records are small JSON; the queue's worst-case growth
            # is one clip's worth of boxes.
            queue: asyncio.Queue = asyncio.Queue()
            done = object()

            async def produce():
                try:
                    async with gpu_lock:
                        with _conn() as c:
                            manual = db.manual_annotations(c, req.video_id)
                            for start, rev, steps in passes:
                                seeds = db.seed_annotations(c, req.video_id, start)
                                if seeds:
                                    gen = sess.propagate(start, rev, steps, seeds)
                                    while True:
                                        if await request.is_disconnected():
                                            return
                                        fi, boxes = await run_in_threadpool(_safe_next, gen)
                                        if fi is None:
                                            break
                                        queue.put_nowait(_persist(c, manual, fi, boxes))
                                elif statics:
                                    # Only static objects: no SAM3 — just walk the
                                    # requested range copying their boxes onto each frame.
                                    for fi in _copy_range(start, rev, steps):
                                        if await request.is_disconnected():
                                            return
                                        queue.put_nowait(_persist(c, manual, fi, {}))
                except Exception as exc:
                    queue.put_nowait(json.dumps({"error": f"{type(exc).__name__}: {exc}"}) + "\n")
                finally:
                    queue.put_nowait(done)

            try:
                if fallback_seed:
                    # Show the copied seed on the start frame (which is never re-emitted).
                    yield json.dumps({
                        "frame_idx": req.start_frame_idx,
                        "boxes": {req.active_obj: fallback_seed},
                    }) + "\n"
                producer = asyncio.create_task(produce())
                try:
                    while True:
                        item = await queue.get()
                        if item is done:
                            break
                        yield item
                finally:
                    producer.cancel()
                    try:
                        await producer
                    except asyncio.CancelledError:
                        pass
            finally:
                _unpin(req.video_id)  # matches the _pin taken in the handler

        return StreamingResponse(stream(), media_type="application/x-ndjson")

    # --- annotations -----------------------------------------------------------

    @app.get("/api/annotations")
    def annotations(video_id: int):
        with _conn() as c:
            return db.get_annotations(c, video_id)

    @app.get("/api/annotation-status")
    def annotation_status():
        """Aggregate labeled-status across all clips (for the project hub).

        Behind the same current_user gate as every route. Clips with no objects
        are absent from the map → the caller treats absence as unlabeled.
        """
        with _conn() as c:
            return {"videos": db.annotation_status(c)}

    @app.put("/api/annotations")
    def manual_annotation(req: ManualReq):
        """Hand-edited box/polygon → stored with origin='manual' (propagation-proof).

        An outline edit (polygon sent) is rasterised into the source-of-truth
        ``mask_rle``; the polygon itself is kept only in polygon mode. Corners
        alone are a pure bounding box and stay exactly that — coordinates only,
        never a mask or polygon.
        """
        sess = _session(req.video_id)
        corners = req.corners
        polygon = req.polygon
        # Vertex edit sends polygon only → derive the OBB from it. A box transform
        # of an outline-backed annotation sends both → trust both as-is.
        if corners is None and polygon is not None:
            corners = _obb_from_polygon(polygon) or corners
        if corners is None:
            raise HTTPException(400, "manual annotation needs corners or polygon")
        if polygon is not None:
            ds_h, ds_w = mask_target_size(sess.height, sess.width)
            binary = rle.rasterize_polygon(polygon, ds_h, ds_w, sess.height, sess.width)
            mask_rle = rle.encode(binary) if binary.any() else None
        else:
            mask_rle = None
        stored_polygon = polygon if store_polygon else None
        with _conn() as c:
            db.upsert_annotation(
                c, req.video_id, req.object_id, req.frame_idx,
                sess.source_frame(req.frame_idx), corners, stored_polygon, None, "manual",
                mask_rle=mask_rle,
            )
        # The polygon is echoed even in brush mode (where it isn't persisted) so the
        # client can keep vertex-editing it within the session.
        return {"object_id": req.object_id, "frame_idx": req.frame_idx,
                "corners": corners, "polygon": polygon,
                "has_mask": mask_rle is not None}

    @app.put("/api/mask")
    def manual_mask(req: MaskReq):
        """Brush-painted mask → source-of-truth ``mask_rle`` + derived OBB, origin='manual'.

        The browser paints at display resolution; the mask is downscaled to the
        canonical mask resolution and run-length encoded. An empty mask (erased to
        nothing) just deletes the annotation. The outline polygon is kept only in
        polygon mode; brush mode stores none.
        """
        import numpy as np

        sess = _session(req.video_id)
        try:
            raw = base64.b64decode(req.mask_png)
            arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
        except (ValueError, cv2.error):
            arr = None
        if arr is None:
            raise HTTPException(400, "could not decode mask PNG")
        # Painted pixels are opaque; erased ones transparent — use alpha when present.
        if arr.ndim == 3 and arr.shape[2] == 4:
            gray = arr[:, :, 3]
        elif arr.ndim == 3:
            gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        else:
            gray = arr
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        if not binary.any():
            with _conn() as c:
                db.delete_annotation(c, req.object_id, req.frame_idx)
            return {"object_id": req.object_id, "frame_idx": req.frame_idx,
                    "corners": None, "polygon": None}
        ds_h, ds_w = mask_target_size(sess.height, sess.width)
        binary = cv2.resize(binary, (ds_w, ds_h), interpolation=cv2.INTER_NEAREST)
        scale_x, scale_y = sess.width / ds_w, sess.height / ds_h
        corners = _mask_to_corners(binary, scale_x, scale_y)
        if corners is None:
            raise HTTPException(400, "painted mask is too small")
        mask_rle = rle.encode(binary)
        polygon = _mask_polygon(binary, scale_x, scale_y) if store_polygon else None
        with _conn() as c:
            db.upsert_annotation(
                c, req.video_id, req.object_id, req.frame_idx,
                sess.source_frame(req.frame_idx), corners, polygon, None, "manual",
                mask_rle=mask_rle,
            )
        return {"object_id": req.object_id, "frame_idx": req.frame_idx,
                "corners": corners, "polygon": polygon, "has_mask": True}

    @app.get("/api/mask")
    def get_mask(video_id: int, object_id: int, frame_idx: int):
        """One annotation's mask as an RGBA PNG (white, alpha=mask) for the brush overlay.

        Decoded from the stored ``mask_rle``; falls back to rasterising an old row's
        polygon. The PNG is at the downscaled mask resolution — the canvas scales it.
        """
        import numpy as np

        sess = _session(video_id)
        with _conn() as c:
            row = db.get_annotation_mask(c, video_id, object_id, frame_idx)
        if row is None:
            raise HTTPException(404, "annotation not found")
        if row["mask_rle"]:
            binary = rle.decode(row["mask_rle"])
        elif row["polygon"]:
            ds_h, ds_w = mask_target_size(sess.height, sess.width)
            binary = rle.rasterize_polygon(row["polygon"], ds_h, ds_w, sess.height, sess.width)
        else:
            raise HTTPException(404, "no mask for this annotation")
        h, w = binary.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:, :, :3] = 255
        rgba[:, :, 3] = (binary > 0).astype(np.uint8) * 255
        ok, buf = cv2.imencode(".png", rgba)
        if not ok:
            raise HTTPException(500, "mask encode failed")
        # Masks are mutable for a given (video_id, object_id, frame_idx): re-segmenting
        # or brush-editing rewrites them. The URL is stable, so without this the browser
        # serves a stale PNG from its image cache and the overlay never updates.
        return Response(
            content=buf.tobytes(),
            media_type="image/png",
            headers={"Cache-Control": "no-store"},
        )

    @app.delete("/api/annotations")
    def delete_annotation(req: DeleteReq):
        with _conn() as c:
            db.delete_annotation(c, req.object_id, req.frame_idx)
        return {"ok": True}

    # --- find all --------------------------------------------------------------

    @app.post("/api/find_all")
    async def find_all(req: FindAllReq, user: str = Depends(current_user)):
        async with _pinned_session_async(req.video_id) as sess:
            return await _find_all_impl(req, user, sess)

    async def _find_all_impl(req: FindAllReq, user: str, sess: VideoSession):
        async with gpu_lock:
            loader = _load_sam3det if req.detector != "owlv2" else _load_owl
            try:
                det_model = await run_in_threadpool(loader)
            except RuntimeError as exc:
                raise HTTPException(503, str(exc))

            pil_image = PILImage.fromarray(sess.fs.get_frame(req.frame_idx))
            query = (req.query or req.label).strip() or req.label
            saved_prompts, saved_thr = det_model.text_prompts, det_model.threshold
            det_model.text_prompts = [query]
            det_model.threshold = req.threshold
            try:
                detections = await run_in_threadpool(det_model.detect_image, pil_image)
            finally:
                det_model.text_prompts = saved_prompts
                det_model.threshold = saved_thr

            created: list[dict] = []
            with _conn() as c:
                for det in detections:
                    obj_id = db.create_object(c, req.video_id, req.label, user)
                    sess.unforget_object(obj_id)  # rowid may be a recycled, tombstoned id
                    xs = [p[0] for p in det.corners]
                    ys = [p[1] for p in det.corners]
                    box = [min(xs), min(ys), max(xs), max(ys)]
                    try:
                        fit = await run_in_threadpool(
                            sess.segment, req.frame_idx, obj_id, None, None, box
                        )
                    except Exception:
                        fit = None
                    corners = fit["corners"] if fit else det.corners
                    polygon = fit["polygon"] if (fit and store_polygon) else None
                    db.upsert_annotation(
                        c, req.video_id, obj_id, req.frame_idx,
                        sess.source_frame(req.frame_idx),
                        corners, polygon, det.score, "seed",
                        mask_rle=fit["mask_rle"] if fit else None,
                    )
                    created.append({
                        "id": obj_id, "label": req.label, "corners": corners,
                        "polygon": polygon, "score": det.score, "seeded": bool(fit),
                    })
        return {"created": created, "frame_idx": req.frame_idx}

    # --- export ----------------------------------------------------------------

    @app.post("/api/export")
    def export(req: ExportReq):
        with _pinned_session(req.video_id) as sess:
            return _export_impl(req, sess)

    def _export_impl(req: ExportReq, sess: VideoSession):
        writer = store.export_writer(req.out_name)
        with _conn() as c:
            ann = db.get_annotations(c, req.video_id, include_mask=True)

        def _export_mask(fit) -> dict | None:
            """Full-res COCO RLE matching the exported image, or None."""
            if fit.get("mask_rle"):
                binary = rle.decode(fit["mask_rle"])
                if binary.shape != (sess.height, sess.width):
                    binary = cv2.resize(binary, (sess.width, sess.height),
                                        interpolation=cv2.INTER_NEAREST)
                return rle.encode(binary)
            if fit.get("polygon"):  # old rows without a stored mask
                return rle.encode(rle.rasterize_polygon(
                    fit["polygon"], sess.height, sess.width, sess.height, sess.width))
            return None

        obj_label = {o["id"]: o["label"] for o in ann["objects"]}
        stem = sess.name.rsplit(".", 1)[0]
        labels: dict[str, list[dict]] = {}
        n_boxes = 0
        for frame_idx in sorted(int(k) for k in ann["frames"]):
            perobj = ann["frames"][frame_idx]
            rows = []
            for oid, fit in perobj.items():
                if not (fit and fit.get("corners")):
                    continue
                row = {"corners": fit["corners"], "label": obj_label.get(oid, "object")}
                mask = _export_mask(fit)
                if mask is not None:
                    row["mask"] = mask
                rows.append(row)
            if not rows:
                continue
            fname = f"{stem}-frame_{sess.source_frame(frame_idx):06d}.jpg"
            if req.include_images:
                ok, buf = cv2.imencode(".jpg", sess.frame_bgr(frame_idx),
                                       [cv2.IMWRITE_JPEG_QUALITY, req.quality])
                if not ok:
                    raise HTTPException(500, "JPEG encode failed")
                writer.write_bytes(f"images/{fname}", buf.tobytes())
            labels[fname] = rows
            n_boxes += len(rows)
        writer.write_bytes("labels.json", json.dumps(labels, indent=2).encode())
        return {"dir": writer.location, "n_images": len(labels), "n_boxes": n_boxes}

    return app
