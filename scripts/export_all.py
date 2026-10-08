#!/usr/bin/env python
# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx>=0.27", "numpy>=2.0", "opencv-python-headless>=4.8"]
# ///
"""Export every labeled clip into a Mask2Former-ready instance-segmentation dataset.

There is no "export all" endpoint — export is per clip — so this walks the API:
``/api/videos`` → ``/api/open`` (per clip) → ``/api/annotations`` → per annotated
frame pull the JPEG (``/api/frame``) and each mask (``/api/mask``), and write a
dataset laid out exactly how ``transformers.Mask2FormerImageProcessor`` wants it:

    <out>/
      images/<clip>__<srcframe>.jpg        RGB frame (as served, JPEG q95)
      annotations/<clip>__<srcframe>.png   uint8 instance map: 0=bg, 1..N=instances
      metadata.jsonl                       one line per image (see below)
      id2label.json                        {"0": "forklift", "1": "pallet", ...}
      README.md                            copy-paste loader + training snippet

metadata.jsonl line:
    {"file_name": "images/x__000042.jpg",
     "annotation": "annotations/x__000042.png",
     "instances": {"1": 0, "2": 1}}        instance_id -> category_id (id2label key)

Auth: kumo-track's ingress is IP-locked to the proxy, so from a laptop go through
the public proxy (track.<domain>) with your browser session cookie — log in, copy
the `kv_session` cookie value from devtools, and pass it. (Direct --secret/--user
is only for local dev or a host on the ingress allow-list; the proxy strips any
client x-kumo-* headers, so they do nothing there.)

    uv run scripts/export_all.py --out ~/data/kumo \
        --base https://kumo-track.example.com --cookie "$KUMO_COOKIE"

    uv run scripts/export_all.py --self-check      # no server; exercises the mask logic
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_RETRY_STATUS = {502, 503, 504}  # single-replica kumo-track blips under load — transient


# --- pure mask helpers (self-checkable, no network) ----------------------------

def rasterize_corners(corners: list[list[float]], h: int, w: int) -> np.ndarray:
    """A pure bounding box (no stored mask) → filled polygon at full resolution."""
    pts = np.asarray(corners, dtype=np.float32).round().astype(np.int32)
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [pts], 1)
    return m


def mask_from_png(png_bytes: bytes, h: int, w: int) -> np.ndarray:
    """Decode the RGBA mask PNG from /api/mask (downscaled) → full-res binary."""
    arr = cv2.imdecode(np.frombuffer(png_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
    binary = (arr[:, :, 3] > 0).astype(np.uint8) if arr.ndim == 3 else (arr > 0).astype(np.uint8)
    if binary.shape != (h, w):
        binary = cv2.resize(binary, (w, h), interpolation=cv2.INTER_NEAREST)
    return binary


# --- API client ----------------------------------------------------------------

class Track:
    def __init__(self, base: str, secret: str | None, user: str | None, cookie: str | None):
        import httpx

        # Two ways in. Via the public proxy (track.<domain>) the container's ingress
        # is IP-locked to the proxy, so you can't reach it directly — you send your
        # browser session cookie and the proxy injects the real X-Kumo-User (it also
        # STRIPS any x-kumo-* the client sends, so secret/user are pointless there).
        # Direct to the container (local dev, or an allow-listed host) you carry the
        # shared secret + user headers yourself.
        headers, cookies = {}, {}
        if cookie:
            cookies["kv_session"] = cookie
        else:
            if secret:
                headers["X-Kumo-Proxy-Auth"] = secret
            if user:
                headers["X-Kumo-User"] = user
        # opening a clip decodes video server-side — can be slow; be patient.
        self._c = httpx.Client(base_url=base.rstrip("/"), headers=headers,
                               cookies=cookies, timeout=300.0)
        self._open_args: tuple | None = None  # last (name, stride) — to reopen after eviction

    def _send(self, method: str, url: str, **kw):
        """Request with backoff on transient 502/503/504 and connection errors."""
        import httpx

        for attempt in range(6):  # ~1+2+4+8+16s of waiting, then give up
            try:
                r = self._c.request(method, url, **kw)
                if r.status_code not in _RETRY_STATUS:
                    return r.raise_for_status()
            except httpx.TransportError:
                if attempt == 5:
                    raise
            if attempt < 5:
                print(f"    … {url} unavailable, retrying", file=sys.stderr)
                time.sleep(2 ** attempt)
        return r.raise_for_status()  # exhausted → raise the last 5xx

    def videos(self) -> list[str]:
        return self._send("GET", "/api/videos").json()["videos"]

    def labeled_names(self) -> set[str]:
        return set(self._send("GET", "/api/annotation-status").json()["videos"])

    def open(self, name: str, stride: int | None) -> dict:
        body = {"name": name} if stride is None else {"name": name, "stride": stride}
        info = self._send("POST", "/api/open", json=body).json()
        self._open_args = (name, stride)
        return info

    def _session_get(self, url: str, **kw):
        """GET a session-dependent endpoint (frame/mask). A 503 blip can restart the
        single replica and wipe its in-memory open-clip cache, after which these 409
        with 'video not open'. Reopen the clip (same video_id) and retry."""
        import httpx

        for attempt in range(4):
            try:
                return self._send("GET", url, **kw)
            except httpx.HTTPStatusError as e:
                if e.response.status_code != 409 or attempt == 3 or not self._open_args:
                    raise
                print(f"    … session evicted, reopening {self._open_args[0]}", file=sys.stderr)
                self.open(*self._open_args)

    def annotations(self, video_id: int) -> dict:
        return self._send("GET", "/api/annotations", params={"video_id": video_id}).json()

    def frame_jpg(self, video_id: int, idx: int, quality: int) -> bytes:
        return self._session_get(f"/api/frame/{video_id}/{idx}.jpg", params={"q": quality}).content

    def mask_png(self, video_id: int, object_id: int, frame_idx: int) -> bytes:
        return self._session_get("/api/mask", params={"video_id": video_id,
                                 "object_id": object_id, "frame_idx": frame_idx}).content


# --- export --------------------------------------------------------------------

def export_all(t: Track, out: Path, stride: int | None, quality: int) -> None:
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "annotations").mkdir(parents=True, exist_ok=True)
    meta_path, id2label_path = out / "metadata.jsonl", out / "id2label.json"

    # Resume: skip frames already written, and reload the label→id map so class ids
    # stay identical to the earlier run's metadata. (Discovery order is deterministic,
    # so a fresh map would agree anyway — but reloading makes it not depend on that.)
    done: set[str] = set()
    label2id: dict[str, int] = {}
    if meta_path.exists():
        done = {json.loads(l)["file_name"] for l in meta_path.read_text().splitlines() if l.strip()}
        if id2label_path.exists():
            label2id = {v: int(k) for k, v in json.loads(id2label_path.read_text()).items()}
        elif done:  # metadata but no label map (e.g. a pre-resume run) → ids unrecoverable
            sys.exit(f"{meta_path} exists but {id2label_path} is missing — can't resume "
                     f"safely (class ids would collide). Use a fresh --out.")
        print(f"resuming: {len(done)} frames already exported")
    n_persisted = len(label2id)

    # Only labeled clips have frames — skip decoding the rest (much less load on the
    # single replica, which is what tips it into 503s).
    labeled = t.labeled_names()
    names = [n for n in t.videos() if n in labeled]
    print(f"{len(names)} labeled clips")

    n_images = n_instances = 0
    with meta_path.open("a") as meta:
        for name in names:
            try:
                info = t.open(name, stride)
            except Exception as e:  # a clip that won't decode shouldn't sink the run
                print(f"  ! skip {name}: {e}", file=sys.stderr)
                continue
            vid, h, w = info["video_id"], info["height"], info["width"]
            src = info["source_indices"]  # session frame idx -> real video frame number
            ann = t.annotations(vid)
            obj_label = {o["id"]: o["label"] for o in ann["objects"]}
            stem = name.rsplit(".", 1)[0]
            n_frames_clip = 0

            for fidx_s, perobj in ann["frames"].items():
                fidx = int(fidx_s)
                key = f"{stem}__{src[fidx]:06d}"
                if f"images/{key}.jpg" in done:  # resume: already have this frame
                    continue
                # ponytail: uint8 -> max 255 instances per frame; tracked clips have a
                # handful, so this never binds. Bump to uint16 if that ever changes.
                inst_map = np.zeros((h, w), np.uint8)
                inst2class: dict[str, int] = {}
                k = 0
                for oid_s, fit in perobj.items():
                    if not (fit and fit.get("corners")):
                        continue
                    oid = int(oid_s)
                    if fit.get("has_mask"):
                        binary = mask_from_png(t.mask_png(vid, oid, fidx), h, w)
                    else:
                        binary = rasterize_corners(fit["corners"], h, w)
                    if not binary.any():
                        continue
                    k += 1
                    inst_map[binary > 0] = k  # ponytail: last object wins on overlap
                    label = obj_label.get(oid, "object")
                    inst2class[str(k)] = label2id.setdefault(label, len(label2id))
                if k == 0:
                    continue

                (out / "images" / f"{key}.jpg").write_bytes(t.frame_jpg(vid, fidx, quality))
                cv2.imwrite(str(out / "annotations" / f"{key}.png"), inst_map)
                # Invariant for resume: every class id in metadata.jsonl must already
                # be in id2label.json. So persist a newly-seen class BEFORE the line
                # that references it, then flush the line — consistent at any crash.
                if len(label2id) != n_persisted:
                    _write_id2label(id2label_path, label2id)
                    n_persisted = len(label2id)
                meta.write(json.dumps({
                    "file_name": f"images/{key}.jpg",
                    "annotation": f"annotations/{key}.png",
                    "instances": inst2class,
                }) + "\n")
                meta.flush()
                n_images += 1
                n_instances += k
                n_frames_clip += 1
            print(f"  {name}: {n_frames_clip} frames")

    _write_id2label(id2label_path, label2id)
    (out / "README.md").write_text(_README)
    print(f"\n{n_images} new images, {n_instances} instances, {len(label2id)} classes → {out}")


def _write_id2label(path: Path, label2id: dict[str, int]) -> None:
    path.write_text(json.dumps({str(i): lbl for lbl, i in label2id.items()}, indent=2))


_README = '''\
# Kumo instance-segmentation export (Mask2Former-ready)

```python
import json, os, numpy as np
from PIL import Image
from transformers import Mask2FormerImageProcessor

root = "."
id2label = json.load(open(os.path.join(root, "id2label.json")))
# background is instance id 0; ignore_index=0 tells the processor to drop it.
processor = Mask2FormerImageProcessor(ignore_index=0, do_reduce_labels=False)

def load(line):
    r = json.loads(line)
    image = Image.open(os.path.join(root, r["file_name"])).convert("RGB")
    inst  = np.array(Image.open(os.path.join(root, r["annotation"])))   # HxW, 0=bg, 1..N
    inst2class = {int(k): v for k, v in r["instances"].items()}         # instance -> class
    return processor(images=[image], segmentation_maps=[inst],
                     instance_id_to_semantic_id=inst2class, return_tensors="pt")

# model: Mask2FormerForUniversalSegmentation.from_pretrained(
#   "facebook/mask2former-swin-base-coco-instance",
#   id2label=id2label, ignore_mismatched_sizes=True)
for line in open(os.path.join(root, "metadata.jsonl")):
    batch = load(line)   # pixel_values, pixel_mask, mask_labels, class_labels
```
'''


def _self_check() -> None:
    m = rasterize_corners([[1, 1], [4, 1], [4, 4], [1, 4]], 8, 8)
    assert m.shape == (8, 8) and m.sum() > 0, "box did not rasterize"
    inst = np.zeros((8, 8), np.uint8)
    inst[m > 0] = 3
    assert (inst == 3).sum() == int((m > 0).sum()), "instance paint lost pixels"
    # PNG round-trip keeps instance ids exact (no JPEG here — lossless).
    ok, buf = cv2.imencode(".png", inst)
    back = cv2.imdecode(np.frombuffer(buf.tobytes(), np.uint8), cv2.IMREAD_UNCHANGED)
    assert np.array_equal(back, inst), "instance PNG round-trip changed ids"
    print("self-check ok")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("kumo-dataset"), help="output dir (default: ./kumo-dataset)")
    p.add_argument("--base", default=os.environ.get("KUMO_TRACK_URL"), help="kumo-track base URL (env KUMO_TRACK_URL)")
    p.add_argument("--cookie", default=os.environ.get("KUMO_COOKIE"), help="kv_session cookie value (via the public proxy; env KUMO_COOKIE)")
    p.add_argument("--secret", default=os.environ.get("KUMO_TRACK_SHARED_SECRET"), help="X-Kumo-Proxy-Auth for DIRECT/local access (env KUMO_TRACK_SHARED_SECRET)")
    p.add_argument("--user", default=os.environ.get("KUMO_USER"), help="X-Kumo-User for DIRECT/local access (env KUMO_USER)")
    p.add_argument("--stride", type=int, default=None, help="decode stride; omit to use the server default (what the UI labels with)")
    p.add_argument("--quality", type=int, default=95, help="exported JPEG quality (default 95)")
    p.add_argument("--self-check", action="store_true", help="run the offline mask self-check and exit")
    a = p.parse_args()

    if a.self_check:
        _self_check()
        return
    if not a.base:
        p.error("--base (or KUMO_TRACK_URL) is required — e.g. https://kumo-track.example.com")
    export_all(Track(a.base, a.secret, a.user, a.cookie), a.out, a.stride, a.quality)


if __name__ == "__main__":
    main()
