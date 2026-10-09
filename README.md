# Kumo Track

SAM3-assisted **video annotation**: annotate objects by hand (brush, polygon,
or box — per frame) or seed SAM3 with visual prompts and let its video tracker
propagate the mask across the whole clip, then export a labeled dataset.
Labels are free-form text, no built-in class set.

## Setup

A **uv** project driven by [poethepoet](https://poethepoet.natn.io/) tasks.
Install the task runner once, outside the project env:

```bash
uv tool install poethepoet   # provides `poe`
poe sync-local               # in-process SAM 3 on a local GPU (torch + transformers)
# — or —
poe sync                     # CPU-only; runs against the hosted SAM3 service (SAM3_URL)
```

SAM 3 runs one of two ways:

- **Local** (`poe sync-local`): loads `facebook/sam3` (~3.3 GB, pulled from
  HuggingFace on first run) in-process on a local GPU.
- **Remote** (`poe sync`): no torch/transformers installed — set `SAM3_URL` (and
  `SAM3_API_KEY`) and all detection/tracking is offloaded to the hosted SAM3
  service. See [Remote SAM3 service](#remote-sam3-service). The service itself
  is not part of this repository: you host your own, and
  `scripts/mock_sam3_service.py` documents the API it has to serve.

Copy `.env.example` to `.env` to override defaults.

### Docker

```bash
HF_TOKEN=hf_... docker compose up   # serves http://localhost:8080
```

Weights cache in the `hf-cache` volume; clips, exports and the annotation DB
land in `./data` / `./outputs` on the host. The default image is CPU-only
(~2 GB). For a GPU box, build with the stock PyPI torch instead and pass the
GPU through (needs nvidia-container-toolkit):

```bash
docker build --build-arg TORCH_INDEX=https://pypi.org/simple -t kumo-track:cuda .
docker run --gpus all -p 8080:8080 -e HF_TOKEN=hf_... \
  -v hf-cache:/app/hf-cache -v ./data:/app/data -v ./outputs:/app/outputs \
  kumo-track:cuda
```

### Deployment

> **kumo-track runs as exactly one replica.** The open-clip session cache and the
> GPU serialization lock are **per-process**: with ≥2 replicas behind a gateway,
> `/api/open` caches a session on one replica and a follow-up `/api/segment`
> round-robins to another → cache miss → `409 video not open`, and GPU calls stop
> being serialized. In local-GPU mode a single replica is also physically required
> (one GPU). Pin `replicas: 1` / `minReplicas: 1, maxReplicas: 1` (no autoscale);
> the app logs `single-replica mode — do not scale out` at boot as a reminder.
> Horizontal scale would need sticky-session routing by `video_id` **and** moving
> GPU serialization into the shared SAM3 service — out of scope today.

A `GET /api/health` (and `/healthz`) endpoint returns `200` without an
`X-Kumo-User` header even when `REQUIRE_AUTH=on`, for orchestrator
liveness/readiness probes — point the probe at it.

The metadata store is PostgreSQL (`DATABASE_URL=postgresql://…`) and
videos/exports live in Azure Blob (`STORAGE_BACKEND=azure`); dev and the test
suite use SQLite + local FS.

## Usage

```bash
poe annotate    # serves http://localhost:8080 (PORT to override)
```

Then in the browser:

1. **Open** a clip via the header clip-name button (list, upload, or switch).
   Frames are decoded *on demand* — even a long 4K clip opens in seconds.
2. **Pick a mode** in the tool strip: **pointer** = manual annotation, **wand**
   = AI-assisted (SAM3). Add an object in the panel (type a label, press Enter)
   — or just start drawing: with no object selected, one named "object" is
   created for you.
3. **AI mode** (`1`/`2`/`3`): stage **+ / −** points or drag a prompt **box**,
   then press **Predict** (`Enter`) to run SAM3. Prompts stay staged, so
   refinement is additive — add more points, Predict again. While the hosted
   service wakes from idle, a "SAM3 waking up…" pill shows and
   Predict/track/find-all are disabled; manual tools keep working.
4. **Manual mode** (`V`; the brush comes pre-armed): paint with the **brush**
   (`B`), outline a mask with the **polygon** (`P`), or drag a **box** — saved
   to the current frame only, and used as the SAM3 seed when you track from it.
   Finish an edit with **✓ Done**, `Enter`, or `Esc`; switching tool, frame or
   object also saves. A box stays pure coordinates (never a mask) and drops you
   to the pointer to move / resize / rotate it via the corner handles and the
   rotation knob.
5. **Track** `◀ N ▶` frames backward/forward from the current frame (N editable,
   default 50). Use **whole clip** for a full two-pass run, or press **Stop**
   (`Esc`) at any time — frames already streamed stay saved.
6. **Correct** a frame with the **pointer** (drag / resize / rotate) or repaint
   it with the brush. The brush shows an on-canvas toolbar to toggle the eraser
   and resize (or use `Alt`-drag to erase and `[` / `]` to resize). Hand edits
   are marked amber and propagation never overwrites them. Add **+/−** points
   on any frame to refine, then track onward.
7. **Static** objects (📌) skip SAM3: their current box is copied to every
   visited frame instead. Toggle off when the object starts moving.
8. **Find all** (button next to the label input) runs SAM3 or OWLv2 detection
   on the current frame to seed every instance of a label at once.
9. **Export** (header ⚙ menu): writes `data/<name>/images/*.jpg` + `labels.json`.

Config (all optional, via env / `.env`):

| Variable          | Default                  | Meaning                                         |
|-------------------|--------------------------|-------------------------------------------------|
| `VIDEO_DIR`       | `data/videos`            | Where clips are listed/served (+ uploads)       |
| `ANNOTATION_DB`   | `outputs/annotations.db` | SQLite annotation store                         |
| `PORT`            | `8080`                   | Port the app serves on                          |
| `STRIDE`          | `5`                      | Default frame-sampling stride (adjustable in ⚙) |
| `FRAME_CACHE_MB`  | `2048`                   | RAM budget for decoded frames                   |
| `TRACK_WINDOW`    | `200`                    | Max sampled frames per SAM3 tracking window (local only) |
| `TRACK_WINDOW_MB` | `1024`                   | RAM budget per window; caps it for big frames (local only) |
| `SAM3_URL`        | —                        | Hosted SAM3 service base URL; **set ⇒ remote mode** |
| `SAM3_API_KEY`    | —                        | `X-API-Key` for the service (keep out of git)   |
| `SAM3_TIMEOUT`    | `300`                    | Per-request timeout (s) for service calls       |
| `SAM3_READY_TIMEOUT` | `300`                 | Cold-start budget (s) polling `/health`         |

### Remote SAM3 service

When `SAM3_URL` is set, the app offloads all SAM 3 inference (find-all detection
and the segment/propagate tracker) to the hosted service over HTTP — no local
torch/transformers/GPU needed (`poe sync` skips them). The service exposes the
same `Fit` contract, so behaviour is unchanged; only the transport differs.

```bash
export SAM3_URL=https://<your-sam3-service>
export SAM3_API_KEY=…       # from your secret store — never commit it
poe annotate
```

Notes:

- The service **scales to zero when idle**, so the *first* call after a quiet
  spell stalls ~2 min while a replica spins up (the client polls `/health` and
  retries automatically). Subsequent calls are fast.
- Tracking sessions are ephemeral; the clip is uploaded on first use and the
  session is transparently reopened if it expires.
- **OWLv2 find-all stays local** (the service only serves SAM 3 detection); pick
  the SAM3 detector in remote mode, or `poe sync-local` to use OWLv2.

For UI work without the hosted service (it costs money while awake), run the
bundled **mock**, which mirrors the service API including the scale-to-zero /
cold-start lifecycle (tune with `MOCK_PORT`, `MOCK_COLD_START_S`, `MOCK_IDLE_S`):

```bash
uv run python scripts/mock_sam3_service.py             # :9911, starts "asleep"
SAM3_URL=http://127.0.0.1:9911 SAM3_API_KEY=mock poe annotate
```

(Shell-set variables beat `.env`, so this overrides a real `SAM3_URL` there.)

The frontend CSS is Tailwind 4, compiled with the standalone CLI (no Node —
the binary is fetched on first run). `index.html` links the generated
`css/app.build.css`, which is gitignored and rebuilt inside the Docker image;
locally, build it once after cloning and after editing `app.css` or markup:

```bash
poe css            # one-shot; `poe css --watch` while doing UI work
```

How masks are **stored and displayed** is set in `config.toml` at the repo root
(override the path with `CONFIG_FILE`):

```toml
[annotation]
edit_tool = "polygon"  # or "brush"
```

`polygon` keeps a polygon outline per annotation, drawn as a filled outline;
`brush` makes the rasterised mask the source of truth, rendered as a tinted
overlay. Manual mode always offers the brush, polygon and box tools regardless.
A missing or invalid file falls back to `polygon`.

Run the tests (no GPU / SAM3 weights needed — they use a fake tracker / stubbed
service client). The local-tracker test additionally needs torch and is skipped
unless `poe sync-local` is installed:

```bash
poe test
```

## Keyboard shortcuts

`←/→` frame · `Shift+←/→` jump 10 · `V` manual mode (pointer) · `B`/`P`
brush/polygon · `1/2/3` AI mode: point+ / point− / prompt box · `[`/`]` brush
size · `Alt`-drag erase · `Enter` finish edit (manual) / predict (AI) ·
`Ctrl+Z`/`U` undo prompt · `Ctrl+Y` redo · `Space` track fwd N ·
`Esc` stop / finish edit / close overlay · `Ctrl+scroll` zoom · `Ctrl`-drag pan

## Box format

Annotations and exports use a 4-corner format (top-left → clockwise, rotation
baked in). Mask-backed annotations (brush, polygon, SAM3) additionally export
their mask as COCO RLE; a manual bounding box is coordinates only and never
carries a mask:

```json
{ "corners": [[x1,y1], [x2,y1], [x2,y2], [x1,y2]], "label": "<your label>" }
```

## Layout

```
scripts/annotate_app.py          thin entrypoint (`poe annotate`) over create_app()
scripts/mock_sam3_service.py     mock of the hosted SAM3 service (UI dev without GPU/costs)
src/kumo_track/annotate/
  app.py        FastAPI routes (create_app factory; tracker injected for tests)
  frames.py     FrameSource — on-demand decode, RAM LRU + JPEG disk cache
  tracker.py    TrackerManager/TrackerWindow — sliding-window SAM3 tracking
  session.py    VideoSession — ties frames + tracker together
  db.py         SQLite store (video / object / annotation; static + manual)
  static/
    index.html  shell (header, canvas, bottom bar, right panel, overlays)
    css/app.css Tailwind 4 source: brand tokens (@theme) + component styles → app.build.css (`poe css`)
    js/api.js   fetch helpers, NDJSON stream reader, AbortController
    js/state.js app-state + pub/sub
    js/canvas.js rendering, zoom/pan, hit-testing
    js/tools.js manual/AI modes, brush + polygon + box tools, box transform, SAM3 prompts
    js/timeline.js scrubber, coverage ticks, playhead
    js/objects.js object panel, static toggle, rename/delete, find-all
    js/track.js propagation controls + streaming progress + stop
    js/sam3.js  SAM3 service health poll → wake-up pill + button gating
    js/main.js  bootstrapping, keyboard map
src/kumo_track/models/           SAM3 + OWLv2 detectors (used by "Find all")
src/kumo_track/{base,tiled,geometry}.py  shared detector base + tiling/IoU helpers
tests/                           GPU-free unit + API tests (fake tracker)
```

The DB is the source of truth — annotations survive restarts and any tracking
window is reconstructible from it. Annotation **origin** precedence is
`manual` > `seed` > `propagated`, so propagation never clobbers hand-corrected
boxes. A single SAM3 model loads once per process; one clip is active at a time
and every GPU call is serialised.

## Models and licenses

KumoTrack is licensed under Apache 2.0 (see `LICENSE`). The models it downloads
are not part of this repository and come with their own licenses:

- [`facebook/sam3`](https://huggingface.co/facebook/sam3) (Meta SAM 3): released
  under Meta's SAM License, not Apache 2.0. The weights are gated on Hugging Face:
  accept the license on the model page and log in (`hf auth login`) before the
  first local run.
- [`google/owlv2-large-patch14-ensemble`](https://huggingface.co/google/owlv2-large-patch14-ensemble)
  (OWLv2): Apache 2.0.

Check the model licenses before you use KumoTrack commercially.
