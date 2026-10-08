/* Canvas interactions: two toolstrip modes — manual (select/brush/box, saved per
   frame with origin='manual', propagation-proof) and AI-assisted (SAM3 point/box
   prompts committed by an explicit Predict button/Enter) — plus select-tool box
   transform (move/resize/rotate) and zoom/pan. */
import { S, bus, changed, colorFor } from "./state.js";
import { $, toast } from "./dom.js";
import { jpost, maskUrl } from "./api.js";
import { ensureActiveObject } from "./objects.js";
import { gpuCall } from "./sam3.js";
import {
  cv, toFull, render, clampView, screenTol, pointInPoly, dx, dy,
  cornersToOBB, obbToCorners, transformPoint, rotHandle, invalidateMask, brushRadius,
} from "./canvas.js";

let panning = false, panStart = null;
export let ctrlHeld = false;
export function setCtrlHeld(v) { ctrlHeld = v; }

// Erase mode is sticky for the session (not persisted): switching frames must not
// silently drop back to paint. Read by enterBrushEdit, written by setEraseMode.
let stickyErase = false;

/* ---- modes & tools ----------------------------------------------------------
   The top two strip buttons switch the mode; each mode shows its own tools.
   Manual work is per frame; AI tools stage SAM3 prompts for Predict. */
function applyModeUi(m) {
  S.uiMode = m;
  $("manualTools").classList.toggle("off", m !== "manual");
  $("aiTools").classList.toggle("off", m !== "ai");
  $("modeManual").setAttribute("aria-pressed", String(m === "manual"));
  $("modeAI").setAttribute("aria-pressed", String(m === "ai"));
}

export function setMode(m) {
  const again = S.uiMode === m;
  applyModeUi(m);
  // Manual mode defaults to the brush; re-clicking the pointer drops to select.
  if (m === "manual") { if (again) setTool("select"); else armBrush(); }
  else if (!again) setTool("pos");
}

export function setTool(t) {
  // Leaving a manual editing tool commits its in-progress edit.
  if (S.tool === "brush" && t !== "brush" && S.edit && S.edit.mode === "brush") exitBrushEdit(true);
  if (S.tool === "poly" && t !== "poly" && S.edit && S.edit.mode === "poly") exitPolyEdit(true);
  S.tool = t;
  $("toolstrip").querySelectorAll("button[data-tool]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.tool === t)));
  cv.style.cursor = t === "select" ? "default" : "crosshair";
  render(); // overlays are tool-dependent (select handles, prompt markers)
}

/* Arm the brush/polygon tool. The edit opens right away when an object is
   active; with none it waits for the first stroke/click, which auto-creates
   one — so merely picking a tool never spawns an "object". */
function armBrush() {
  setTool("brush");
  if (S.activeObj != null && !(S.edit && S.edit.mode === "brush")) enterBrushEdit();
}
function armPoly() {
  setTool("poly");
  if (S.activeObj != null && !(S.edit && S.edit.mode === "poly")) enterPolyEdit();
}
export function activateBrush() {
  if (!S.video) return;
  applyModeUi("manual");
  armBrush();
}
export function activatePoly() {
  if (!S.video) return;
  applyModeUi("manual");
  armPoly();
}

/* Manual edits are scoped to one (object, frame). On every state change:
   discard an edit whose object was deleted (never commit onto a dead — and,
   with SQLite rowid reuse, resurrectable — id), re-target when the user moved
   to another object/frame, and drop a staged SAM3 seed of a deleted object. */
function syncManualEdit() {
  const e = S.edit;
  if (e && (e.mode === "brush" || e.mode === "poly")) {
    const alive = e.obj != null && S.objects.some((o) => o.id === e.obj);
    if (!alive) {
      e.mode === "brush" ? exitBrushEdit(false) : exitPolyEdit(false);
    } else if (e.obj !== S.activeObj || e.frame !== S.cur) {
      if (e.mode === "brush") {
        exitBrushEdit(true);
        if (S.tool === "brush" && S.activeObj != null) enterBrushEdit();
      } else {
        exitPolyEdit(true);
        if (S.tool === "poly" && S.activeObj != null) enterPolyEdit();
      }
    }
  }
  if (S.seed && S.seed.obj != null && !S.objects.some((o) => o.id === S.seed.obj)) clearSeed();
}

function activeFit() {
  return (S.ann.frames[S.cur] || {})[S.activeObj] || null;
}

/* ---- segment (explicit) ----------------------------------------------------
   Seeding only *stages* points / a box (drawn as a dashed overlay); pressing the
   Segment button or Enter runs SAM3. This is deliberate — auto-firing on mouse-up
   made it unclear what triggered segmentation (and each silent fire rebuilt the
   tracker window).

   The staged prompts persist across Segment presses so refinement is additive:
   place points → Segment → add +/− points → Segment, and SAM3 is re-queried each
   time on the full (previous + new) prompt set. The seed is scoped to one
   (object, frame); switching either discards it so prompts never bleed across
   targets. */
let segSeq = 0;

/* A fresh, empty seed tagged with the object + frame it belongs to. `order`
   records the staging sequence ("point" | "box") so undo is LIFO; `redo` holds
   the prompts undo peeled off (newest last) so they can be re-applied. */
function freshSeed() {
  return { points: [], labels: [], box: null, dragStart: null, order: [], redo: [], obj: S.activeObj, frame: S.cur };
}

/* Drop the staged seed unless it still belongs to the active object + frame. */
function ensureSeedScope() {
  if (S.seed.obj !== S.activeObj || S.seed.frame !== S.cur) S.seed = freshSeed();
}

/* Enable + highlight the Predict button to reflect the staged prompts: armed
   when something is staged, ready once SAM3 is awake. Undo/Redo enablement is
   routed by context (manual edit vs. seed prompts) — see syncUndoRedo. */
export function syncSeg() {
  const armed = S.activeObj != null && (S.seed.points.length > 0 || !!S.seed.box);
  const b = $("segBtn");
  if (b) { b.disabled = !armed || !S.sam3Ready; b.classList.toggle("armed", armed && S.sam3Ready); }
  syncUndoRedo();
}

/* Undo/Redo buttons are shared between manual edits and AI seed prompts: in an
   active brush/polygon edit they drive its snapshot stack, otherwise the staged
   SAM3 points/box. Called from syncSeg, on every manual-stack mutation, and on
   edit enter/exit so the buttons never reflect a stale context. */
export function syncUndoRedo() {
  const e = S.edit;
  const u = $("undoBtn"), r = $("redoBtn");
  if (e && (e.mode === "brush" || e.mode === "poly")) {
    if (u) u.disabled = !e.undo.length;
    if (r) r.disabled = !e.redo.length;
  } else {
    const armed = S.activeObj != null && (S.seed.points.length > 0 || !!S.seed.box);
    if (u) u.disabled = !armed;
    if (r) r.disabled = !(S.activeObj != null && (S.seed.redo || []).length > 0);
  }
}

/* Route Undo/Redo to the active context: a manual edit in progress, else the
   staged SAM3 seed. Shared by the toolbar buttons and the keyboard shortcuts. */
export function undoActive() { inMaskEdit() ? undoManualEdit() : undoSeedPoint(); }
export function redoActive() { inMaskEdit() ? redoManualEdit() : redoSeedPoint(); }

export async function doSegment() {
  if (S.activeObj == null) { toast("Pick or add an object first", "info"); return; }
  ensureSeedScope(); // a seed from another object/frame isn't ours to run
  if (!S.seed.points.length && !S.seed.box) {
    toast("Click + on the object or drag a box first, then Segment", "info");
    return;
  }
  const seq = ++segSeq;
  const atFrame = S.cur, atObj = S.activeObj;
  const hadBox = !!activeFit();
  try {
    const res = await gpuCall(() => jpost("/api/segment", {
      video_id: S.video.video_id, frame_idx: atFrame, obj_id: atObj,
      points: S.seed.points.length ? S.seed.points : null,
      labels: S.seed.points.length ? S.seed.labels : null,
      box: S.seed.box,
    }));
    if (seq !== segSeq || atObj !== S.activeObj || atFrame !== S.cur) return; // stale
    if (res.corners) {
      (S.ann.frames[atFrame] = S.ann.frames[atFrame] || {})[atObj] =
        { corners: res.corners, polygon: res.polygon, origin: "seed" };
      S.live.add(atObj); S.seedFrameOf[atObj] = atFrame;
      invalidateMask(atFrame, atObj); // brush mode: drop the cached overlay so the new mask re-fetches
      // Keep the staged prompts so the next Segment refines on top of them.
      syncSeg();
      changed();
      toast(hadBox
        ? "Refined — add more +/− points to keep adjusting, or ▶ / Space to track"
        : "Segmented — add +/− points to refine, or ▶ / Space to track", "success");
    } else {
      toast("No mask found — try another point or a box", "error");
    }
  } catch (e) { toast("Segment failed: " + e.message, "error"); }
}

/* ---- manual commit ----------------------------------------------------------
   A polygon becomes a rasterised mask server-side; corners alone stay a pure
   bounding box (no mask — see /api/annotations). obj/frame default to the
   current target but are passed explicitly by edits that may outlive it.

   Saves are optimistic and async. `S.pendingSave` tracks the in-flight one so a
   rapid tool switch (poly → brush, brush → poly) reads the committed result
   instead of a stale mask/outline — seeding the brush before a polygon's save
   landed would paint on blank and overwrite the polygon on the next commit.
   canvas.js also skips mask fetches while it is set (the mask is mid-write);
   the trailing render() fetches the settled mask once the save lands. */
function trackSave(work) {
  const p = work.finally(() => {
    if (S.pendingSave === p) { S.pendingSave = null; render(); }
  });
  S.pendingSave = p;
}

function commitManual(corners, polygon, obj = S.activeObj, frame = S.cur) {
  (S.ann.frames[frame] = S.ann.frames[frame] || {})[obj] =
    { corners, polygon, origin: "manual", has_mask: polygon != null };
  changed();
  trackSave((async () => {
    try {
      const res = await jpost("/api/annotations", {
        video_id: S.video.video_id, object_id: obj, frame_idx: frame,
        corners, polygon,
      }, "PUT");
      S.ann.frames[frame][obj] =
        { corners: res.corners, polygon: res.polygon, origin: "manual", has_mask: res.has_mask };
      invalidateMask(frame, obj); // the stored mask changed (or vanished) — re-fetch the overlay
      changed();
    } catch (e) { toast("Save failed: " + e.message, "error"); }
  })());
}

/* ---- polygon vertex editing --------------------------------------------------
   Loads the active object's outline (polygon, else box corners) for vertex
   editing; with no annotation on this frame it starts a draft — each click
   appends a vertex. Commits (≥3 vertices, rasterised to a mask) like the brush:
   on tool/frame/object switch or Esc. */
export function enterPolyEdit() {
  if (S.activeObj == null) return;
  // A just-committed brush stroke updates the outline async — wait for it so
  // the vertex editor opens on the committed shape, not a stale one.
  if (S.pendingSave) {
    const obj = S.activeObj, frame = S.cur;
    S.pendingSave.then(() => {
      if (S.tool === "poly" && S.activeObj === obj && S.cur === frame && !S.edit) beginPolyEdit();
    });
    return;
  }
  beginPolyEdit();
}

function beginPolyEdit() {
  const fit = activeFit();
  const pts = fit ? (fit.polygon || fit.corners) : null;
  S.edit = {
    mode: "poly", obj: S.activeObj, frame: S.cur,
    poly: pts ? pts.map((p) => [p[0], p[1]]) : [],
    draft: !pts, dragVertex: null, dirty: false,
    undo: [], redo: [],
  };
  syncDonePill();
  syncUndoRedo();
  render();
}

export function exitPolyEdit(commit) {
  if (!S.edit || S.edit.mode !== "poly") { if (S.edit) S.edit = null; return; }
  const { poly, dirty, obj, frame } = S.edit;
  S.edit = null;
  syncDonePill();
  syncUndoRedo();
  if (commit && dirty && poly.length >= 3) commitManual(null, poly, obj, frame);
  else render();
}

/* ---- mask-edit helpers ------------------------------------------------------ */
export function inMaskEdit() {
  return !!(S.edit && (S.edit.mode === "poly" || S.edit.mode === "brush"));
}
export function exitMaskEdit(commit) {
  if (!inMaskEdit()) return;
  const mode = S.edit.mode;
  if (mode === "brush") exitBrushEdit(commit);
  else exitPolyEdit(commit);
  setTool("select"); // an explicit exit (Enter/Esc/✓) drops back to the pointer
}

/* The ✓ Done pill doubles as the "you're in an edit" indicator. */
function syncDonePill() {
  const b = $("doneBtn");
  if (b) b.style.display = inMaskEdit() ? "" : "none";
}

/* ---- brush mask editing ----------------------------------------------------
   Paint (add) / Alt-paint (erase) onto an offscreen mask canvas kept at *native
   video* resolution (not display resolution — see enterBrushEdit), seeded with
   the object's *real* mask (fetched from /api/mask). On commit the PNG is sent
   to /api/mask, where it becomes the source-of-truth mask. Keeping the edit
   canvas 1:1 with the video avoids a second, independently-rounded resolution
   (the display canvas is capped at 1600px, the stored mask at 1024px — two
   unrelated scale factors that used to resample the mask on every edit cycle
   and visibly erode/shift thin strokes). */

/* Paint the object's stored mask into the brush canvas. Fetched fresh so the edit
   always starts from the real mask (never a box approximation); the guard drops a
   late arrival if the user moved on or already started painting. Waits for an
   in-flight save (e.g. the polygon just committed) so it seeds the new mask, and
   skips the request entirely when there is no mask to fetch (blank start). */
function seedBrushFromMask(mctx, mc, color, obj, frame) {
  const start = () => {
    const fit = (S.ann.frames[frame] || {})[obj];
    if (!fit || fit.has_mask === false) return; // nothing annotated / pure box — no mask exists
    const img = new Image();
    img.addEventListener("load", () => {
      if (!img.naturalWidth) return;
      if (!(S.edit && S.edit.mode === "brush" && S.activeObj === obj && S.cur === frame && !S.edit.dirty)) return;
      mctx.save();
      // Nearest-neighbour seed: the server thresholds at 127 and NEAREST-samples
      // back to the mask grid, so block replication makes untouched pixels
      // round-trip exactly — smoothed edges would creep a little on every cycle.
      mctx.imageSmoothingEnabled = false;
      mctx.globalCompositeOperation = "source-over";
      mctx.drawImage(img, 0, 0, mc.width, mc.height);
      mctx.globalCompositeOperation = "source-in";
      mctx.fillStyle = color; mctx.fillRect(0, 0, mc.width, mc.height);
      mctx.restore();
      render();
    }, { once: true });
    img.src = maskUrl(obj, frame);
  };
  S.pendingSave ? S.pendingSave.then(start) : start();
}

export function enterBrushEdit() {
  if (S.activeObj == null) { toast("Pick or add an object first", "info"); return; }
  const mc = document.createElement("canvas");
  mc.width = S.video.width; mc.height = S.video.height; // native res, not display (see block comment above)
  const mctx = mc.getContext("2d");
  const color = colorFor(S.activeObj);
  // obj + frame pin the edit to its target: commits go there even if the user
  // switched frame/object in the meantime (syncBrushEdit re-opens on the new one).
  // eraseMode/erasing start from the sticky session choice, not a hardcoded paint.
  S.edit = {
    mode: "brush", obj: S.activeObj, frame: S.cur, canvas: mc, mctx, color, dirty: false,
    stroking: false, eraseMode: stickyErase, erasing: stickyErase, last: null, cursor: null,
    undo: [], redo: [],
  };
  seedBrushFromMask(mctx, mc, color, S.activeObj, S.cur);
  $("brushbar").style.display = "";
  syncBrushBar();
  syncDonePill();
  syncUndoRedo();
  cv.style.cursor = "crosshair";
  render();
}

/* ---- brush toolbar (mouse-driven erase toggle + size) ---------------------- */
export function syncBrushBar() {
  const bar = $("brushbar");
  if (!bar) return;
  const erasing = !!(S.edit && S.edit.mode === "brush" && S.edit.eraseMode);
  $("brushPaint").setAttribute("aria-pressed", String(!erasing));
  $("brushErase").setAttribute("aria-pressed", String(erasing));
  $("brushSizeVal").textContent = erasing ? S.settings.eraserSize : S.settings.brushSize;
}

function setEraseMode(on) {
  if (!S.edit || S.edit.mode !== "brush") return;
  stickyErase = on; // sticky across frames for the rest of the session
  S.edit.eraseMode = on;
  S.edit.erasing = on; // reflect on the cursor immediately (Alt still overrides per-stroke)
  syncBrushBar();
  render();
}

export function bumpBrush(delta) {
  const erasing = !!(S.edit && S.edit.mode === "brush" && S.edit.eraseMode);
  const key = erasing ? "eraserSize" : "brushSize";
  S.settings[key] = Math.max(4, Math.min(200, S.settings[key] + delta));
  localStorage.setItem(erasing ? "annotate.eraserSize" : "annotate.brushSize", S.settings[key]);
  syncBrushBar();
  if (S.edit && S.edit.mode === "brush") render();
}

function paintBrush(a, b) {
  const e = S.edit, r = brushRadius(e.erasing), c = e.mctx;
  c.save();
  c.globalCompositeOperation = e.erasing ? "destination-out" : "source-over";
  c.fillStyle = e.color; c.strokeStyle = e.color;
  c.lineCap = "round"; c.lineJoin = "round"; c.lineWidth = r * 2;
  c.beginPath();
  if (b) { c.moveTo(a[0], a[1]); c.lineTo(b[0], b[1]); c.stroke(); }
  else { c.arc(a[0], a[1], r, 0, 7); c.fill(); }
  c.restore();
}

export function exitBrushEdit(commit) {
  if (!S.edit || S.edit.mode !== "brush") { if (S.edit) S.edit = null; return; }
  // Commit to the edit's own target, not the current one — the user may already
  // be on another frame/object by the time this runs.
  const { canvas, dirty, obj, frame } = S.edit;
  S.edit = null;
  syncDonePill();
  syncUndoRedo();
  const bar = $("brushbar"); if (bar) bar.style.display = "none";
  cv.style.cursor = S.tool === "select" ? "default" : "crosshair";
  if (!commit || !dirty) { render(); return; }
  commitMask(obj, frame, canvas.toDataURL("image/png").split(",")[1]);
}

function commitMask(obj, frame, mask_png) {
  trackSave((async () => {
    try {
      const res = await jpost("/api/mask", {
        video_id: S.video.video_id, object_id: obj, frame_idx: frame, mask_png,
      }, "PUT");
      const f = (S.ann.frames[frame] = S.ann.frames[frame] || {});
      if (res.corners) f[obj] = { corners: res.corners, polygon: res.polygon, origin: "manual", has_mask: true };
      else delete f[obj]; // erased to nothing → annotation removed server-side
      invalidateMask(frame, obj); // overlay re-fetches the just-stored mask
      changed();
    } catch (e) { toast("Brush save failed: " + e.message, "error"); render(); }
  })());
}

/* ---- manual undo/redo -------------------------------------------------------
   Scoped to the active edit session (S.edit.undo/redo), not persisted. A brush
   snapshot is a copy of the whole edit canvas; a polygon snapshot is a deep copy
   of the vertex array. Either way: push before the mutation, clear redo (a new
   edit invalidates whatever was undone), pop to undo/redo. */
// ponytail: 10-entry cap — a full-res brush-canvas snapshot is ~33MB, so this
// bounds memory; reused for polygon snapshots too even though those are cheap.
const UNDO_CAP = 10;

function cloneCanvas(src) {
  const c = document.createElement("canvas");
  c.width = src.width; c.height = src.height;
  c.getContext("2d").drawImage(src, 0, 0);
  return c;
}

function pushBrushUndo(e) {
  e.undo.push(cloneCanvas(e.canvas));
  if (e.undo.length > UNDO_CAP) e.undo.shift();
  e.redo = [];
  syncUndoRedo();
}

function pushPolyUndo(e) {
  e.undo.push(e.poly.map((v) => [v[0], v[1]]));
  if (e.undo.length > UNDO_CAP) e.undo.shift();
  e.redo = [];
  syncUndoRedo();
}

export function undoManualEdit() {
  const e = S.edit;
  if (!e || !e.undo.length) return;
  if (e.mode === "brush") {
    e.redo.push(cloneCanvas(e.canvas));
    const prev = e.undo.pop();
    e.mctx.clearRect(0, 0, e.canvas.width, e.canvas.height);
    e.mctx.drawImage(prev, 0, 0);
  } else { // poly
    e.redo.push(e.poly.map((v) => [v[0], v[1]]));
    e.poly = e.undo.pop();
  }
  e.dirty = true;
  syncUndoRedo();
  render();
}

export function redoManualEdit() {
  const e = S.edit;
  if (!e || !e.redo.length) return;
  if (e.mode === "brush") {
    e.undo.push(cloneCanvas(e.canvas));
    const next = e.redo.pop();
    e.mctx.clearRect(0, 0, e.canvas.width, e.canvas.height);
    e.mctx.drawImage(next, 0, 0);
  } else {
    e.undo.push(e.poly.map((v) => [v[0], v[1]]));
    e.poly = e.redo.pop();
  }
  e.dirty = true;
  syncUndoRedo();
  render();
}

/* ---- local/world helpers for box resize ------------------------------------ */
function worldToLocal(p, O) {
  const dx = p[0] - O.cx, dy = p[1] - O.cy, ca = Math.cos(-O.a), sa = Math.sin(-O.a);
  return [dx * ca - dy * sa, dx * sa + dy * ca];
}
function localToWorld(l, O) {
  const ca = Math.cos(O.a), sa = Math.sin(O.a);
  return [O.cx + l[0] * ca - l[1] * sa, O.cy + l[0] * sa + l[1] * ca];
}

/* ---- mouse ----------------------------------------------------------------- */
let mboxStart = null; // manual-box drag anchor (full-res px)
let mboxHintShown = false; // one adjust-hint toast per session

/* Cursor affordance for the select tool: what would a mousedown at p grab? */
function hoverCursor(p) {
  const fit = activeFit();
  if (!((S.config.editTool !== "brush" || fit?.has_mask === false) && fit && fit.corners)) return "default";
  const c = fit.corners, tol = screenTol(9);
  const rh = rotHandle(c);
  if (Math.hypot(rh[0] - p[0], rh[1] - p[1]) < tol) return "grab";
  for (let k = 0; k < 4; k++) {
    if (Math.hypot(c[k][0] - p[0], c[k][1] - p[1]) < tol) {
      // resize arrow along the corner's direction from the centre (fold to 4 axes)
      const O = cornersToOBB(c);
      const oct = Math.round(Math.atan2(c[k][1] - O.cy, c[k][0] - O.cx) / (Math.PI / 4)) & 3;
      return ["ew-resize", "nwse-resize", "ns-resize", "nesw-resize"][oct];
    }
  }
  if (pointInPoly(p[0], p[1], c)) return "move";
  const frame = S.ann.frames[S.cur] || {};
  for (const oid in frame) { // another object's box → click selects it
    const f = frame[oid];
    if (f && f.corners && Number(oid) !== S.activeObj && pointInPoly(p[0], p[1], f.corners)) return "pointer";
  }
  return "default";
}

/* First dab of a brush stroke at frame point p (drag continues in onMove). Paint
   coords (last) stay full-res — the edit canvas is native video res (see
   enterBrushEdit) — only the cursor circle is converted to display coords. */
function beginStroke(ev, p) {
  const e = S.edit;
  pushBrushUndo(e); // snapshot pre-stroke state so undo can restore it
  e.stroking = true; e.erasing = e.eraseMode || ev.altKey;
  e.last = p; e.cursor = [dx(p[0]), dy(p[1])];
  paintBrush(p, null); e.dirty = true; render();
}

/* Append a vertex to the draft polygon and let the press keep dragging it. */
function appendVertex(p) {
  pushPolyUndo(S.edit);
  S.edit.poly.push(p);
  S.edit.dragVertex = S.edit.poly.length - 1;
  S.edit.dirty = true;
  render();
}

async function onDown(ev) {
  if (!S.video) return;
  if (ev.ctrlKey) { // pan
    panning = true;
    const r = cv.getBoundingClientRect();
    panStart = { cx: ev.clientX, cy: ev.clientY, ox: S.view.ox, oy: S.view.oy, s: cv.width / r.width };
    cv.style.cursor = "grabbing";
    return;
  }
  const p = toFull(ev);

  // brush paint mode
  if (S.edit && S.edit.mode === "brush") {
    beginStroke(ev, p);
    return;
  }

  // polygon vertex mode
  if (S.edit && S.edit.mode === "poly") {
    const poly = S.edit.poly, tol = screenTol(8);
    for (let i = 0; i < poly.length; i++) {
      if (Math.hypot(poly[i][0] - p[0], poly[i][1] - p[1]) < tol) {
        if (ev.altKey) { if (poly.length > 3) { pushPolyUndo(S.edit); poly.splice(i, 1); S.edit.dirty = true; render(); } return; }
        pushPolyUndo(S.edit); // one snapshot per drag, taken at drag start — not per mousemove
        S.edit.dragVertex = i; return;
      }
    }
    for (let i = 0; i < poly.length; i++) { // edge midpoint → insert
      const a = poly[i], b = poly[(i + 1) % poly.length];
      const m = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
      if (Math.hypot(m[0] - p[0], m[1] - p[1]) < tol) {
        pushPolyUndo(S.edit);
        poly.splice(i + 1, 0, p); S.edit.dragVertex = i + 1; S.edit.dirty = true; render(); return;
      }
    }
    if (S.edit.draft) appendVertex(p); // drawing from scratch: clicks add vertices
    return;
  }

  if (S.tool === "select") {
    const fit = activeFit();
    // A raster mask can't follow box handles, so in brush mode the transform is
    // limited to pure bounding boxes (has_mask false); selection still works.
    if ((S.config.editTool !== "brush" || fit?.has_mask === false) && fit && fit.corners) {
      const c = fit.corners, tol = screenTol(9);
      const rh = rotHandle(c);
      if (Math.hypot(rh[0] - p[0], rh[1] - p[1]) < tol) {
        const O0 = cornersToOBB(c);
        S.edit = { mode: "rotate", O0, startAng: Math.atan2(p[1] - O0.cy, p[0] - O0.cx), origPoly: fit.polygon, preview: { corners: c, polygon: fit.polygon } };
        cv.style.cursor = "grabbing";
        return;
      }
      for (let k = 0; k < 4; k++) {
        if (Math.hypot(c[k][0] - p[0], c[k][1] - p[1]) < tol) {
          const O0 = cornersToOBB(c);
          S.edit = { mode: "resize", O0, k, origPoly: fit.polygon, preview: { corners: c, polygon: fit.polygon } };
          return;
        }
      }
      if (pointInPoly(p[0], p[1], c)) {
        const O0 = cornersToOBB(c);
        S.edit = { mode: "move", O0, start: p, origPoly: fit.polygon, preview: { corners: c, polygon: fit.polygon } };
        return;
      }
    }
    // select another object under the cursor
    const frame = S.ann.frames[S.cur] || {};
    for (const oid in frame) {
      const f = frame[oid];
      if (f && f.corners && Number(oid) !== S.activeObj && pointInPoly(p[0], p[1], f.corners)) {
        S.activeObj = Number(oid); changed(); return;
      }
    }
    return;
  }

  // Brush/polygon with no open edit yet (armed with no object): the first
  // stroke/click creates the "object" and opens the edit, then lands normally.
  if (S.tool === "brush") {
    if (!(await ensureActiveObject())) return;
    if (!(S.edit && S.edit.mode === "brush")) enterBrushEdit();
    if (S.edit && S.edit.mode === "brush") beginStroke(ev, p);
    return;
  }
  if (S.tool === "poly") {
    if (!(await ensureActiveObject())) return;
    if (!(S.edit && S.edit.mode === "poly")) enterPolyEdit();
    if (S.edit && S.edit.mode === "poly" && S.edit.draft) appendVertex(p);
    return;
  }

  // drawing tools (manual box + SAM3 prompts) need an object — auto-create one
  if (!(await ensureActiveObject())) return;

  if (S.tool === "mbox") { mboxStart = p; S.mboxPreview = null; return; }

  // SAM3 prompt staging
  ensureSeedScope(); // start fresh if the staged seed belongs to another object/frame
  if (S.tool === "box") { S.seed.box = null; S.seed.order = S.seed.order.filter((o) => o !== "box"); S.seed.dragStart = p; }
  else { S.seed.points.push(p); S.seed.labels.push(S.tool === "pos" ? 1 : 0); S.seed.order.push("point"); S.seed.redo = []; syncSeg(); render(); }
}

function onMove(ev) {
  if (panning && panStart) {
    S.view.ox = panStart.ox + (ev.clientX - panStart.cx) * panStart.s;
    S.view.oy = panStart.oy + (ev.clientY - panStart.cy) * panStart.s;
    clampView(); render();
    return;
  }
  if (S.edit && S.edit.mode === "brush") {
    const p = toFull(ev); // full-res paint coords; the cursor circle stays display-res
    S.edit.cursor = [dx(p[0]), dy(p[1])]; S.edit.erasing = S.edit.eraseMode || ev.altKey;
    if (S.edit.stroking) { paintBrush(S.edit.last, p); S.edit.last = p; S.edit.dirty = true; }
    render();
    return;
  }
  if (S.edit && S.edit.mode === "poly" && S.edit.dragVertex != null) {
    S.edit.poly[S.edit.dragVertex] = toFull(ev); S.edit.dirty = true; render(); return;
  }
  if (S.edit && S.edit.preview) {
    const p = toFull(ev), e = S.edit, O0 = e.O0;
    let O1;
    if (e.mode === "move") O1 = { ...O0, cx: O0.cx + (p[0] - e.start[0]), cy: O0.cy + (p[1] - e.start[1]) };
    else if (e.mode === "rotate") O1 = { ...O0, a: O0.a + (Math.atan2(p[1] - O0.cy, p[0] - O0.cx) - e.startAng) };
    else { // resize
      const hw = O0.w / 2, hh = O0.h / 2;
      const loc = [[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]];
      const opp = loc[(e.k + 2) % 4];
      const pl = worldToLocal(p, O0);
      const cxl = (pl[0] + opp[0]) / 2, cyl = (pl[1] + opp[1]) / 2;
      const w = Math.max(2, Math.abs(pl[0] - opp[0])), h = Math.max(2, Math.abs(pl[1] - opp[1]));
      const cw = localToWorld([cxl, cyl], O0);
      O1 = { cx: cw[0], cy: cw[1], w, h, a: O0.a };
    }
    e.preview.corners = obbToCorners(O1);
    e.preview.polygon = e.origPoly ? e.origPoly.map((pt) => transformPoint(pt, O0, O1)) : e.preview.corners;
    e._O1 = O1;
    render();
    return;
  }
  if (mboxStart) {
    const p = toFull(ev), s = mboxStart;
    S.mboxPreview = [Math.min(s[0], p[0]), Math.min(s[1], p[1]), Math.max(s[0], p[0]), Math.max(s[1], p[1])];
    render();
    return;
  }
  if (S.seed.dragStart) {
    const p = toFull(ev), s = S.seed.dragStart;
    S.seed.box = [Math.min(s[0], p[0]), Math.min(s[1], p[1]), Math.max(s[0], p[0]), Math.max(s[1], p[1])];
    syncSeg();
    render();
    return;
  }
  // idle move with the pointer: show what a click would grab (knob/corner/box)
  if (S.tool === "select" && !ctrlHeld && S.video) cv.style.cursor = hoverCursor(toFull(ev));
}

function onUp() {
  if (panning) { panning = false; panStart = null; cv.style.cursor = ctrlHeld ? "grab" : (S.tool === "select" ? "default" : "crosshair"); return; }
  if (S.edit && S.edit.mode === "brush") { S.edit.stroking = false; S.edit.last = null; return; }
  if (S.edit && S.edit.mode === "poly") { S.edit.dragVertex = null; return; }
  if (S.edit && S.edit.preview) {
    const e = S.edit; S.edit = null;
    // Only outline-backed annotations carry a polygon through the transform; a
    // pure box (origPoly null) stays corners-only — never grows a mask.
    if (e._O1) commitManual(e.preview.corners, e.origPoly ? e.preview.polygon : null);
    else render();
    return;
  }
  if (mboxStart) {
    const b = S.mboxPreview;
    mboxStart = null; S.mboxPreview = null;
    if (b && b[2] - b[0] >= 4 && b[3] - b[1] >= 4) {
      const corners = [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]];
      commitManual(corners, null); // a bounding box is coordinates only — no mask
      setTool("select"); // draw → adjust: handles show right away
      if (!mboxHintShown) {
        mboxHintShown = true;
        toast("Box saved — drag it to move, corners to resize, the knob above to rotate", "info", 5000);
      }
    } else render();
    return;
  }
  if (S.seed.dragStart) {
    if (S.seed.box && (S.seed.box[2] - S.seed.box[0] < 4 || S.seed.box[3] - S.seed.box[1] < 4)) S.seed.box = null;
    if (S.seed.box) { S.seed.order.push("box"); S.seed.redo = []; } // committed → newest on the undo stack
    S.seed.dragStart = null;
    syncSeg();
    render();
    return;
  }
}

export function clearSeed() { S.seed = freshSeed(); syncSeg(); render(); }
/* Undo the most-recently-staged prompt (LIFO): the last point or the box,
   whichever was added last — read off the `order` stack onto `redo`. */
export function undoSeedPoint() {
  const order = S.seed.order || (S.seed.order = []);
  const redo = S.seed.redo || (S.seed.redo = []);
  const last = order.pop();
  if (last === "box") { redo.push({ kind: "box", box: S.seed.box }); S.seed.box = null; }
  else if (last === "point") { redo.push({ kind: "point", point: S.seed.points.pop(), label: S.seed.labels.pop() }); }
  syncSeg();
  render();
}
/* Redo: re-stage whatever undo last peeled off, back onto the `order` stack. */
export function redoSeedPoint() {
  const redo = S.seed.redo || (S.seed.redo = []);
  const item = redo.pop();
  if (!item) return;
  const order = S.seed.order || (S.seed.order = []);
  if (item.kind === "box") { S.seed.box = item.box; order.push("box"); }
  else { S.seed.points.push(item.point); S.seed.labels.push(item.label); order.push("point"); }
  syncSeg();
  render();
}

export function initTools() {
  $("modeManual").onclick = () => setMode("manual");
  $("modeAI").onclick = () => setMode("ai");
  $("toolstrip").querySelectorAll("button[data-tool]").forEach((b) => {
    const t = b.dataset.tool;
    b.onclick = () => (t === "brush" ? activateBrush() : t === "poly" ? activatePoly() : setTool(t));
  });
  $("segBtn").onclick = doSegment;
  $("undoBtn").onclick = undoActive;
  $("redoBtn").onclick = redoActive;
  $("doneBtn").onclick = () => exitMaskEdit(true);
  if ($("brushbar")) {
    $("brushPaint").onclick = () => setEraseMode(false);
    $("brushErase").onclick = () => setEraseMode(true);
    $("brushMinus").onclick = () => bumpBrush(-4);
    $("brushPlus").onclick = () => bumpBrush(4);
  }
  syncSeg();
  // Keep an active manual edit glued to the current object + frame (panel
  // clicks fire "changed", scrubbing fires "frame") and drop state that
  // belongs to deleted objects.
  bus.addEventListener("changed", syncManualEdit);
  bus.addEventListener("frame", syncManualEdit);
  cv.addEventListener("mousedown", onDown);
  cv.addEventListener("mousemove", onMove);
  window.addEventListener("mouseup", onUp);
  cv.addEventListener("contextmenu", (ev) => {
    if (S.edit && S.edit.mode === "poly") { // right-click vertex = delete
      const p = toFull(ev), poly = S.edit.poly, tol = screenTol(8);
      for (let i = 0; i < poly.length; i++) {
        if (Math.hypot(poly[i][0] - p[0], poly[i][1] - p[1]) < tol && poly.length > 3) {
          pushPolyUndo(S.edit);
          poly.splice(i, 1); S.edit.dirty = true; render(); ev.preventDefault(); return;
        }
      }
    }
  });
  cv.addEventListener("wheel", (ev) => {
    if (!S.video || !ev.ctrlKey) return;
    ev.preventDefault();
    const factor = ev.deltaY < 0 ? 1.12 : 1 / 1.12;
    const r = cv.getBoundingClientRect();
    const cx = (ev.clientX - r.left) * cv.width / r.width;
    const cy = (ev.clientY - r.top) * cv.height / r.height;
    const newScale = Math.max(1, Math.min(20, S.view.scale * factor));
    const actual = newScale / S.view.scale;
    S.view.ox = cx - (cx - S.view.ox) * actual;
    S.view.oy = cy - (cy - S.view.oy) * actual;
    S.view.scale = newScale;
    clampView(); render();
    $("zoomPill").textContent = Math.round(S.view.scale * 100) + "%";
    $("zoomPill").style.display = S.view.scale !== 1 ? "" : "none";
  }, { passive: false });
}
