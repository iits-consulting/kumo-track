/* Canvas: frame display (with flicker-free keep-last-frame), annotation overlays,
   seed markers, manual-edit accents, select/edit handles, zoom + pan, plus the
   geometry helpers (OBB <-> corners, hit-testing) shared with tools.js. */
import { S, colorFor, objById } from "./state.js";
import { frameUrl, maskUrl, bumpMaskVer } from "./api.js";

export const cv = document.getElementById("cv");
const ctx = cv.getContext("2d");
const HANDLE_R = 6; // screen px

/* ---- frame image cache (keep last frame on screen until the next loads) ---- */
const frameCache = new Map();
const FRAME_CACHE_MAX = 96;
let curImg = null;
/* Scrub-flood guard. Dragging the scrubber fires goto() per pixel, and every
   uncached frame is a server-side seek + decode — a fast back-and-forth drag
   could pile up hundreds of in-flight requests. At most MAX_INFLIGHT fetches run
   at once; further requests collapse into `wanted` (latest wins), so a fast drag
   shows frames as quickly as the server can serve them and skips the rest.
   Cached frames still render instantly, keeping the video-like feel. */
const MAX_INFLIGHT = 2;
let inflight = 0;
let wanted = -1;
let cacheGen = 0;

export function clearFrameCache() {
  frameCache.clear(); curImg = null;
  inflight = 0; wanted = -1; cacheGen++;
}

/* ---- mask overlay cache (brush mode) --------------------------------------
   In brush mode the segmentation is the real mask, fetched per (frame,object)
   as a white+alpha PNG from /api/mask and tinted to the object colour. Entries
   persist (a 404 → `empty`, so missing masks aren't re-fetched every render). */
const maskCache = new Map(); // `${frame}:${oid}` -> {img, ready, empty, tint, tintColor}

export function clearMaskCache() { maskCache.clear(); }
export function invalidateMask(frame, oid) { bumpMaskVer(oid, frame); maskCache.delete(`${frame}:${oid}`); }

function maskEntry(frame, oid) {
  const key = `${frame}:${oid}`;
  let e = maskCache.get(key);
  if (e) return e;
  // A save is in flight: the stored mask is mid-write, so fetching now would
  // 404 or return the old one. The post-save render comes back for it.
  if (S.pendingSave) return null;
  e = { img: null, ready: false, empty: false, tint: null, tintColor: null };
  maskCache.set(key, e);
  const im = new Image();
  im.addEventListener("load", () => {
    e.ready = im.naturalWidth > 0; e.empty = !e.ready; if (e.ready) e.img = im;
    if (S.cur === frame) render();
  }, { once: true });
  im.addEventListener("error", () => { e.empty = true; }, { once: true });
  im.src = maskUrl(oid, frame);
  return e;
}

function tintedMask(e, color) {
  if (!e.ready) return null;
  if (e.tint && e.tintColor === color) return e.tint;
  const c = document.createElement("canvas");
  c.width = e.img.naturalWidth; c.height = e.img.naturalHeight;
  const cx = c.getContext("2d");
  cx.drawImage(e.img, 0, 0);
  cx.globalCompositeOperation = "source-in"; // paint colour through the mask's alpha
  cx.fillStyle = color; cx.fillRect(0, 0, c.width, c.height);
  e.tint = c; e.tintColor = color;
  return c;
}

function drawMaskFor(oid, color) {
  const e = maskEntry(S.cur, oid);
  if (!e) return;
  const t = tintedMask(e, color);
  if (!t) return;
  ctx.save();
  ctx.globalAlpha = 0.5;
  try { ctx.drawImage(t, 0, 0, cv.width, cv.height); } catch {}
  ctx.restore();
}

function fetchFrame(i) {
  const gen = cacheGen;
  inflight++;
  const im = new Image();
  const done = (ok) => {
    if (gen !== cacheGen) return; // cache cleared mid-flight (clip switch)
    inflight--;
    if (!ok) frameCache.delete(i);
    if (S.cur === i) { curImg = ok && im.naturalWidth > 0 ? im : null; render(); }
    if (wanted >= 0) { const w = wanted; wanted = -1; showFrame(w); }
  };
  im.addEventListener("load", () => done(true), { once: true });
  im.addEventListener("error", () => done(false), { once: true });
  im.src = frameUrl(i);
  frameCache.set(i, im);
  if (frameCache.size > FRAME_CACHE_MAX) frameCache.delete(frameCache.keys().next().value);
}

export function showFrame(i) {
  // Fetch only the requested frame. Each frame decode is server-side work (a far
  // seek + decode on long/4K clips), so prefetching neighbours multiplied that
  // cost on every scrub and piled memory pressure on the decoder; one frame per
  // navigation keeps clicking between frames fast. Recently-viewed frames stay in
  // the browser Image cache, so stepping back is still instant.
  const im = frameCache.get(i);
  if (im) {
    frameCache.delete(i); frameCache.set(i, im); // LRU touch
    if (im.complete) curImg = im.naturalWidth > 0 ? im : null;
    // else still in flight — its load handler renders if we're still here.
    wanted = -1;
    return;
  }
  if (inflight >= MAX_INFLIGHT) { wanted = i; return; }
  fetchFrame(i);
}

export function setupCanvas() {
  clearFrameCache();
  clearMaskCache();
  const maxSide = 1600;
  const long = Math.max(S.video.width, S.video.height);
  S.canvasScale = Math.min(1, maxSide / long);
  cv.width = Math.round(S.video.width * S.canvasScale);
  cv.height = Math.round(S.video.height * S.canvasScale);
  S.view = { scale: 1, ox: 0, oy: 0 };
}

/* ---- coordinate transforms (display <-> full-res frame px) ----------------- */
export function toFull(ev) {
  const r = cv.getBoundingClientRect();
  const cx = (ev.clientX - r.left) * cv.width / r.width;
  const cy = (ev.clientY - r.top) * cv.height / r.height;
  return [(cx - S.view.ox) / S.view.scale * S.video.width / cv.width,
          (cy - S.view.oy) / S.view.scale * S.video.height / cv.height];
}
export const dx = (x) => x * cv.width / S.video.width;
export const dy = (y) => y * cv.height / S.video.height;
/* full-coords tolerance equivalent to `px` screen pixels (handle hit-testing) */
export function screenTol(px) {
  const r = cv.getBoundingClientRect();
  return px * (cv.width / r.width) / S.view.scale * S.video.width / cv.width;
}

/* ---- zoom / pan ------------------------------------------------------------ */
export function clampView() {
  const v = S.view;
  if (v.scale <= 1) { v.scale = 1; v.ox = 0; v.oy = 0; return; }
  v.ox = Math.min(0, Math.max(cv.width * (1 - v.scale), v.ox));
  v.oy = Math.min(0, Math.max(cv.height * (1 - v.scale), v.oy));
}

/* ---- geometry (shared with tools.js) --------------------------------------- */
export function pointInPoly(px, py, pts) {
  let inside = false;
  for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
    const xi = pts[i][0], yi = pts[i][1], xj = pts[j][0], yj = pts[j][1];
    if ((yi > py) !== (yj > py) && px < (xj - xi) * (py - yi) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}
// corners (cyclic rectangle) -> oriented box {cx,cy,w,h,a}
export function cornersToOBB(c) {
  const cx = (c[0][0] + c[1][0] + c[2][0] + c[3][0]) / 4;
  const cy = (c[0][1] + c[1][1] + c[2][1] + c[3][1]) / 4;
  const e1x = c[1][0] - c[0][0], e1y = c[1][1] - c[0][1];
  const e2x = c[2][0] - c[1][0], e2y = c[2][1] - c[1][1];
  return { cx, cy, w: Math.hypot(e1x, e1y), h: Math.hypot(e2x, e2y), a: Math.atan2(e1y, e1x) };
}
export function obbToCorners(o) {
  const ca = Math.cos(o.a), sa = Math.sin(o.a);
  const hw = o.w / 2, hh = o.h / 2;
  const loc = [[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]];
  return loc.map(([x, y]) => [o.cx + x * ca - y * sa, o.cy + x * sa + y * ca]);
}
// map a point through the box transform O0 -> O1 (covers move/rotate/resize)
export function transformPoint(p, O0, O1) {
  const ca0 = Math.cos(-O0.a), sa0 = Math.sin(-O0.a);
  let lx = (p[0] - O0.cx) * ca0 - (p[1] - O0.cy) * sa0;
  let ly = (p[0] - O0.cx) * sa0 + (p[1] - O0.cy) * ca0;
  lx *= O1.w / (O0.w || 1); ly *= O1.h / (O0.h || 1);
  const ca1 = Math.cos(O1.a), sa1 = Math.sin(O1.a);
  return [O1.cx + lx * ca1 - ly * sa1, O1.cy + lx * sa1 + ly * ca1];
}
// rotation handle position (full coords): outside the top edge midpoint.
// Box-local "up" (toward the top edge) maps to world (sin a, -cos a).
export function rotHandle(corners) {
  const o = cornersToOBB(corners);
  const off = o.h / 2 + screenTol(22);
  return [o.cx + Math.sin(o.a) * off, o.cy - Math.cos(o.a) * off];
}

/* ---- rendering ------------------------------------------------------------- */
export function render() {
  ctx.save();
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = "#04141c";
  ctx.fillRect(0, 0, cv.width, cv.height);
  ctx.setTransform(S.view.scale, 0, 0, S.view.scale, S.view.ox, S.view.oy);
  if (curImg && curImg.complete && curImg.naturalWidth > 0) {
    try { ctx.drawImage(curImg, 0, 0, cv.width, cv.height); } catch {}
  }
  const brushing = S.edit && S.edit.mode === "brush";
  const f = S.ann.frames[S.cur] || {};
  for (const oid in f) {
    const fit = f[oid];
    if (!fit) continue;
    if (objById(Number(oid))?.hidden) continue; // visibility toggle: display only

    // The active object's box may be mid-edit; tools.js stashes a preview on S.edit.
    const previewing = S.edit && S.edit.preview && Number(oid) === S.activeObj;
    const corners = previewing ? S.edit.preview.corners : fit.corners;
    const polygon = previewing ? S.edit.preview.polygon : fit.polygon;
    const col = colorFor(Number(oid));
    const isActive = Number(oid) === S.activeObj;
    const brushCfg = S.config.editTool === "brush";
    // The brush canvas replaces the active object's mask while painting — but its
    // box outline stays (a pure box seeds a blank brush and would vanish entirely).
    const brushingThis = brushing && isActive;
    // real mask is the segmentation; pure boxes (has_mask false) have none to fetch
    if (brushCfg) { if (fit.has_mask !== false && !brushingThis) drawMaskFor(Number(oid), col); }
    else if (polygon && !brushingThis) drawPath(polygon, col, 1, true);
    if (corners) drawPath(corners, col, isActive ? 3 : 1.6, false);
    if (corners && fit.origin === "manual") drawManualAccents(corners);
    // Handles mirror the transform gate in tools.js: in brush config only a
    // pure box (no raster mask that couldn't follow) can be dragged/rotated.
    if (isActive && corners && (!brushCfg || fit.has_mask === false) && S.tool === "select" && !(S.edit && S.edit.mode === "poly"))
      drawBoxHandles(corners);
  }
  if (S.edit && S.edit.mode === "poly" && S.edit.poly) drawPolyHandles(S.edit.poly);
  if (brushing) drawBrush(S.edit);
  drawSeed();
  if (S.mboxPreview) { // manual box being dragged — amber like other manual edits
    const b = S.mboxPreview;
    ctx.strokeStyle = "#F6A609"; ctx.setLineDash([6, 4]); ctx.lineWidth = 1.5;
    ctx.strokeRect(dx(b[0]), dy(b[1]), dx(b[2] - b[0]), dy(b[3] - b[1]));
    ctx.setLineDash([]);
  }
  ctx.restore();
}

/* Brush/eraser radius in full-res px for a given erasing state (S.edit.erasing:
   mode + the per-stroke Alt override) — shared by the paint canvas (tools.js,
   itself full-res) and the cursor circle below (scaled to display via dx). */
export function brushRadius(erasing) {
  const size = erasing ? S.settings.eraserSize : S.settings.brushSize;
  return Math.max(1, screenTol(size / 2));
}

function drawBrush(e) {
  ctx.save();
  ctx.globalAlpha = 0.5;
  try { ctx.drawImage(e.canvas, 0, 0, cv.width, cv.height); } catch {}
  ctx.globalAlpha = 1;
  if (e.cursor) {
    const r = dx(brushRadius(e.erasing));
    ctx.beginPath(); ctx.arc(e.cursor[0], e.cursor[1], r, 0, 7);
    ctx.lineWidth = 1.5 / S.view.scale;
    ctx.strokeStyle = e.erasing ? "#E6423A" : "#fff"; ctx.stroke();
  }
  ctx.restore();
}

function drawPath(pts, color, w, fill) {
  if (!pts || !pts.length) return;
  ctx.beginPath();
  ctx.moveTo(dx(pts[0][0]), dy(pts[0][1]));
  for (let i = 1; i < pts.length; i++) ctx.lineTo(dx(pts[i][0]), dy(pts[i][1]));
  ctx.closePath();
  ctx.lineWidth = w; ctx.strokeStyle = color; ctx.stroke();
  if (fill) { ctx.globalAlpha = 0.16; ctx.fillStyle = color; ctx.fill(); ctx.globalAlpha = 1; }
}

function drawManualAccents(corners) {
  const len = 10;
  ctx.strokeStyle = "#F6A609"; ctx.lineWidth = 3;
  for (const c of corners) {
    ctx.beginPath();
    ctx.arc(dx(c[0]), dy(c[1]), 4, 0, 7);
    ctx.stroke();
  }
}

function drawBoxHandles(corners) {
  ctx.fillStyle = "#fff"; ctx.strokeStyle = "#021D27"; ctx.lineWidth = 1.5;
  for (const c of corners) {
    ctx.beginPath(); ctx.rect(dx(c[0]) - HANDLE_R, dy(c[1]) - HANDLE_R, HANDLE_R * 2, HANDLE_R * 2);
    ctx.fill(); ctx.stroke();
  }
  const r = rotHandle(corners);
  const top = [(corners[0][0] + corners[1][0]) / 2, (corners[0][1] + corners[1][1]) / 2];
  ctx.beginPath(); ctx.moveTo(dx(top[0]), dy(top[1])); ctx.lineTo(dx(r[0]), dy(r[1]));
  ctx.strokeStyle = "#fff"; ctx.lineWidth = 1.5; ctx.stroke();
  ctx.beginPath(); ctx.arc(dx(r[0]), dy(r[1]), HANDLE_R, 0, 7);
  ctx.fillStyle = "#1DB5B0"; ctx.fill(); ctx.strokeStyle = "#fff"; ctx.stroke();
}

function drawPolyHandles(poly) {
  // outline
  drawPath(poly, "#F6A609", 2, false);
  // edge midpoints (insert) + vertices (drag/delete)
  ctx.fillStyle = "rgba(246,166,9,.5)";
  for (let i = 0; i < poly.length; i++) {
    const a = poly[i], b = poly[(i + 1) % poly.length];
    const m = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
    ctx.beginPath(); ctx.arc(dx(m[0]), dy(m[1]), 3.5, 0, 7); ctx.fill();
  }
  ctx.fillStyle = "#fff"; ctx.strokeStyle = "#F6A609"; ctx.lineWidth = 2;
  for (const v of poly) {
    ctx.beginPath(); ctx.arc(dx(v[0]), dy(v[1]), HANDLE_R, 0, 7); ctx.fill(); ctx.stroke();
  }
}

function drawSeed() {
  const s = S.seed;
  if (s.frame !== S.cur || s.obj !== S.activeObj) return; // the seed belongs to one object+frame
  if (s.box) {
    const b = s.box;
    ctx.strokeStyle = "#fff"; ctx.setLineDash([6, 4]); ctx.lineWidth = 1.5;
    ctx.strokeRect(dx(b[0]), dy(b[1]), dx(b[2] - b[0]), dy(b[3] - b[1]));
    ctx.setLineDash([]);
  }
  s.points.forEach((p, i) => {
    ctx.beginPath(); ctx.arc(dx(p[0]), dy(p[1]), 5, 0, 7);
    ctx.fillStyle = s.labels[i] ? "#1DB5B0" : "#E6423A"; ctx.fill();
    ctx.lineWidth = 2; ctx.strokeStyle = "#fff"; ctx.stroke();
  });
}
