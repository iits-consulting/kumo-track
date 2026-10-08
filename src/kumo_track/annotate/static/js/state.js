/* Central mutable app state + a tiny event bus.

   Modules import `S` and mutate it directly, then call `changed()` to ask the UI
   to re-render everything (main.js subscribes). High-frequency canvas updates call
   the canvas `render()` directly instead, to avoid full refreshes per mouse-move. */

export const S = {
  video: null,        // {video_id, n_frames, width, height, source_indices, stride, fps}
  objects: [],        // [{id,label,static,hidden,color}]
  activeObj: null,
  ann: { frames: {} },// {frameIdx: {objId: {corners,polygon,origin}}}
  cur: 0,
  uiMode: "manual",   // manual (select/brush/poly/mbox) | ai (SAM3 prompting)
  tool: "select",     // manual: select | brush | poly | mbox · ai: pos | neg | box
  seed: { points: [], labels: [], box: null, dragStart: null },
  mboxPreview: null,  // in-progress manual box drag [x1,y1,x2,y2] (full-res px)
  sam3Ready: false,   // /api/sam3/health said "ready" (sam3.js polls)
  live: new Set(),    // object ids seeded into the tracker this session
  seedFrameOf: {},    // objId -> last frame it was seeded on
  edit: null,         // active manual edit: {mode, ...} (canvas/tools own the shape)
  pendingSave: null,  // in-flight annotation save (tools.js) — gates mask reads/fetches
  view: { scale: 1, ox: 0, oy: 0 },
  canvasScale: 1,
  propagating: false,
  config: { editTool: "polygon" }, // server-set (GET /api/config): "polygon" | "brush"
  settings: {
    colorByCategory: (localStorage.getItem("annotate.colorByCategory") ?? "true") !== "false",
    stride: parseInt(localStorage.getItem("annotate.stride") || "5", 10),
    findThreshold: parseFloat(localStorage.getItem("annotate.findThreshold") || "0.50"),
    findDetector: localStorage.getItem("annotate.findDetector") || "sam3",
    trackN: parseInt(localStorage.getItem("annotate.trackN") || "50", 10),
    brushSize: parseInt(localStorage.getItem("annotate.brushSize") || "28", 10), // screen px diameter
    eraserSize: parseInt(localStorage.getItem("annotate.eraserSize") || "28", 10), // screen px diameter
  },
};

export const bus = new EventTarget();
export const changed = () => bus.dispatchEvent(new Event("changed"));

/* Functional, brand-adjacent identity palette (no gold/orange — reserved for manual). */
export const PALETTE = ["#1DB5B0", "#E6423A", "#6C8CFF", "#C792EA", "#F06292", "#26C6DA", "#7E57C2", "#4DB6AC"];

function labelHash(str) {
  let h = 5381;
  for (let i = 0; i < str.length; i++) h = (Math.imul(h, 31) + str.charCodeAt(i)) >>> 0;
  return h % PALETTE.length;
}

export function colorFor(oid) {
  const o = S.objects.find((x) => x.id === oid);
  if (o && o.color) return o.color; // per-object override beats either palette mode
  if (S.settings.colorByCategory) {
    return PALETTE[labelHash(o ? o.label : String(oid))];
  }
  const idx = S.objects.findIndex((x) => x.id === oid);
  return idx >= 0 ? PALETTE[idx % PALETTE.length] : "#9aa3af";
}

export function objById(oid) {
  return S.objects.find((o) => o.id === oid) || null;
}

export function annCount(oid) {
  let c = 0;
  for (const k in S.ann.frames) { const f = S.ann.frames[k]; if (f && f[oid] && f[oid].corners) c++; }
  return c;
}
