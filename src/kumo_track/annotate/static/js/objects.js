/* Objects panel: create / rename / delete, static toggle, find-all,
   and the per-object kebab menu. */
import { S, changed, colorFor, objById, annCount } from "./state.js";
import { $, svg, toast, busy } from "./dom.js";
import { jpost, jget } from "./api.js";
import { invalidateMask } from "./canvas.js";
import { goto } from "./timeline.js";
import { gpuCall } from "./sam3.js";

let editingObj = null;
const LABELS_KEY = "annotate.recentLabels";

export function currentLabel() { return $("labelInput").value.trim(); }

function recentLabels() {
  try { return JSON.parse(localStorage.getItem(LABELS_KEY) || "[]"); } catch { return []; }
}
function rememberLabel(label) {
  if (!label) return;
  const next = [label, ...recentLabels().filter((l) => l !== label)].slice(0, 50);
  localStorage.setItem(LABELS_KEY, JSON.stringify(next));
}
export function refreshLabelOptions() {
  const seen = new Set(), opts = [];
  for (const l of [...S.objects.map((o) => o.label), ...recentLabels()]) {
    if (l && !seen.has(l)) { seen.add(l); opts.push(l); }
  }
  $("labelOptions").innerHTML = opts.map((l) => `<option value="${l.replace(/"/g, "&quot;")}">`).join("");
}

export async function reloadAnnotations() {
  const a = await jget(`/api/annotations?video_id=${S.video.video_id}`);
  S.objects = (a.objects || []).map((o) => ({
    id: o.id, label: o.label, static: o.static, hidden: o.hidden, color: o.color,
  }));
  const frames = {};
  for (const k in (a.frames || {})) frames[+k] = a.frames[k];
  S.ann = { frames };
  changed();
}

export function updateMode() {
  const sw = $("modeSw"), txt = $("modeTxt");
  if (S.activeObj == null) { sw.style.background = "#5a7077"; txt.textContent = "no object selected"; return; }
  const o = objById(S.activeObj);
  sw.style.background = colorFor(S.activeObj);
  txt.textContent = o ? `${o.label}${o.static ? " · static" : ""}` : "object";
}

export function renderObjs() {
  const el = $("objList");
  el.innerHTML = "";
  if (!S.objects.length) {
    el.innerHTML = `<div class="obj-empty">No objects yet — type a label and press <span class="kbd">Enter</span>.</div>`;
    updateMode();
    return;
  }
  S.objects.forEach((o) => {
    const d = document.createElement("div");
    d.className = "obj" + (o.id === S.activeObj ? " active" : "");
    if (editingObj === o.id) {
      d.innerHTML = `<span class="swatch" style="background:${colorFor(o.id)}"></span>
        <input class="rename" value="${o.label.replace(/"/g, "&quot;")}">`;
      el.appendChild(d);
      const inp = d.querySelector("input"); inp.focus(); inp.select();
      const commit = async () => {
        const v = inp.value.trim(); editingObj = null;
        if (v && v !== o.label) await renameObj(o.id, v); else changed();
      };
      inp.addEventListener("keydown", (e) => {
        if (e.key === "Enter") commit();
        if (e.key === "Escape") { editingObj = null; changed(); }
      });
      inp.addEventListener("blur", commit);
      return;
    }
    if (o.hidden) d.classList.add("hidden-obj");
    const live = S.live.has(o.id);
    const liveDot = o.static
      ? `<span class="pin on" title="static — box copied when tracking, not passed to SAM3">${svg("pin")}</span>`
      : `<span class="live ${live ? "on" : "off"}" title="${live ? "live in tracker" : "re-seed to track"}"></span>`;
    const col = colorFor(o.id);
    d.innerHTML = `
      <label class="swatch" style="background:${col}" title="Pick color">
        <input type="color" class="color-input" value="${col}" aria-label="Object color"></label>
      <span class="lbl">${o.label}</span>
      <span class="count" title="annotated frames">${annCount(o.id)}f</span>
      ${liveDot}
      <button class="btn icon sm ghost vis-toggle" title="${o.hidden ? "Show overlay" : "Hide overlay"}">${svg(o.hidden ? "eye-off" : "eye")}</button>
      <button class="btn icon sm ghost pin-toggle" title="${o.static ? "Make trackable" : "Make static"}">${svg("pin")}</button>
      <button class="btn icon sm ghost kebab" title="More">${svg("more")}</button>`;
    d.onclick = (e) => { if (e.target.closest(".swatch,.vis-toggle,.pin-toggle,.kebab")) return; S.activeObj = o.id; changed(); };
    // `change` (not `input`): fire once on commit, so the re-render that follows
    // doesn't tear down the live picker element mid-drag.
    d.querySelector(".color-input").onchange = (e) => { e.stopPropagation(); setColor(o.id, e.target.value); };
    d.querySelector(".vis-toggle").onclick = (e) => { e.stopPropagation(); toggleHidden(o.id, !o.hidden); };
    d.querySelector(".pin-toggle").onclick = (e) => { e.stopPropagation(); toggleStatic(o.id, !o.static); };
    d.querySelector(".kebab").onclick = (e) => { e.stopPropagation(); openKebab(e.currentTarget, o); };
    el.appendChild(d);
  });
  updateMode();
}

/* Prompting or painting with nothing selected shouldn't dead-end on a toast:
   auto-create a placeholder object (label "object", renameable) and select it. */
export async function ensureActiveObject() {
  if (S.activeObj != null) return true;
  if (!S.video) return false;
  try {
    const { obj_id } = await jpost("/api/objects", { video_id: S.video.video_id, label: "object" });
    S.objects.push({ id: obj_id, label: "object", static: false, hidden: false, color: null });
    S.activeObj = obj_id;
    changed();
    toast("Added “object” — rename it in the panel when you're done", "info");
    return true;
  } catch (e) { toast("Could not add object: " + e.message, "error"); return false; }
}

export async function createNewObject() {
  if (!S.video) { toast("Open a clip first", "error"); return; }
  const label = currentLabel();
  if (!label) { toast("Type a label first", "error"); $("labelInput").focus(); return; }
  try {
    const { obj_id } = await jpost("/api/objects", { video_id: S.video.video_id, label });
    S.objects.push({ id: obj_id, label, static: false, hidden: false, color: null });
    S.activeObj = obj_id;
    rememberLabel(label);
    changed();
    toast(`Added “${label}” — now click it on the frame`, "success");
  } catch (e) { toast("Could not add object: " + e.message, "error"); }
}

async function renameObj(oid, label) {
  try {
    await jpost(`/api/objects/${oid}`, { label }, "PATCH");
    const o = objById(oid); if (o) o.label = label;
    changed();
  } catch (e) { toast("Rename failed: " + e.message, "error"); changed(); }
}

async function deleteObj(oid) {
  const o = objById(oid); if (!o) return;
  const cnt = annCount(oid);
  const tail = cnt ? ` and its ${cnt} annotated frame${cnt > 1 ? "s" : ""}` : "";
  if (!confirm(`Delete “${o.label}”${tail}? This can't be undone.`)) return;
  try {
    await jpost(`/api/objects/${oid}`, {}, "DELETE");
    S.objects = S.objects.filter((x) => x.id !== oid);
    S.live.delete(oid); delete S.seedFrameOf[oid];
    for (const k in S.ann.frames) {
      if (!S.ann.frames[k]) continue;
      // Drop the cached mask overlay too: SQLite reuses object ids, and a stale
      // tinted mask would show up under the next object created with this id.
      if (S.ann.frames[k][oid]) invalidateMask(+k, oid);
      delete S.ann.frames[k][oid];
    }
    if (S.activeObj === oid) S.activeObj = S.objects.length ? S.objects[0].id : null;
    changed();
    toast(`Deleted “${o.label}”`, "success");
  } catch (e) { toast("Delete failed: " + e.message, "error"); }
}

async function toggleStatic(oid, on) {
  // A pure mode toggle: no annotations are created or deleted. Static objects
  // get their current box copied onto tracked frames instead of running SAM3.
  try {
    await jpost(`/api/objects/${oid}`, { static: on }, "PATCH");
    const o = objById(oid); if (o) o.static = on;
    if (on) S.live.delete(oid);
    changed();
    toast(on
      ? "Static — tracking now copies its box instead of running SAM3"
      : "Trackable — tracking follows it again from the frame you track from", "success");
  } catch (e) { toast("Static toggle failed: " + e.message, "error"); }
}

async function toggleHidden(oid, hidden) {
  // Display-only: hide/show this object's overlay on every frame. Tracking and
  // export are unaffected.
  try {
    await jpost(`/api/objects/${oid}`, { hidden }, "PATCH");
    const o = objById(oid); if (o) o.hidden = hidden;
    changed();
  } catch (e) { toast("Visibility toggle failed: " + e.message, "error"); }
}

async function setColor(oid, color) {
  try {
    await jpost(`/api/objects/${oid}`, { color }, "PATCH");
    const o = objById(oid); if (o) o.color = color;
    changed();
  } catch (e) { toast("Color change failed: " + e.message, "error"); }
}

/* ---- find all -------------------------------------------------------------- */
export async function findAll(query) {
  if (!S.video) { toast("Open a clip first", "error"); return; }
  // The detection prompt doubles as the object label.
  const label = (query || "").trim() || currentLabel();
  if (!label) { toast("Type a detection prompt first", "error"); return; }
  const btn = $("findBtn"); busy(btn, true);
  try {
    const res = await gpuCall(() => jpost("/api/find_all", {
      video_id: S.video.video_id, frame_idx: S.cur, label, query: label,
      threshold: S.settings.findThreshold, detector: S.settings.findDetector,
    }));
    if (!res.created || !res.created.length) {
      toast(`No “${label}” found on frame ${S.cur} — try a lower threshold`, "info", 5000);
      return;
    }
    S.ann.frames[S.cur] = S.ann.frames[S.cur] || {};
    for (const obj of res.created) {
      S.objects.push({ id: obj.id, label: obj.label, static: false, hidden: false, color: null });
      S.ann.frames[S.cur][obj.id] = { corners: obj.corners, polygon: obj.polygon, origin: "seed" };
      if (obj.seeded) S.live.add(obj.id);
    }
    S.activeObj = res.created[0].id;
    rememberLabel(label);
    changed();
    toast(`Found ${res.created.length} “${label}” — review then track`, "success", 5000);
  } catch (e) { toast("Find all failed: " + e.message, "error", 5000); }
  finally { busy(btn, false); }
}

/* ---- menus ----------------------------------------------------------------- */
export function closeMenus() {
  document.querySelectorAll(".menu").forEach((m) => m.remove());
}

function spawnMenu(anchor, html) {
  closeMenus();
  const m = document.createElement("div");
  m.className = "menu";
  m.innerHTML = html;
  document.body.appendChild(m);
  const r = anchor.getBoundingClientRect();
  let left = Math.min(r.left, window.innerWidth - m.offsetWidth - 8);
  let top = r.bottom + 4;
  if (top + m.offsetHeight > window.innerHeight - 8) top = r.top - m.offsetHeight - 4;
  m.style.left = Math.max(8, left) + "px";
  m.style.top = Math.max(8, top) + "px";
  return m;
}

function openKebab(anchor, o) {
  const m = spawnMenu(anchor, `
    <button class="item" data-a="rename">${svg("pencil")} Rename</button>
    <button class="item" data-a="static">${svg("pin")} ${o.static ? "Make trackable" : "Make static"}</button>
    <button class="item" data-a="seedframe">${svg("target")} Go to seed frame</button>
    <div class="sep"></div>
    <button class="item danger" data-a="delete">${svg("trash")} Delete</button>`);
  m.addEventListener("click", (e) => {
    const a = e.target.closest(".item")?.dataset.a;
    closeMenus();
    if (a === "rename") { editingObj = o.id; changed(); }
    else if (a === "static") toggleStatic(o.id, !o.static);
    else if (a === "seedframe") {
      if (S.seedFrameOf[o.id] !== undefined) goto(S.seedFrameOf[o.id]);
      else toast("This object hasn't been seeded this session", "info");
    } else if (a === "delete") deleteObj(o.id);
  });
}

export function openFindPopover() {
  const m = spawnMenu($("findBtn"), `
    <div class="field-wrap"><label>Detection prompt (English) — becomes the object label</label>
      <input id="findQ" type="text" placeholder="${currentLabel() || "e.g. box"}" autocomplete="off"></div>
    <button class="item" data-a="go">${svg("target")} Find all on frame ${S.cur}</button>`);
  const inp = m.querySelector("#findQ");
  inp.value = currentLabel();
  setTimeout(() => { inp.focus(); inp.select(); }, 40);
  const run = () => { const q = inp.value.trim(); closeMenus(); findAll(q); };
  inp.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); run(); } });
  m.querySelector('[data-a="go"]').onclick = run;
}

export function initObjects() {
  $("newObj").onclick = createNewObject;
  $("labelInput").addEventListener("keydown", (e) => { if (e.key === "Enter") createNewObject(); });
  // stopPropagation: the document listener below would otherwise close the
  // popover on the very click that opened it.
  $("findBtn").onclick = (e) => { e.stopPropagation(); openFindPopover(); };
  document.addEventListener("click", (e) => { if (!e.target.closest(".menu")) closeMenus(); });
}
