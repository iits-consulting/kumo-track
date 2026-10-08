/* Bootstrap: wire modules, header actions, overlays, settings, export, keyboard. */
import { S, bus } from "./state.js";
import { $, svg, toast, busy, initTooltips } from "./dom.js";
import { jpost, jget, uploadFile } from "./api.js";
import { setupCanvas, render, cv } from "./canvas.js";
import { goto, renderTimeline, initTimeline } from "./timeline.js";
import {
  renderObjs, refreshLabelOptions, initObjects, closeMenus,
} from "./objects.js";
import {
  initTools, setMode, setTool, activateBrush, activatePoly, doSegment, undoActive, redoActive, exitMaskEdit, inMaskEdit, bumpBrush, setCtrlHeld, syncSeg,
} from "./tools.js";
import { initTrack, startTrack, stopTrack, isTracking } from "./track.js";
import { initSam3Health } from "./sam3.js";
import { ACTIONS, getCombos, comboLabel, matchAction, normalizeCombo, rebind, resetKeymap } from "./keymap.js";

/* ---- buttons that get icons from JS ---------------------------------------- */
$("settingsBtn").innerHTML = svg("settings");
$("helpBtn").innerHTML = svg("help");
document.querySelectorAll(".close[data-close]").forEach((b) => { b.innerHTML = svg("x"); });

/* ---- full UI refresh on `changed` ------------------------------------------ */
function refreshAll() {
  if (!S.video) return;
  renderObjs();
  renderTimeline();
  refreshLabelOptions();
  syncSeg();
  render();
}
bus.addEventListener("changed", refreshAll);

/* ---- overlays -------------------------------------------------------------- */
function openOverlay(id) { $(id).classList.add("show"); }
function closeOverlay(id) {
  $(id).classList.remove("show");
  // Leaving the help modal mid-recording must tear down the capturing
  // keydown listener, or the next keystroke anywhere gets swallowed/rebound.
  if (stopRecording) { stopRecording(); renderKeymap(); }
}
document.querySelectorAll(".close[data-close]").forEach((b) => {
  b.onclick = () => closeOverlay(b.dataset.close);
});
document.querySelectorAll(".overlay").forEach((ov) => {
  ov.addEventListener("click", (e) => { if (e.target === ov) closeOverlay(ov.id); });
});

/* ---- open / switch clip ---------------------------------------------------- */
async function refreshVideos(select) {
  try {
    const { videos, location } = await jget("/api/videos");
    const sel = $("videoSel");
    sel.innerHTML = "";
    videos.forEach((v) => { const o = document.createElement("option"); o.textContent = v; sel.appendChild(o); });
    if (select && videos.includes(select)) sel.value = select;
    $("openHint").textContent = videos.length ? "" : `No clips in ${location} — upload one to start.`;
    return videos;
  } catch { $("openHint").innerHTML = `<span class="warn">Could not reach the server.</span>`; return []; }
}

function enterWorkspace() {
  for (const id of ["toolstrip", "stageHud", "bottombar"]) $(id).style.display = "";
  $("panel").style.display = "flex";
  $("stageEmpty").style.display = "none";
}

async function openClip() {
  const name = $("videoSel").value;
  if (!name) { $("openHint").innerHTML = `<span class="warn">Pick or upload a clip first.</span>`; return; }
  const btn = $("openGo"); busy(btn, true, "Decoding…");
  $("openHint").textContent = "Opening…";
  try {
    const v = await jpost("/api/open", { name, stride: S.settings.stride });
    S.video = v;
    const a = v.annotations || { frames: {}, objects: [] };
    S.objects = (a.objects || []).map((o) => ({
      id: o.id, label: o.label, static: o.static, hidden: o.hidden, color: o.color,
    }));
    const frames = {}; for (const k in (a.frames || {})) frames[+k] = a.frames[k];
    S.ann = { frames };
    S.activeObj = S.objects.length ? S.objects[0].id : null;
    S.live = new Set(); S.seedFrameOf = {}; S.edit = null;
    S.seed = { points: [], labels: [], box: null, dragStart: null };
    setupCanvas();
    $("scrub").max = v.n_frames - 1;
    $("clipBtnLabel").textContent = name;
    $("clipBtnMeta").textContent = ` · ${v.n_frames}f · ${v.width}×${v.height}`;
    enterWorkspace();
    closeOverlay("openModal");
    goto(0);
    activateBrush(); // manual mode with the brush armed is the working default
    renderObjs(); renderTimeline(); refreshLabelOptions(); render();
    toast(`Opened ${name} — ${v.n_frames} frames`, "success");
    if (v.truncated_at != null) {
      toast(`⚠ ${name} looks corrupt — decoding stopped near source frame ${v.truncated_at}. `
        + `Only ${v.n_frames} frame(s) recovered; re-encode to fix.`, "error", 9000);
    }
    if (S.objects.length) toast(`${S.objects.length} object(s) loaded — re-seed before tracking`, "info", 4200);
  } catch (e) {
    $("openHint").innerHTML = `<span class="warn">Open failed: ${e.message}</span>`;
    toast("Open failed: " + e.message, "error");
  } finally { busy(btn, false); }
}

$("clipBtn").onclick = () => { refreshVideos(S.video ? S.video.name : null); openOverlay("openModal"); };
$("openGo").onclick = openClip;
$("uploadBtn").onclick = () => $("fileInput").click();
$("fileInput").onchange = async () => {
  const f = $("fileInput").files[0]; if (!f) return;
  try {
    const { name } = await uploadFile(f, (p) => $("openHint").textContent = `Uploading ${f.name}… ${Math.round(p * 100)}%`);
    await refreshVideos(name);
    $("openHint").innerHTML = `Uploaded <b>${name}</b> — click <b>Open clip</b>.`;
    toast(`Uploaded ${name}`, "success");
  } catch (e) { $("openHint").innerHTML = `<span class="warn">Upload failed: ${e.message}</span>`; toast("Upload failed: " + e.message, "error"); }
  $("fileInput").value = "";
};

/* ---- export ---------------------------------------------------------------- */
$("exportBtn").onclick = () => {
  if (!S.video) { toast("Open a clip first", "error"); return; }
  closeMenus();
  $("expInfo").textContent = "";
  openOverlay("exportModal");
  setTimeout(() => $("expName").focus(), 40);
};
async function runExport() {
  const out_name = $("expName").value.trim();
  if (!out_name) { $("expInfo").innerHTML = `<span class="warn">Enter a dataset name.</span>`; $("expName").focus(); return; }
  const include_images = document.querySelector('input[name="expMode"]:checked').value === "full";
  const btn = $("expGo"); busy(btn, true, "Exporting…");
  try {
    const res = await jpost("/api/export", { video_id: S.video.video_id, out_name, include_images });
    $("expInfo").innerHTML = `✓ ${res.n_boxes} boxes across ${res.n_images} frame${res.n_images === 1 ? "" : "s"}`
      + `${include_images ? " (images + labels.json)" : " (labels.json)"} → <b>${res.dir}</b>`;
    toast(`Exported ${res.n_boxes} boxes across ${res.n_images} frame(s)`, "success", 4500);
  } catch (err) {
    $("expInfo").innerHTML = `<span class="warn">${err.message}</span>`;
    toast("Export failed: " + err.message, "error", 5000);
  } finally { busy(btn, false); }
}
$("expGo").onclick = runExport;
$("expName").addEventListener("keydown", (e) => { if (e.key === "Enter") runExport(); });

/* ---- settings -------------------------------------------------------------- */
(function initSettings() {
  $("settingsBtn").onclick = () => openOverlay("settingsModal");
  const colSel = $("setColorMode");
  colSel.value = S.settings.colorByCategory ? "category" : "instance";
  colSel.onchange = () => {
    S.settings.colorByCategory = colSel.value === "category";
    localStorage.setItem("annotate.colorByCategory", S.settings.colorByCategory);
    if (S.video) refreshAll();
  };
  const stride = $("setStride"), strideVal = $("setStrideVal");
  stride.value = S.settings.stride; strideVal.textContent = S.settings.stride;
  stride.oninput = () => { strideVal.textContent = stride.value; };
  stride.onchange = () => {
    S.settings.stride = parseInt(stride.value, 10);
    localStorage.setItem("annotate.stride", S.settings.stride);
    if (S.video) toast("Stride changed — re-open the clip to apply", "info", 4000);
  };
  const det = $("setFindDetector");
  det.value = S.settings.findDetector;
  det.onchange = () => { S.settings.findDetector = det.value; localStorage.setItem("annotate.findDetector", det.value); };
  const thr = $("setFindThreshold"), thrVal = $("setFindThresholdVal");
  thr.value = S.settings.findThreshold.toFixed(2); thrVal.textContent = S.settings.findThreshold.toFixed(2);
  thr.oninput = () => {
    S.settings.findThreshold = parseFloat(thr.value);
    thrVal.textContent = S.settings.findThreshold.toFixed(2);
    localStorage.setItem("annotate.findThreshold", S.settings.findThreshold);
  };
})();
$("helpBtn").onclick = () => openOverlay("helpModal");
$("zoomPill").onclick = () => { S.view = { scale: 1, ox: 0, oy: 0 }; $("zoomPill").style.display = "none"; render(); };

/* ---- keyboard -------------------------------------------------------------- */
window.addEventListener("keydown", (e) => {
  if (e.key === "Control") { setCtrlHeld(true); if (S.video) cv.style.cursor = "grab"; return; }
  if (/^(INPUT|SELECT|TEXTAREA)$/.test(e.target.tagName)) return;
  const action = matchAction(e);
  if (action === "help") { $("helpModal").classList.toggle("show"); return; }
  if (e.key === "Escape") {
    if (isTracking()) { stopTrack(); return; }
    if (inMaskEdit()) { exitMaskEdit(true); return; }
    const open = document.querySelector(".overlay.show");
    if (open) { open.classList.remove("show"); return; }
    closeMenus();
  }
  if (!S.video) return;
  switch (action) {
    case "resetZoom": S.view = { scale: 1, ox: 0, oy: 0 }; $("zoomPill").style.display = "none"; render(); e.preventDefault(); break;
    case "undo": undoActive(); e.preventDefault(); break;
    case "redo": redoActive(); e.preventDefault(); break;
    case "prevFrame": case "nextFrame": {
      // Shift jumps 10 — unless the bound combo itself already spends shift
      // (rebound to a "shift+…" combo), in which case there's no "extra" shift left to detect.
      const usesShift = getCombos(action).some((c) => c.includes("shift"));
      const jump = e.shiftKey && !usesShift ? 10 : 1;
      goto(S.cur + (action === "nextFrame" ? jump : -jump));
      e.preventDefault();
      break;
    }
    case "manualMode": setMode("manual"); break;
    case "aiPoint": setMode("ai"); setTool("pos"); break;
    case "aiNegPoint": setMode("ai"); setTool("neg"); break;
    case "aiBox": setMode("ai"); setTool("box"); break;
    case "brush": activateBrush(); break;
    case "polygon": activatePoly(); break;
    case "brushSmaller": bumpBrush(-4); break;
    case "brushLarger": bumpBrush(4); break;
    case "finishOrPredict": if (inMaskEdit()) exitMaskEdit(true); else doSegment(); break;
    case "trackForward": startTrack({ direction: "fwd", n_frames: Math.max(1, parseInt($("trackN").value || "50", 10)) }); e.preventDefault(); break;
  }
});
window.addEventListener("keyup", (e) => { if (e.key === "Control") { setCtrlHeld(false); if (S.video) cv.style.cursor = S.tool === "select" ? "default" : "crosshair"; } });

/* ---- keyboard shortcut editor (help modal) ---------------------------------- */
function renderKeymap() {
  const list = $("keymapList");
  list.innerHTML = "";
  ACTIONS.forEach((a) => {
    const row = document.createElement("div"); row.className = "li";
    const k = document.createElement("span");
    k.className = "k kbd-edit"; k.title = "Click to record a new shortcut";
    k.innerHTML = getCombos(a.id).map((c) => comboLabel(c).map((l) => `<span class="kbd">${l}</span>`).join("")).join(" / ");
    k.onclick = () => startRecording(a.id, k);
    const desc = document.createElement("span"); desc.textContent = a.desc;
    row.append(k, desc);
    list.appendChild(row);
  });
}
let stopRecording = null; // active recorder teardown — one recording at a time
function startRecording(id, chipsEl) {
  // Clicking any row while recording cancels it (a re-render would orphan the
  // old row's "press keys…" chip otherwise); click again to record that row.
  if (stopRecording) { stopRecording(); renderKeymap(); return; }
  chipsEl.innerHTML = `<span class="kbd recording">press keys…</span>`;
  const onKey = (e) => {
    e.preventDefault(); e.stopPropagation();
    // A chord fires a keydown for the bare modifier first — keep listening for
    // the real key, or Ctrl+Z would record as "ctrl+control".
    if (["Control", "Shift", "Alt", "Meta"].includes(e.key)) return;
    stopRecording();
    if (e.key === "Escape") { renderKeymap(); return; }
    const conflict = rebind(id, normalizeCombo(e));
    if (conflict) toast(`Already bound to “${ACTIONS.find((a) => a.id === conflict).desc}” — pick another key`, "error");
    renderKeymap();
  };
  stopRecording = () => { window.removeEventListener("keydown", onKey, true); stopRecording = null; };
  window.addEventListener("keydown", onKey, { capture: true });
}
$("keymapReset").onclick = () => { if (stopRecording) stopRecording(); resetKeymap(); renderKeymap(); toast("Shortcuts reset to defaults", "info"); };
renderKeymap();

/* ---- init ------------------------------------------------------------------ */
jget("/api/config").then((c) => {
  if (c && c.edit_tool) S.config.editTool = c.edit_tool;
  const b = $("activeEditTool");
  if (b) { b.textContent = S.config.editTool; b.hidden = false; }
}).catch(() => {});
initTooltips();
initTimeline();
initObjects();
initTools();
initTrack();
initSam3Health();

/* Deep link (?video=<name>&stride=<n>): open that clip straight away instead of
   the picker, with a loading state on the stage while it decodes. Falls back to
   the picker with a toast if the name isn't available (e.g. a stale link to a
   deleted blob). A deep link also means the user came from the hub's clip grid,
   so a "Finished" button appears to take them back to it. */
(async function initOpen() {
  const params = new URLSearchParams(location.search);
  const name = params.get("video");
  const stride = parseInt(params.get("stride") || "", 10);
  if (name && !isNaN(stride)) S.settings.stride = stride;
  if (name) {
    const empty = $("stageEmpty");
    empty.innerHTML = `<svg class="ic spin" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12a9 9 0 1 1-6.219-8.56"/></svg> Opening <b></b>…`;
    empty.querySelector("b").textContent = name; // URL input — never innerHTML it
    // ponytail: history.back() is the whole "return to the clip list" story — on
    // this flow the grid is always the previous history entry; pass an explicit
    // &back= URL from the hub if that ever stops holding.
    $("finishBtn").style.display = "";
    $("finishBtn").onclick = () => history.back();
  }
  const videos = await refreshVideos(name || null);
  if (name && videos.includes(name)) {
    $("videoSel").value = name;
    await openClip();   // reads $("videoSel").value + S.settings.stride; closes the modal itself
    if (S.video) return;              // opened — enterWorkspace() hid the stage message
  } else if (name) {
    toast(`Clip “${name}” not found — pick one below.`, "error", 6000);
  }
  $("stageEmpty").innerHTML = "No clip open — click <b>Open clip</b> to begin.";
  openOverlay("openModal");
})();
