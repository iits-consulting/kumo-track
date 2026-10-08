/* Propagation: track N frames / whole clip, live progress on the timeline, stop. */
import { S, changed, annCount } from "./state.js";
import { $, toast } from "./dom.js";
import { propagateStream } from "./api.js";
import { render, invalidateMask } from "./canvas.js";
import { renderTimeline } from "./timeline.js";
import { ensureSam3Ready, syncSam3Ui } from "./sam3.js";

let controller = null;

function nonStaticWithBox() {
  const f = S.ann.frames[S.cur] || {};
  return S.objects.filter((o) => !o.static && f[o.id] && f[o.id].corners);
}
function nonStaticWithoutBox() {
  const f = S.ann.frames[S.cur] || {};
  return S.objects.filter((o) => !o.static && !(f[o.id] && f[o.id].corners));
}

export function isTracking() { return S.propagating; }

export function stopTrack() {
  if (controller) controller.abort();
}

function setTracking(on) {
  S.propagating = on;
  $("stopBtn").style.display = on ? "" : "none";
  const off = on || !S.sam3Ready; // stay disabled while SAM3 is still waking
  $("trackFwd").disabled = off;
  $("trackBack").disabled = off;
  $("trackMore").disabled = off;
}

export async function startTrack({ direction = "fwd", n_frames = null, full_clip = false }) {
  if (!S.video || S.propagating) return;
  // Static objects don't need a box on this frame — the backend copies their box
  // from the nearest annotated frame onto every tracked frame. The *selected*
  // object doesn't either: the backend seeds it from its nearest annotated frame.
  const pinned = S.objects.filter((o) => o.static && annCount(o.id) > 0);
  const active = S.objects.find((o) => o.id === S.activeObj);
  const activeCanSeed = !!active && !active.static && annCount(active.id) > 0;
  if (!nonStaticWithBox().length && !pinned.length && !activeCanSeed) {
    toast(`Nothing to track from frame ${S.cur} — segment an object first`, "error", 4500);
    return;
  }
  const missing = nonStaticWithoutBox().filter((o) => !(activeCanSeed && o.id === active.id));
  if (activeCanSeed && !(S.ann.frames[S.cur] || {})[active.id]?.corners) {
    toast(`“${active.label}” has no box here — seeding it from its nearest annotated frame`, "info", 4000);
  }
  if (missing.length) {
    toast(`${missing.length} unselected object${missing.length > 1 ? "s have" : " has"} no box on this frame and won't track`, "info", 4000);
  }

  // Warm a possibly-slept replica before streaming. Deliberately NO auto-retry on
  // the stream itself: partial frames may already be applied, so a blind restart
  // would duplicate/clobber work — a gateway error is surfaced in the catch instead.
  await ensureSam3Ready();
  if (S.propagating) return; // a second click may have parked on ensure too

  controller = new AbortController();
  setTracking(true);
  $("propInfo").textContent = "Tracking…";
  const startFrame = S.cur;
  let nFrames = 0, nHit = 0, nBoxes = 0;
  try {
    for await (const o of propagateStream(
      { video_id: S.video.video_id, start_frame_idx: startFrame, direction, n_frames, full_clip,
        active_obj: S.activeObj },
      controller.signal,
    )) {
      const fi = o.frame_idx;
      S.ann.frames[fi] = S.ann.frames[fi] || {};
      const echo = fi === startFrame; // copied seed shown on the start frame — not a tracked frame
      let hit = false;
      for (const oid in o.boxes) {
        S.ann.frames[fi][+oid] = o.boxes[oid];
        if (S.config.editTool === "brush") invalidateMask(fi, +oid); // mask changed on this frame
        if (!echo && o.boxes[oid] && o.boxes[oid].corners) { nBoxes++; hit = true; }
      }
      if (echo) { if (fi === S.cur) render(); continue; }
      nFrames++; if (hit) nHit++;
      $("propInfo").textContent = `${nFrames} frames · ${nHit} with boxes…`;
      if (nFrames % 5 === 0) renderTimeline();
      if (fi === S.cur) render();
    }
    renderTimeline();
    $("propInfo").innerHTML = `<span class="ok">✓ ${nHit}/${nFrames} frames · ${nBoxes} boxes</span> — scrub to review`;
    changed();
    toast(`Tracked ${nBoxes} boxes across ${nHit} frames`, "success");
  } catch (e) {
    renderTimeline(); changed();
    if (e.name === "AbortError") {
      $("propInfo").innerHTML = `<span>Stopped — ${nBoxes} boxes kept</span>`;
      toast("Tracking stopped", "info");
    } else if ([502, 503, 504].includes(e.status)) {
      // Replica slept/restarted mid-stream. No blind retry (see startTrack); reset
      // readiness so the pill + poll come back and the user can re-run cleanly.
      S.sam3Ready = false; syncSam3Ui();
      $("propInfo").innerHTML = `<span class="warn">GPU service restarted — hit Track again</span>`;
      toast("GPU service restarted — hit Track again", "error", 5000);
    } else {
      $("propInfo").innerHTML = `<span class="warn">${e.message}</span>`;
      toast("Track failed: " + e.message, "error", 5000);
    }
  } finally {
    setTracking(false);
    controller = null;
  }
}

export function initTrack() {
  const n = () => Math.max(1, parseInt($("trackN").value || "50", 10));
  $("trackN").value = S.settings.trackN;
  $("trackN").onchange = () => {
    S.settings.trackN = n();
    localStorage.setItem("annotate.trackN", S.settings.trackN);
  };
  $("trackFwd").onclick = () => startTrack({ direction: "fwd", n_frames: n() });
  $("trackBack").onclick = () => startTrack({ direction: "rev", n_frames: n() });
  $("trackMore").onclick = (e) => {
    // whole-clip popover; stopPropagation so the document click listener
    // (objects.js) doesn't close it on the opening click
    e.stopPropagation();
    document.querySelectorAll(".menu").forEach((m) => m.remove());
    const m = document.createElement("div");
    m.className = "menu";
    m.innerHTML = `<button class="item" data-a="full">Track whole clip (both directions)</button>
      <button class="item" data-a="fwdend">Track forward to end</button>
      <button class="item" data-a="revstart">Track back to start</button>`;
    document.body.appendChild(m);
    const r = e.currentTarget.getBoundingClientRect();
    m.style.left = Math.max(8, Math.min(r.left, window.innerWidth - m.offsetWidth - 8)) + "px";
    m.style.top = (r.top - m.offsetHeight - 4) + "px";
    m.addEventListener("click", (ev) => {
      const a = ev.target.closest(".item")?.dataset.a;
      m.remove();
      if (a === "full") startTrack({ full_clip: true });
      else if (a === "fwdend") startTrack({ direction: "fwd", n_frames: null });
      else if (a === "revstart") startTrack({ direction: "rev", n_frames: null });
    });
  };
  $("stopBtn").onclick = stopTrack;
}
