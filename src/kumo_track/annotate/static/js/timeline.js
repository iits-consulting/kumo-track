/* Timeline: scrubber, coverage ticks, playhead, and frame navigation (goto). */
import { S, bus } from "./state.js";
import { $ } from "./dom.js";
import { showFrame, render } from "./canvas.js";

export function goto(i) {
  if (!S.video) return;
  S.cur = Math.max(0, Math.min(S.video.n_frames - 1, i));
  bus.dispatchEvent(new Event("frame")); // an active brush edit re-targets (tools.js)
  $("scrub").value = S.cur;
  const src = S.video.source_indices ? ` <span class="src">src ${S.video.source_indices[S.cur]}</span>` : "";
  $("frameInfo").innerHTML = `frame ${S.cur} / ${S.video.n_frames - 1}` + src;
  $("prev").disabled = S.cur <= 0;
  $("next").disabled = S.cur >= S.video.n_frames - 1;
  showFrame(S.cur);
  positionPlayhead();
  render();
}

export function positionPlayhead() {
  if (!S.video) return;
  const n = Math.max(1, S.video.n_frames - 1);
  $("play").style.left = (S.cur / n * 100) + "%";
}

export function renderTimeline() {
  if (!S.video) return;
  const cov = $("cov");
  cov.querySelectorAll(".tick").forEach((n) => n.remove());
  const n = Math.max(1, S.video.n_frames - 1);
  for (const k in S.ann.frames) {
    const perobj = S.ann.frames[k];
    if (!perobj) continue;
    const fi = +k;
    const hasActive = S.activeObj != null && perobj[S.activeObj] && perobj[S.activeObj].corners;
    const hasAny = Object.values(perobj).some((v) => v && v.corners);
    if (!hasAny) continue;
    const t = document.createElement("div");
    t.className = "tick " + (hasActive ? "active" : "other");
    t.style.left = (fi / n * 100) + "%";
    t.style.background = hasActive ? "var(--fire-red)" : "var(--sea-green)";
    cov.appendChild(t);
  }
  positionPlayhead();
}

export function initTimeline() {
  $("cov").addEventListener("click", (e) => {
    if (!S.video) return;
    const r = $("cov").getBoundingClientRect();
    const frac = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width));
    goto(Math.round(frac * (S.video.n_frames - 1)));
  });
  $("prev").onclick = () => goto(S.cur - 1);
  $("next").onclick = () => goto(S.cur + 1);
  $("scrub").oninput = (e) => goto(+e.target.value);
}
