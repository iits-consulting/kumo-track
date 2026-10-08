/* SAM3 service readiness + GPU-call gating. The hosted inference service scales
   to zero when idle and takes ~2 min to wake, so /api/sam3/health is polled while
   it reports "loading": a spinner pill shows on the stage and the SAM3 actions
   (Predict, track, find-all) are disabled — everything manual keeps working.
   Polling stops once ready (health traffic wakes the replica, so we don't ping it
   forever), but a stale "ready" is re-verified before each GPU call because the
   replica may have slept again while the user idled (see gpuCall/ensureSam3Ready). */
import { S } from "./state.js";
import { $, toast } from "./dom.js";
import { jget } from "./api.js";
import { syncSeg } from "./tools.js";

const POLL_MS = 3000;
// The replica sleeps after ~5 min idle (KEDA scale-to-zero cooldown), so a
// ready-confirmation older than this is worthless — S.sam3Ready stays stale-true
// and the next GPU call would hang ~4 min against a dead replica. Re-verify first.
const STALE_MS = 4 * 60_000;
let timer = null;
let lastReadyAt = 0; // ms of the last confirmed-ready health answer
let wakeStart = 0;   // ms we first saw "loading" this wake cycle (0 while ready)
let waiters = [];    // resolvers parked in ensureSam3Ready, flushed when ready flips true

/* Reflect S.sam3Ready (and S.propagating for the track buttons) in the UI. */
export function syncSam3Ui() {
  const ready = S.sam3Ready;
  $("sam3Pill").style.display = ready ? "none" : "";
  $("findBtn").disabled = !ready;
  for (const id of ["trackFwd", "trackBack", "trackMore"]) $(id).disabled = !ready || S.propagating;
  syncSeg();
  // Whoever flips us back to not-ready (a gateway error in track.js/gpuCall)
  // gets the poll + pill back for free — one place owns "loading ⇒ polling".
  if (!ready && timer == null) timer = setTimeout(check, POLL_MS);
}

async function check() {
  let ready = false;
  try { ready = (await jget("/api/sam3/health")).status === "ready"; } catch { /* keep loading */ }
  if (ready) {
    lastReadyAt = Date.now();
    wakeStart = 0;
  } else {
    if (!wakeStart) wakeStart = Date.now(); // first not-ready observation of this wake
    const secs = Math.round((Date.now() - wakeStart) / 1000);
    $("sam3PillText").textContent = `GPU starting up… ${secs}s (usually ~2 min)`;
  }
  if (ready !== S.sam3Ready) {
    S.sam3Ready = ready;
    syncSam3Ui();
    if (ready) { const w = waiters; waiters = []; w.forEach((r) => r()); } // let parked callers proceed
  }
  // check() may be called concurrently (poll tick + ensureSam3Ready); clearing the
  // shared timer before re-arming keeps exactly one self-scheduling chain alive.
  clearTimeout(timer);
  timer = ready ? null : setTimeout(check, POLL_MS);
}

/* Resolve once the GPU is confirmed awake. Fast-path a fresh "ready"; otherwise
   probe once and, if still not ready, park until the poll flips it. NOTE: check()'s
   probe of /api/sam3/health is itself what wakes a sleeping replica, so *calling
   this is the warm-up trigger* — there's no separate "start the GPU" request. */
export async function ensureSam3Ready() {
  if (S.sam3Ready && Date.now() - lastReadyAt < STALE_MS) return;
  await check();
  if (S.sam3Ready) return;
  return new Promise((resolve) => waiters.push(resolve));
}

/* Run a GPU-backed action, warming the replica first. If it slept mid-idle the
   call comes back as a gateway error (502/503/504 = replica gone): mark not-ready,
   warm again, retry exactly once. Any other error is the action's own — rethrow. */
export async function gpuCall(fn) {
  await ensureSam3Ready();
  try {
    return await fn();
  } catch (e) {
    if (![502, 503, 504].includes(e.status)) throw e;
    S.sam3Ready = false; syncSam3Ui();
    toast("GPU was asleep — waking it up (~2 min), your action will run automatically", "info", 6000);
    await ensureSam3Ready();
    return await fn();
  }
}

export function initSam3Health() {
  syncSam3Ui(); // start in "loading" until the first health answer
  check();
}
