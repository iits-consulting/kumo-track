/* Rebindable keyboard shortcuts: action registry, combo normalization, and
   localStorage persistence. Escape (stop/finish/close chain) is hardcoded in
   main.js and deliberately absent here — it can't be rebound or stolen. */

const STORAGE_KEY = "annotate.keymap";

// One entry per rebindable shortcut. `combos` are the defaults — order is
// display order in the help modal. Rebinding overwrites the whole array.
export const ACTIONS = [
  { id: "prevFrame", combos: ["arrowleft"], desc: "Previous frame · Shift jumps 10" },
  { id: "nextFrame", combos: ["arrowright"], desc: "Next frame · Shift jumps 10" },
  { id: "manualMode", combos: ["v"], desc: "Manual mode (brush)" },
  { id: "aiPoint", combos: ["1"], desc: "AI mode: point+" },
  { id: "aiNegPoint", combos: ["2"], desc: "AI mode: point−" },
  { id: "aiBox", combos: ["3"], desc: "AI mode: prompt box" },
  { id: "brush", combos: ["b"], desc: "Brush tool" },
  { id: "polygon", combos: ["p"], desc: "Polygon tool" },
  { id: "brushSmaller", combos: ["["], desc: "Smaller brush" },
  { id: "brushLarger", combos: ["]"], desc: "Larger brush" },
  { id: "undo", combos: ["ctrl+z", "u"], desc: "Undo" },
  { id: "redo", combos: ["ctrl+y", "ctrl+shift+z"], desc: "Redo" },
  { id: "finishOrPredict", combos: ["enter"], desc: "Finish edit (manual) / predict (AI)" },
  { id: "trackForward", combos: ["space"], desc: "Track forward N frames" },
  { id: "resetZoom", combos: ["ctrl+0"], desc: "Reset zoom" },
  { id: "help", combos: ["?"], desc: "Toggle this help" },
];

let overrides = loadOverrides();

function loadOverrides() {
  const ids = new Set(ACTIONS.map((a) => a.id));
  try {
    const stored = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
    const out = {};
    for (const id in stored) if (ids.has(id)) out[id] = stored[id];
    return out;
  } catch { return {}; }
}
function saveOverrides() { localStorage.setItem(STORAGE_KEY, JSON.stringify(overrides)); }

export function getCombos(id) {
  return overrides[id] || ACTIONS.find((a) => a.id === id).combos;
}

function comboOwner(combo) {
  for (const a of ACTIONS) if (getCombos(a.id).includes(combo)) return a.id;
  return null;
}

/* Normalize a keydown into a canonical combo string: ctrl+alt+shift prefix (fixed
   order), then the key. Shift only prefixes when it doesn't already change which
   character was produced (arrows, letters) — punctuation/digits shift produces
   (e.g. "?", "!") keep their own character with no redundant "shift+". */
function buildCombo(e, includeShift) {
  let key = e.key === " " ? "space" : e.key.toLowerCase();
  const parts = [];
  if (e.ctrlKey) parts.push("ctrl");
  if (e.altKey) parts.push("alt");
  if (includeShift && e.shiftKey && (key.length > 1 || /^[a-z]$/.test(key))) parts.push("shift");
  parts.push(key);
  return parts.join("+");
}
export const normalizeCombo = (e) => buildCombo(e, true);

// Exact match first; if that fails and Shift was held, retry without it — this is
// what makes Shift+Arrow still hit prevFrame/nextFrame (jump-10 is a *behavior* of
// those actions, not a separate binding) without special-casing arrow keys here.
export function matchAction(e) {
  return comboOwner(buildCombo(e, true)) || (e.shiftKey ? comboOwner(buildCombo(e, false)) : null);
}

const DISPLAY = { arrowleft: "←", arrowright: "→", arrowup: "↑", arrowdown: "↓", space: "Space", enter: "Enter", ctrl: "Ctrl", alt: "Alt", shift: "Shift" };
export function comboLabel(combo) {
  return combo.split("+").map((p) => DISPLAY[p] || p.toUpperCase());
}

// Returns the id of the action already owning `combo` if it differs from `id`
// (caller should toast and not rebind), else applies the rebind and returns null.
export function rebind(id, combo) {
  const owner = comboOwner(combo);
  if (owner && owner !== id) return owner;
  overrides[id] = [combo];
  saveOverrides();
  return null;
}

export function resetKeymap() { overrides = {}; localStorage.removeItem(STORAGE_KEY); }
