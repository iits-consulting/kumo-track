/* DOM helpers, inline icons, toasts, button-busy state, and floating tooltips. */
export const $ = (id) => document.getElementById(id);

export const ICONS = {
  help: `<circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><line x1="12" y1="17" x2="12.01" y2="17"/>`,
  settings: `<path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"/><circle cx="12" cy="12" r="3"/>`,
  x: `<path d="M18 6 6 18M6 6l12 12"/>`,
  check: `<path d="M20 6 9 17l-5-5"/>`,
  pencil: `<path d="M12 20h9"/><path d="M16.5 3.5a2.12 2.12 0 0 1 3 3L7 19l-4 1 1-4Z"/>`,
  trash: `<path d="M3 6h18M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>`,
  info: `<circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/>`,
  alert: `<path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>`,
  loader: `<path d="M21 12a9 9 0 1 1-6.219-8.56"/>`,
  pin: `<path d="M12 17v5M9 10.76V6a2 2 0 0 1 2-2h2a2 2 0 0 1 2 2v4.76l1.5 2.24H7.5z"/>`,
  target: `<circle cx="12" cy="12" r="3"/><circle cx="12" cy="12" r="9"/>`,
  more: `<circle cx="12" cy="12" r="1"/><circle cx="19" cy="12" r="1"/><circle cx="5" cy="12" r="1"/>`,
  eye: `<path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7z"/><circle cx="12" cy="12" r="3"/>`,
  "eye-off": `<path d="M9.88 9.88a3 3 0 0 0 4.24 4.24"/><path d="M10.73 5.08A10.43 10.43 0 0 1 12 5c7 0 10 7 10 7a13.16 13.16 0 0 1-1.67 2.68"/><path d="M6.61 6.61A13.526 13.526 0 0 0 2 12s3 7 10 7a9.74 9.74 0 0 0 5.39-1.61"/><line x1="2" y1="2" x2="22" y2="22"/>`,
};
export const svg = (name, extra = "") =>
  `<svg class="ic ${extra}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${ICONS[name] || ""}</svg>`;

export function toast(msg, type = "info", ms = 3200) {
  const el = document.createElement("div");
  el.className = `toast ${type}`;
  const icon = type === "success" ? "check" : type === "error" ? "alert" : "info";
  el.innerHTML = svg(icon) + `<span>${msg}</span>`;
  $("toasts").appendChild(el);
  setTimeout(() => {
    el.style.opacity = "0";
    el.style.transition = "opacity .25s";
    setTimeout(() => el.remove(), 260);
  }, ms);
}

export function busy(btn, on, labelHtml) {
  if (on) {
    btn.dataset.html = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = svg("loader", "spin") + (labelHtml ? `<span>${labelHtml}</span>` : "");
  } else {
    btn.disabled = false;
    if (btn.dataset.html !== undefined) btn.innerHTML = btn.dataset.html;
  }
}

/* Floating tooltips driven by title=, immune to overflow-scroll clipping. */
export function initTooltips() {
  const tipEl = $("tip");
  let host = null;
  function show(el) {
    const txt = el.getAttribute("data-tip") || el.getAttribute("title") || el.__t;
    if (!txt) return hide();
    if (el.getAttribute("title") != null) { el.__t = el.getAttribute("title"); el.removeAttribute("title"); }
    tipEl.textContent = txt; tipEl.style.display = "block";
    const r = el.getBoundingClientRect(), t = tipEl.getBoundingClientRect();
    const below = (r.top - t.height - 8) < 6;
    tipEl.style.top = (below ? r.bottom + 8 : r.top - t.height - 8) + "px";
    tipEl.style.left = Math.max(6, Math.min(r.left + r.width / 2 - t.width / 2, window.innerWidth - t.width - 6)) + "px";
    requestAnimationFrame(() => tipEl.classList.add("show"));
  }
  function hide() { tipEl.classList.remove("show"); tipEl.style.display = "none"; }
  document.addEventListener("mouseover", (e) => {
    const h = e.target.closest ? e.target.closest("[title],[data-tip]") : null;
    if (h === host) return;
    if (host && host.__t != null) { host.setAttribute("title", host.__t); host.__t = null; }
    host = h; host ? show(host) : hide();
  });
}
