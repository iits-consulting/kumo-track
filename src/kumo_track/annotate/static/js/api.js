/* Backend calls + the NDJSON propagation stream reader. */
import { S } from "./state.js";

export async function jpost(url, body, method = "POST") {
  const r = await fetch(url, {
    method,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw httpErr(r, (await r.json().catch(() => ({}))).detail);
  return r.json();
}

export async function jget(url) {
  const r = await fetch(url);
  if (!r.ok) throw httpErr(r, (await r.json().catch(() => ({}))).detail);
  return r.json();
}

/* An Error carrying the HTTP status, so callers (sam3.js) can spot a sleeping
   GPU replica (gateway 502/503/504) and warm it instead of failing hard. */
function httpErr(r, detail) {
  const err = new Error(detail || r.statusText);
  err.status = r.status;
  return err;
}

export function frameUrl(i) {
  // v=2: the pre-pts-seek server sent path-dependent pictures under these (immutable) URLs.
  return `/api/frame/${S.video.video_id}/${i}.jpg?q=80&v=2`;
}

/* /api/mask is mutable — re-segmenting or brush-editing rewrites it — but its URL
   is otherwise stable, so the browser serves a stale PNG from its (in-memory) image
   cache on re-fetch. invalidateMask() bumps this per-(frame,object) version so each
   changed mask gets a unique URL and is fetched fresh. */
const maskVer = new Map(); // `${frame}:${objId}` -> int
export function bumpMaskVer(objId, frame) {
  const k = `${frame}:${objId}`;
  maskVer.set(k, (maskVer.get(k) || 0) + 1);
}
export function maskUrl(objId, frame) {
  const v = maskVer.get(`${frame}:${objId}`) || 0;
  return `/api/mask?video_id=${S.video.video_id}&object_id=${objId}&frame_idx=${frame}&v=${v}`;
}

/* POST a propagation and yield each parsed NDJSON line. `signal` lets the caller
   cancel; the server stops cleanly on client disconnect. Throws on {error:...}. */
export async function* propagateStream(body, signal) {
  const r = await fetch("/api/propagate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!r.ok) throw httpErr(r, (await r.json().catch(() => ({}))).detail);
  const reader = r.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let nl;
    while ((nl = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, nl);
      buf = buf.slice(nl + 1);
      if (!line.trim()) continue;
      let o;
      try { o = JSON.parse(line); } catch { continue; }
      if (o.error) throw new Error(o.error);
      yield o;
    }
  }
}

export function uploadFile(file, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/upload");
    xhr.upload.onprogress = (e) => { if (e.lengthComputable) onProgress(e.loaded / e.total); };
    xhr.onload = () => {
      try {
        const j = JSON.parse(xhr.responseText || "{}");
        xhr.status < 300 ? resolve(j) : reject(new Error(j.detail || xhr.statusText));
      } catch (err) { reject(err); }
    };
    xhr.onerror = () => reject(new Error("network error"));
    const fd = new FormData();
    fd.append("file", file);
    xhr.send(fd);
  });
}
