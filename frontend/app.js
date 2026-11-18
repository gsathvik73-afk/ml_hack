const D = window.DEMO;
const $ = id => document.getElementById(id);
const pct = x => (100 * x).toFixed(2) + "%";
const toLL = s => [[s[0], s[1]], [s[2], s[3]]];
const COLOR = { match: "#1f6f4a", miss: "#c2410c", ref: "#3d5a80", ping: "#221f1a", flagged: "#a16207" };
const FOLLOW_ZOOM = 16;

// ---- masthead and metrics table (all 10 trips) ----
const O = D.overall;
$("model").textContent = D.model + " | " + O.valid_points.toLocaleString() + " valid pings";
$("overall").innerHTML = [
  ["Top-1", pct(O.top1)], ["Macro F1", pct(O.macro_f1)], ["Log loss", O.log_loss.toFixed(3)],
  ["Continuity", pct(O.route_continuity)], ["Inference / trip", O.prediction_ms_per_trip.toFixed(0) + " ms"],
].map(([l, v]) => `<div><dt>${l}</dt><dd>${v}</dd></div>`).join("");

$("rows").innerHTML = D.trips.map(t => `
  <tr data-id="${t.id}"><td>${t.id}</td><td class="n">${t.points.toLocaleString()}</td>
  <td class="n">${pct(t.top1)}</td><td class="n">${pct(t.macro_f1)}</td>
  <td class="n">${t.log_loss.toFixed(3)}</td><td class="n">${pct(t.route_continuity)}</td></tr>`).join("");
$("total").innerHTML = `<tr><td>All trips</td><td class="n">${O.valid_points.toLocaleString()}</td>
  <td class="n">${pct(O.top1)}</td><td class="n">${pct(O.macro_f1)}</td>
  <td class="n">${O.log_loss.toFixed(3)}</td><td class="n">${pct(O.route_continuity)}</td></tr>`;
$("rows").onclick = e => { const r = e.target.closest("tr"); if (r) loadTrip(r.dataset.id); };

// ---- map: muted basemap so the route layers carry the colour ----
const map = L.map("map", { preferCanvas: true });
L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 19, attribution: "&copy; OpenStreetMap contributors",
}).addTo(map);
const layers = {
  "Road network (candidates)": L.layerGroup(), "Original route (full)": L.layerGroup(),
  "Original path (traversed)": L.layerGroup(), "Predicted path (traversed)": L.layerGroup(), "GPS pings": L.layerGroup(),
};
Object.values(layers).forEach(l => l.addTo(map));
layers["Original route (full)"].remove();
L.control.layers(null, layers, { collapsed: false }).addTo(map);
const [network, refFull, refDone, predDone, trail] = Object.values(layers);
const marker = L.circleMarker([0, 0], { radius: 8, color: COLOR.ping, weight: 2, fillColor: "#fde047", fillOpacity: 1 });

// ---- replay state ----
let trip, shown = -1, simT = 0, playing = false, lastFrame = 0;
let drawnRef, drawnPred, okPrefix, validPrefix;

function fitTrip() {
  map.fitBounds(L.latLngBounds(trip.lat.map((la, i) => [la, trip.lon[i]])), { padding: [20, 20] });
}

function loadTrip(id) {
  setPlaying(false);
  trip = D.trips.find(t => t.id === id);
  [network, refFull, refDone, predDone, trail].forEach(l => l.clearLayers());
  marker.remove();
  const segs = trip.segments;
  network.addLayer(L.polyline(Object.values(segs).map(toLL), { color: "#9a9486", weight: 1, opacity: 0.7, interactive: false }));
  refFull.addLayer(L.polyline(trip.route.map(a => toLL(segs[a])), { color: COLOR.ref, weight: 3, dashArray: "6 6", interactive: false }));
  okPrefix = []; validPrefix = []; let ok = 0, valid = 0;
  trip.pred.forEach((p, i) => {
    if (!trip.bad[i]) { valid++; ok += p === trip.truth[i]; }
    okPrefix.push(ok); validPrefix.push(valid);
  });
  $("scrub").max = trip.points - 1;
  document.querySelectorAll("#rows tr").forEach(r => r.classList.toggle("sel", r.dataset.id === id));
  const wrong = valid - ok;
  $("detail").innerHTML = [
    ["Trip", trip.id], ["Duration", Math.round(trip.duration_s).toLocaleString() + " s"],
    ["Valid pings", valid.toLocaleString()], ["Flagged pings", trip.flagged_points],
    ["Top-1 correct", `${ok.toLocaleString()} (${pct(trip.top1)})`], ["Wrong pings", wrong.toLocaleString()],
    ["Candidate coverage", pct(trip.candidate_coverage)], ["Disconnected jumps", trip.disconnected_jumps],
  ].map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");
  fitTrip();
  restart();
}

function resetDynamic() {
  refDone.clearLayers(); predDone.clearLayers(); trail.clearLayers();
  drawnRef = new Set(); drawnPred = new Set(); shown = -1;
}

function addPing(i) {
  const segs = trip.segments, truth = trip.truth[i], pred = trip.pred[i], good = pred === truth;
  if (!drawnRef.has(truth) && segs[truth]) {
    drawnRef.add(truth);
    refDone.addLayer(L.polyline(toLL(segs[truth]), { color: COLOR.ref, weight: 10, opacity: 0.4, interactive: false }));
  }
  const key = pred + (good ? "g" : "r");
  if (pred >= 0 && !drawnPred.has(key)) {
    drawnPred.add(key);
    predDone.addLayer(L.polyline(toLL(segs[pred]), { color: good ? COLOR.match : COLOR.miss, weight: 4, opacity: 0.95, interactive: false }));
  }
  trail.addLayer(L.circleMarker([trip.lat[i], trip.lon[i]], { radius: 2, color: trip.bad[i] ? COLOR.flagged : COLOR.ping, weight: 1, interactive: false }));
}

function renderTo(n) {
  if (n < shown) resetDynamic();
  for (let i = shown + 1; i <= n; i++) addPing(i);
  shown = n;
  if (n < 0) { marker.remove(); updateStats(); return; }
  const here = [trip.lat[n], trip.lon[n]];
  marker.setLatLng(here).addTo(map).bringToFront();
  if ($("follow").checked) map.setView(here, map.getZoom(), { animate: false });
  updateStats();
}

function updateStats() {
  const n = shown;
  $("s-trip").textContent = trip.id;
  $("s-ping").textContent = `${n + 1} / ${trip.points}`;
  $("s-time").textContent = n < 0 ? "0s" : `${Math.round(trip.t[n] - trip.t[0])}s / ${Math.round(trip.duration_s)}s`;
  $("s-top1").textContent = n < 0 || !validPrefix[n] ? "-" : pct(okPrefix[n] / validPrefix[n]);
  const status = $("s-last");
  if (n < 0) { status.className = "status"; status.textContent = "waiting"; }
  else if (trip.bad[n]) { status.className = "status"; status.textContent = "flagged ping"; }
  else {
    const good = trip.pred[n] === trip.truth[n];
    status.className = "status " + (good ? "ok" : "no");
    status.textContent = `${good ? "matched" : "mismatch"} (${Math.round(100 * trip.conf[n])}% conf.)`;
  }
  $("scrub").value = Math.max(n, 0);
}

// ---- time-series playback: trip time advances `speed` x faster than real time ----
function indexAt(t) {                       // last ping with time <= t
  let lo = -1, hi = trip.points - 1;
  while (lo < hi) { const mid = (lo + hi + 1) >> 1; trip.t[mid] <= t ? lo = mid : hi = mid - 1; }
  return lo;
}
function tick(now) {
  if (!playing) return;
  simT += (now - lastFrame) / 1000 * Number($("speed").value);
  lastFrame = now;
  renderTo(indexAt(simT));
  if (shown >= trip.points - 1) return setPlaying(false);
  requestAnimationFrame(tick);
}
function setPlaying(on) {
  playing = on;
  $("play").textContent = on ? "Pause" : "Play";
  if (on) { lastFrame = performance.now(); requestAnimationFrame(tick); }
}
function restart() { resetDynamic(); simT = trip.t[0] - 1; renderTo(-1); }

$("play").onclick = () => {
  if (shown >= trip.points - 1) restart();
  if ($("follow").checked && shown < 0) map.setZoom(FOLLOW_ZOOM, { animate: false });   // close-up for a fresh run
  setPlaying(!playing);
};
$("restart").onclick = () => { setPlaying(false); restart(); fitTrip(); };
$("fit").onclick = () => { $("follow").checked = false; fitTrip(); };
$("follow").onchange = e => { if (e.target.checked && shown >= 0) map.setView([trip.lat[shown], trip.lon[shown]], FOLLOW_ZOOM, { animate: false }); };
$("scrub").oninput = e => { const n = Number(e.target.value); simT = trip.t[n]; renderTo(n); };
document.addEventListener("keydown", e => { if (e.code === "Space" && !["SELECT", "INPUT"].includes(e.target.tagName)) { e.preventDefault(); $("play").click(); } });

loadTrip(D.trips[0].id);
