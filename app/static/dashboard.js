const COLORS = { accent: "#4cc2a0", amber: "#f0b45a", blue: "#6aa8ff", grid: "#2a3541", text: "#93a1af", bad: "#ff7a7a" };
Chart.defaults.color = COLORS.text;
Chart.defaults.borderColor = COLORS.grid;
Chart.defaults.maintainAspectRatio = false;

const $ = (id) => document.getElementById(id);
const getJSON = (url) => fetch(url).then((r) => r.json());
const pct = (x) => (x * 100).toFixed(1) + "%";

function tile(label, value, hint = "") {
  return `<div class="tile"><div class="tile-value">${value}</div><div class="tile-label">${label}</div>${hint ? `<div class="tile-hint">${hint}</div>` : ""}</div>`;
}

function duration(seconds) {
  if (seconds == null) return "n/a";
  const d = Math.floor(seconds / 86400), h = Math.floor((seconds % 86400) / 3600), m = Math.floor((seconds % 3600) / 60);
  return d ? `${d}d ${h}h` : h ? `${h}h ${m}m` : `${m}m`;
}

async function loadStatus() {
  const data = await getJSON("/api/uptime");
  const dep = data.deployment;
  $("status-tiles").innerHTML =
    tile("Running for", duration(data.uptime_seconds), data.running_since ? "since " + new Date(data.running_since * 1000).toLocaleString() : "") +
    tile("Uptime (24h)", data.uptime_pct == null ? "n/a" : data.uptime_pct + "%") +
    tile("Deployed on", dep.platform, "Python " + dep.python) +
    tile("Model version", "v" + dep.model_version, dep.threshold != null ? "threshold " + dep.threshold.toFixed(3) : "");

  new Chart($("uptime-chart"), {
    type: "bar",
    data: {
      labels: data.buckets.map((b) => new Date(b.t * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })),
      datasets: [{
        data: data.buckets.map((b) => (b.up === null ? 0 : 1)),
        backgroundColor: data.buckets.map((b) => (b.up === null ? "transparent" : b.up ? COLORS.accent : COLORS.bad)),
        barPercentage: 1, categoryPercentage: 1,
      }],
    },
    options: { plugins: { legend: { display: false }, tooltip: { enabled: false } }, scales: { y: { display: false, max: 1 }, x: { ticks: { maxTicksLimit: 8 } } } },
  });
}

const METRIC_ROWS = [["accuracy", "Accuracy"], ["precision", "Precision"], ["recall", "Recall"], ["f1", "F1"], ["auc", "AUC"], ["far", "False accept"], ["frr", "False reject"]];
let rocChart;

function showSplit(report, split) {
  const m = report.metrics[split];
  $("metric-tiles").innerHTML = METRIC_ROWS.map(([key, label]) =>
    tile(label, key === "auc" ? m[key].toFixed(3) : pct(m[key]), key === "far" ? "strangers let in" : key === "frr" ? "you locked out" : "")).join("");
  const points = m.fpr.map((x, i) => ({ x, y: m.tpr[i] }));
  if (rocChart) rocChart.destroy();
  rocChart = new Chart($("roc-chart"), {
    type: "line",
    data: { datasets: [
      { label: "ROC", data: points, borderColor: COLORS.accent, pointRadius: 0, tension: 0 },
      { label: "Chance", data: [{ x: 0, y: 0 }, { x: 1, y: 1 }], borderColor: COLORS.grid, borderDash: [6, 6], pointRadius: 0 },
    ] },
    options: { parsing: false, scales: { x: { type: "linear", min: 0, max: 1, title: { display: true, text: "False positive rate" } }, y: { min: 0, max: 1, title: { display: true, text: "True positive rate" } } }, plugins: { legend: { display: false } } },
  });
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.split === split));
}

async function loadMetrics() {
  const report = await getJSON("/api/metrics");
  if (!report.available) { $("metrics-empty").hidden = false; return; }
  $("metrics-body").hidden = false;
  $("best-run").textContent = report.run;
  $("threshold").textContent = report.threshold.toFixed(3);
  document.querySelectorAll(".tab").forEach((t) => (t.onclick = () => showSplit(report, t.dataset.split)));
  showSplit(report, "test");

  const rows = Object.entries(report.experiments || {});
  $("experiments").innerHTML = "<tr><th>Run</th><th>Test AUC</th><th>Test F1</th><th>Official AUC</th></tr>" +
    rows.map(([name, e]) => `<tr${name === report.run ? ' class="best"' : ""}><td>${name}</td><td>${e.test.auc.toFixed(3)}</td><td>${e.test.f1.toFixed(3)}</td><td>${e.official.auc.toFixed(3)}</td></tr>`).join("");
}

function histogramChart(id, hist, color) {
  const labels = hist.counts.map((_, i) => Math.round(hist.edges[i]));
  new Chart($(id), { type: "bar", data: { labels, datasets: [{ data: hist.counts, backgroundColor: color, barPercentage: 1, categoryPercentage: 0.95 }] }, options: { plugins: { legend: { display: false } }, scales: { x: { ticks: { maxTicksLimit: 8 } } } } });
}

async function loadData() {
  const d = await getJSON("/api/dataset-stats");
  if (!d.available) { $("data-empty").hidden = false; return; }
  $("data-body").hidden = false;
  const names = Object.keys(d.splits);
  new Chart($("splits-chart"), { type: "bar", data: { labels: names, datasets: [
    { label: "People", data: names.map((n) => d.splits[n].people), backgroundColor: COLORS.blue },
    { label: "Photos", data: names.map((n) => d.splits[n].photos), backgroundColor: COLORS.accent },
  ] } });
  new Chart($("ppp-chart"), { type: "bar", data: { labels: d.photos_per_person.labels, datasets: [{ data: d.photos_per_person.people, backgroundColor: COLORS.amber }] }, options: { plugins: { legend: { display: false } }, scales: { y: { type: "logarithmic" } } } });
  histogramChart("width-chart", d.kaggle.width, COLORS.blue);
  histogramChart("brightness-chart", d.kaggle.brightness, COLORS.amber);
  const modes = Object.entries(d.kaggle.colour_modes).map(([k, v]) => `${k}: ${v}`).join(", ");
  $("data-notes").textContent = `LFW photos capped at ${d.photo_cap} per person. Face detector found exactly one face in ${pct(d.detector_success_rate)} of training photos. Kaggle set: ${d.kaggle.total_images} images (colour modes in a ${d.kaggle.sampled}-image sample: ${modes}).`;
}

const RESULT_ROWS = [["auc", "AUC", false], ["accuracy", "Accuracy", true], ["f1", "F1", true], ["far", "False accept", true], ["frr", "False reject", true]];
let pollTimer;

function renderJob(state) {
  const job = state.job;
  if (!job) { $("retrain-status").textContent = `Live model: v${state.active_version}. No retrain yet.`; return; }
  const running = job.status === "running";
  $("retrain-start").disabled = running;
  $("retrain-bar").hidden = !running;
  $("retrain-fill").style.width = Math.round(job.progress * 100) + "%";
  const label = { running: `Retraining (${job.mode}) ${Math.round(job.progress * 100)}%`, done: `Retrain (${job.mode}) finished.`, failed: "Retrain failed: " + (job.message || "unknown error") }[job.status];
  $("retrain-status").textContent = `${label} Live model: v${state.active_version}.`;
  $("retrain-status").className = job.status === "failed" ? "error" : "muted";

  const result = job.result;
  $("retrain-result").hidden = !(job.status === "done" && result);
  if (job.status === "done" && result) {
    $("retrain-table").innerHTML = "<tr><th>Metric</th><th>Old</th><th>New</th></tr>" + RESULT_ROWS.map(([k, name, asPct]) => {
      const fmt = (v) => (asPct ? pct(v) : v.toFixed(3));
      const better = k === "far" || k === "frr" ? result.new[k] <= result.old[k] : result.new[k] >= result.old[k];
      return `<tr><td>${name}</td><td>${fmt(result.old[k])}</td><td class="${better ? "best-cell" : ""}">${fmt(result.new[k])}</td></tr>`;
    }).join("");
    $("replay-block").hidden = !result.replay_old;
    if (result.replay_old) {
      $("replay-table").innerHTML = "<tr><th>Metric</th><th>Old model</th><th>New model</th></tr>" + RESULT_ROWS.map(([k, name, asPct]) => {
        const fmt = (v) => (asPct ? pct(v) : v.toFixed(3));
        const better = k === "far" || k === "frr" ? result.replay_new[k] <= result.replay_old[k] : result.replay_new[k] >= result.replay_old[k];
        return `<tr><td>${name}</td><td>${fmt(result.replay_old[k])}</td><td class="${better ? "best-cell" : ""}">${fmt(result.replay_new[k])}</td></tr>`;
      }).join("");
    }
    $("retrain-note").textContent = `Trained on ${result.replay_people ? result.replay_people + " original and " : ""}${result.people_train} uploaded people, compared on ${result.people_eval} held-out people (${result.eval_pairs} pairs). Small samples are noisy: promote only if the new model is clearly better.`;
    $("promote").hidden = job.promoted;
    if (job.promoted) $("retrain-note").textContent += " This model is live.";
  }
  if (running) pollTimer = setTimeout(pollRetrain, 2000);
}

async function pollRetrain() { clearTimeout(pollTimer); renderJob(await getJSON("/api/retrain/status")); }

$("retrain-start").onclick = async () => {
  const file = $("retrain-file").files[0];
  if (!file) { $("retrain-status").textContent = "Choose a zip file first."; return; }
  const form = new FormData();
  form.append("mode", $("retrain-mode").value);
  form.append("dataset", file);
  $("retrain-status").textContent = "Uploading and checking the zip…";
  const data = await (await fetch("/api/retrain", { method: "POST", body: form })).json();
  if (!data.ok) { $("retrain-status").textContent = data.message; $("retrain-status").className = "error"; return; }
  pollRetrain();
};

$("promote").onclick = async () => {
  if (!confirm("Make the new model live? Every user will be re-enrolled with it.")) return;
  const data = await (await fetch("/api/retrain/promote", { method: "POST" })).json();
  $("retrain-status").textContent = data.message;
  $("retrain-status").className = data.ok ? "ok" : "error";
  if (data.ok) setTimeout(() => location.reload(), 1200);
};

loadStatus();
loadMetrics();
loadData();
pollRetrain();
