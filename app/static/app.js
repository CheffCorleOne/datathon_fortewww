const $ = (id) => document.getElementById(id);
const fileInput = $("file");
const drop = $("drop");
let videoURL = null;

const STAGES = {
  queued: "В очереди",
  starting: "Подготовка",
  decoding: "Чтение видео",
  detecting: "Поиск людей и предметов",
  tracking: "Отслеживание действий",
  events: "Проверка правил",
  verifying: "Проверка локальной визуальной моделью",
  report: "Формирование отчёта",
};

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
function fmt(t) {
  const m = Math.floor(t / 60), s = (t % 60).toFixed(1).padStart(4, "0");
  return `${m}:${s}`;
}

function pick(file) {
  if (!file) return;
  const dt = new DataTransfer();
  dt.items.add(file);
  fileInput.files = dt.files;
  $("file-name").textContent = `${file.name} · ${(file.size / 1e6).toFixed(1)} МБ`;
  $("submit").disabled = false;
}
fileInput.addEventListener("change", () => pick(fileInput.files[0]));
["dragenter", "dragover"].forEach((e) => drop.addEventListener(e, (ev) => { ev.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((e) => drop.addEventListener(e, (ev) => { ev.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (ev) => pick(ev.dataTransfer.files[0]));

$("upload-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const file = fileInput.files[0];
  if (!file) return;
  if (videoURL) URL.revokeObjectURL(videoURL);
  videoURL = URL.createObjectURL(file);
  $("result").classList.add("hidden");
  $("status-card").classList.remove("hidden");
  $("spinner").classList.remove("hidden");
  $("submit").disabled = true;
  setStatus("Загрузка видео на сервер…", "", 0.02);

  const fd = new FormData();
  fd.append("file", file);
  try {
    const r = await fetch("/ui/analyze", { method: "POST", body: fd });
    const j = await r.json();
    if (!r.ok) throw new Error(j.detail || r.statusText);
    history.replaceState(null, "", `?job=${j.job_id}`);
    poll(j.job_id, Date.now());
  } catch (e) {
    fail(`Не удалось загрузить: ${e.message}`);
  }
});

function setStatus(text, sub, frac) {
  $("status-text").textContent = text;
  $("status-sub").textContent = sub || "";
  $("bar").style.width = `${Math.round((frac || 0) * 100)}%`;
}
function fail(msg) {
  setStatus(msg, "Попробуйте ещё раз или выберите другой файл.", 0);
  $("spinner").classList.add("hidden");
  $("submit").disabled = false;
}

async function poll(id, t0) {
  try {
    const r = await fetch(`/ui/jobs/${id}`);
    const j = await r.json();
    if (j.status === "done") return render(j);
    if (r.status === 404) return fail("Задача не найдена");
    if (j.status === "failed") return fail(`Ошибка обработки: ${j.error || "неизвестно"}`);
    const secs = Math.round((Date.now() - t0) / 1000);
    const q = j.queue_position > 1 ? ` · в очереди: ${j.queue_position}` : "";
    setStatus(STAGES[j.stage] || "Обработка…", `Задача ${id} · ${secs} с${q}`, j.progress);
  } catch (e) { /* transient; keep polling */ }
  setTimeout(() => poll(id, t0), 1500);
}

function render(j) {
  $("status-card").classList.add("hidden");
  $("submit").disabled = false;
  $("result").classList.remove("hidden");
  const bad = j.verdict === "violation";
  const v = $("verdict");
  v.className = `verdict ${bad ? "bad" : "ok"}`;
  v.innerHTML = (bad ? `Обнаружены признаки возможных нарушений: ${j.violations.length}` : "Нарушение не обнаружено") +
    `<span class="meta">${esc(j.filename || "")} · длительность ${fmt(j.video?.duration_sec || 0)} · обработано за ${j.processing_time_sec} с</span>`;

  const player = $("player");
  player.src = videoURL || `/ui/jobs/${j.job_id}/video`;
  $("player-card").classList.remove("hidden");
  const tl = $("timeline");
  tl.innerHTML = "";
  const dur = j.video?.duration_sec || 1;
  j.violations.forEach((x) => {
    const m = document.createElement("div");
    m.className = "mark";
    m.style.left = `${Math.min(99.5, (x.timecode_sec / dur) * 100)}%`;
    m.title = `${fmt(x.timecode_sec)} — ${x.type}`;
    m.onclick = () => seek(x.timecode_sec);
    tl.appendChild(m);
  });

  $("violations").innerHTML = j.violations.map((x, i) => {
    const r = x.rule || {};
    const ruleText = r.quote ? `<q>${esc(r.quote)}</q>` : esc(r.summary || "");
    const frames = x.evidence_frames.map((f) =>
      `<figure><img src="data:image/jpeg;base64,${f.image_base64}" alt="кадр ${f.timestamp_sec} с"><figcaption>кадр на ${fmt(f.timestamp_sec)}</figcaption></figure>`).join("");
    return `<article class="card viol">
      <h3><span class="tc" data-t="${x.timecode_sec}">${fmt(x.timecode_sec)}</span>${i + 1}. ${esc(x.type)}</h3>
      <div class="muted small">Интервал ${fmt(x.interval_sec?.[0] ?? x.timecode_sec)}–${fmt(x.interval_sec?.[1] ?? x.timecode_sec)} · уверенность ${Math.round((x.confidence ?? 0) * 100)}%</div>
      <p class="desc">${esc(x.description)}</p>
      <div class="frames">${frames}</div>
      <div class="rule"><b>Норма:</b> ${esc(r.document || "")}, ${esc(r.clause || "")}. ${ruleText}
        ${r.url ? ` <a href="${esc(r.url)}" target="_blank" rel="noopener">adilet.zan.kz</a>` : ""}</div>
    </article>`;
  }).join("");
  document.querySelectorAll(".tc").forEach((el) => (el.onclick = () => seek(+el.dataset.t)));
  document.querySelectorAll(".frames img").forEach((img) => (img.onclick = () => lightbox(img.src)));

  const slim = JSON.parse(JSON.stringify(j));
  slim.violations.forEach((x) => x.evidence_frames.forEach((f) => (f.image_base64 = f.image_base64.slice(0, 40) + "…")));
  $("raw").textContent = JSON.stringify(slim, null, 2);
}

function seek(t) {
  const p = $("player");
  p.currentTime = Math.max(0, t - 1);
  p.scrollIntoView({ behavior: "smooth", block: "center" });
  p.play().catch(() => {});
}
function lightbox(src) {
  const d = document.createElement("div");
  d.className = "lightbox";
  d.innerHTML = `<img src="${src}">`;
  d.onclick = () => d.remove();
  document.body.appendChild(d);
}

const initialJob = new URLSearchParams(location.search).get("job");
if (initialJob && /^[a-z0-9]+$/i.test(initialJob)) {
  $("status-card").classList.remove("hidden");
  setStatus("Загрузка результата…", `Задача ${initialJob}`, 0.05);
  poll(initialJob, Date.now());
}

fetch("/ui/rules").then((r) => r.json()).then((rules) => {
  $("rules").innerHTML = rules.filter((r) => r.code !== "other").map((r) =>
    `<div class="r"><b>${esc(r.title)}</b>${esc(r.clause)}</div>`).join("");
}).catch(() => {});
