// Model comparison tab: the original (centrally trained) models against the federated ones, scored on the same
// held-out patients. Reads /api/federated/comparison (written by carepath_fl.compare, nothing is computed here).
// Relies on apiFetch() and escapeHtml() from app.js and on window.carepathTabs from surveillance.js.
(() => {
  const $ = selector => document.querySelector(selector);
  const esc = escapeHtml;
  // Blue (federated) and orange (original) pass the palette check for colour-blind readers; the markers also differ
  // in shape (square vs circle) so the two never rely on colour alone.
  const COLORS = { federated: "#1d5fa4", original: "#c4571f", grid: "#e4e8ec", surface: "#ffffff", link: "#b9c2cb" };
  let data = null;
  let taskId = "disease";
  let resizeTimer;
  let hospitalById = {};
  const metricFor = {};               // task id -> metric id chosen for the hospital chart

  // ---------------------------------------------------------------- formatting
  function fmt(value, format) {
    if (value === null || value === undefined || Number.isNaN(value)) return "–";
    switch (format) {
      case "pct": return (value * 100).toFixed(1) + "%";
      case "num3": return value.toFixed(3);
      case "num1": return value.toFixed(1);
      case "int": return String(Math.round(value));
      case "days": return Number.isInteger(value) ? String(value) : value.toFixed(2);
      default: return String(value);
    }
  }

  const sign = v => (v > 0 ? "+" : v < 0 ? "−" : "±");

  // A difference of two measures, with its unit: percentage points for shares, days, or plain numbers.
  function fmtDiff(v, format) {
    const a = Math.abs(v);
    if (format === "pct") return `${sign(v)}${(a * 100).toFixed(2)} pts`;
    if (format === "days") return `${sign(v)}${a.toFixed(3)} days`;
    return `${sign(v)}${a.toFixed(3)}`;
  }

  function fmtInterval(ci, format) {
    if (!ci) return "–";
    const part = v => (format === "pct" ? `${sign(v)}${(Math.abs(v) * 100).toFixed(2)}` : `${sign(v)}${Math.abs(v).toFixed(3)}`);
    return `${part(ci[0])} to ${part(ci[1])}${format === "pct" ? " pts" : format === "days" ? " days" : ""}`;
  }

  function niceTicks(lo, hi, count = 5) {
    const raw = (hi - lo) / count || 1;
    const mag = 10 ** Math.floor(Math.log10(raw));
    const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => s >= raw);
    const ticks = [];
    for (let t = Math.ceil(lo / step - 1e-9) * step; t <= hi + 1e-9; t += step) ticks.push(+t.toFixed(10));
    return { ticks, step };
  }

  function tickLabel(value, format, step) {
    if (format === "pct") return (value * 100).toFixed(step * 100 >= 1 ? 0 : 1) + "%";
    if (format === "days") return value.toFixed(step >= 0.1 ? 1 : 2);
    return value.toFixed(step >= 0.01 ? 2 : 3);
  }

  const shortName = (name, max = 24) => (name.length > max ? name.slice(0, max - 1) + "…" : name);
  const hospitalName = id => (hospitalById[id] ? hospitalById[id].name : id);

  const VERDICT = {
    better: { text: "Federated is better", cls: "better" },
    worse: { text: "Federated is worse", cls: "worse" },
    "no clear difference": { text: "No clear difference", cls: "tie" },
  };
  const verdictCell = m => (m.verdict
    ? `<span class="cmp-verdict ${VERDICT[m.verdict].cls}">${VERDICT[m.verdict].text}</span>`
    : `<span class="cmp-verdict none">Not ranked</span>`);

  // ---------------------------------------------------------------- tooltip (values lead, labels follow; textContent only)
  function tipFor(wrap) {
    let tip = wrap.querySelector(".fl-tip");
    if (!tip) {
      tip = document.createElement("div");
      tip.className = "fl-tip hidden";
      tip.setAttribute("role", "status");
      wrap.appendChild(tip);
    }
    return tip;
  }

  function showTip(wrap, event, rows, title) {
    const tip = tipFor(wrap);
    tip.replaceChildren();
    if (title) {
      const head = document.createElement("div");
      head.className = "fl-tip-title";
      head.textContent = title;
      tip.appendChild(head);
    }
    for (const row of rows) {
      const line = document.createElement("div");
      line.className = "fl-tip-row";
      const key = document.createElement("i");
      key.className = "fl-tip-key " + (row.key || "");
      const value = document.createElement("b");
      value.textContent = row.value;
      const label = document.createElement("span");
      label.textContent = row.label;
      line.append(key, value, label);
      tip.appendChild(line);
    }
    tip.classList.remove("hidden");
    const box = wrap.getBoundingClientRect();
    let x = event.clientX - box.left + 14;
    if (x + tip.offsetWidth > box.width - 4) x = event.clientX - box.left - tip.offsetWidth - 14;
    tip.style.left = Math.max(4, x) + "px";
    tip.style.top = Math.max(4, Math.min(event.clientY - box.top - 10, box.height - tip.offsetHeight - 4)) + "px";
  }

  const hideTip = wrap => tipFor(wrap).classList.add("hidden");

  function focusPoint(el) {                      // keyboard focus shows the same tooltip as hover
    const r = el.getBoundingClientRect();
    return { clientX: r.left + r.width / 2, clientY: r.top + r.height / 2 };
  }

  // ---------------------------------------------------------------- summary across all models
  function renderSummary() {
    const scored = data.tasks.filter(t => t.kind === "scored");
    const ranked = scored.flatMap(t => t.metrics.filter(m => m.verdict));
    const count = v => ranked.filter(m => m.verdict === v).length;
    const rows = scored.flatMap(t => t.metrics.map(m => `<tr>
        <th scope="row">${esc(t.title)}<small>${esc(m.label)}</small></th>
        <td class="num">${fmt(m.original, m.format)}</td>
        <td class="num">${fmt(m.federated, m.format)}</td>
        <td class="num">${fmtDiff(m.difference, m.format)}</td>
        <td class="num">${fmtInterval(m.ci, m.format)}</td>
        <td>${verdictCell(m)}</td></tr>`));
    for (const t of data.tasks.filter(t => t.kind === "recorded")) {
      const same = t.rows.filter(r => r.original === r.federated).length;
      rows.push(`<tr><th scope="row">${esc(t.title)}<small>${t.rows.length} held-out measures, recorded</small></th>
        <td class="num">–</td><td class="num">–</td><td class="num">–</td><td class="num">–</td>
        <td><span class="cmp-verdict tie">Identical on ${same} of ${t.rows.length}</span></td></tr>`);
    }
    $("#cmp-summary").innerHTML = `<p class="result-kicker">Summary <span class="surv-day">${data.generated_at ? "scored " + esc(new Date(data.generated_at).toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" })) : ""}</span></p>
      <p class="cmp-lead">On the ${ranked.length} measures that can be compared fairly, the federated models are clearly better on <b>${count("better")}</b>, clearly worse on <b>${count("worse")}</b>, and <b>not distinguishable</b> from the originals on <b>${count("no clear difference")}</b>.</p>
      <div class="heat-wrap"><table class="data-table cmp-table">
        <thead><tr><th scope="col">Model and measure</th><th scope="col" class="num"><i class="cmp-key original"></i>Original</th><th scope="col" class="num"><i class="cmp-key federated"></i>Federated</th><th scope="col" class="num">Difference</th><th scope="col" class="num">95% interval</th><th scope="col">Verdict</th></tr></thead>
        <tbody>${rows.join("")}</tbody></table></div>
      <p class="fl-chart-note">Difference is federated minus original, on the same held-out patients. The interval comes from ${data.resamples ? data.resamples.toLocaleString() : "many"} paired resamples of those patients; a difference counts as better or worse only when the interval leaves zero. Several models are compared, so one clear result among many can still be chance.</p>`;
  }

  function renderTaskButtons() {
    $("#cmp-tasks").innerHTML = data.tasks.map(t => `
      <button class="fl-task" type="button" data-task="${t.id}" aria-pressed="${t.id === taskId}">
        ${esc(t.title)}<small>${t.kind === "scored" ? "scored again on the same patients" : "recorded held-out results"}</small>
      </button>`).join("");
  }

  // ---------------------------------------------------------------- per-hospital chart
  const legendHtml = () => `<div class="fl-legend" aria-label="Legend">
      <span><i class="cmp-key original"></i>Original (trained on pooled data)</span>
      <span><i class="cmp-key federated"></i>Federated</span></div>`;

  function dumbbellSvg(m, W) {
    const rows = m.per_hospital;
    const narrow = W < 520;
    const left = narrow ? 112 : 168, right = 24, top = 30, rowH = 31;
    const H = top + rows.length * rowH + 10;
    const vals = rows.flatMap(r => [r.original, r.federated]).filter(Number.isFinite);
    let lo = Math.min(...vals), hi = Math.max(...vals);
    const pad = (hi - lo) * 0.14 || 0.01;
    lo -= pad; hi += pad;
    const { ticks, step } = niceTicks(lo, hi);
    const x = v => left + ((v - lo) / (hi - lo)) * (W - left - right);
    const grid = ticks.map(t => `<line x1="${x(t).toFixed(1)}" x2="${x(t).toFixed(1)}" y1="${top - 6}" y2="${H - 6}" stroke="${COLORS.grid}" stroke-width="1"/>
        <text x="${x(t).toFixed(1)}" y="${top - 12}" class="fl-axis" text-anchor="middle">${tickLabel(t, m.format, step)}</text>`).join("");
    const body = rows.map((r, i) => {
      const y = top + i * rowH + rowH / 2;
      const link = `<line x1="${x(r.original).toFixed(1)}" x2="${x(r.federated).toFixed(1)}" y1="${y}" y2="${y}" stroke="${COLORS.link}" stroke-width="3" stroke-linecap="round"/>`;
      const original = `<rect x="${(x(r.original) - 6).toFixed(1)}" y="${y - 6}" width="12" height="12" rx="2" fill="${COLORS.original}" stroke="${COLORS.surface}" stroke-width="1.5"/>`;
      const federated = `<circle cx="${x(r.federated).toFixed(1)}" cy="${y}" r="4.5" fill="${COLORS.federated}" stroke="${COLORS.surface}" stroke-width="1.5"/>`;
      return `<g class="fl-row" data-i="${i}" tabindex="0" role="img" aria-label="${esc(hospitalName(r.hospital))}: original ${fmt(r.original, m.format)}, federated ${fmt(r.federated, m.format)}">
          <rect x="0" y="${top + i * rowH}" width="${W}" height="${rowH}" fill="transparent"/>
          <text x="${left - 14}" y="${y}" class="fl-rowlabel" text-anchor="end" dominant-baseline="central">${esc(shortName(hospitalName(r.hospital), narrow ? 15 : 24))}</text>
          ${link}${original}${federated}</g>`;
    }).join("");
    return `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" class="fl-dumbbell" role="group" aria-label="${esc(m.label)} for each hospital">${grid}${body}</svg>`;
  }

  function hospitalTable(m) {
    const body = m.per_hospital.map(r => `<tr><th scope="row">${esc(hospitalName(r.hospital))}</th><td>${r.rows.toLocaleString()}</td><td>${fmt(r.original, m.format)}</td><td>${fmt(r.federated, m.format)}</td><td>${fmtDiff(r.federated - r.original, m.format)}</td></tr>`).join("");
    return `<details class="fl-table"><summary>Show as a table</summary>
      <table><thead><tr><th scope="col">Hospital</th><th scope="col">Test patients</th><th scope="col">Original</th><th scope="col">Federated</th><th scope="col">Difference</th></tr></thead><tbody>${body}</tbody></table></details>`;
  }

  const chartWidth = el => Math.max(300, Math.floor(el.clientWidth - 26));   // 1:1 pixels, so chart text matches the page

  function hospitalChart(task) {
    const options = task.metrics.filter(m => m.per_hospital.length);
    const metric = options.find(m => m.id === metricFor[task.id]) || options[0];
    metricFor[task.id] = metric.id;
    const picker = options.length > 1
      ? `<div class="fl-metrics" role="group" aria-label="Measure">${options.map(m => `<button type="button" class="fl-metric" data-metric="${m.id}" aria-pressed="${m.id === metric.id}">${esc(m.label)}</button>`).join("")}</div>` : "";
    const better = metric.hospitals_better;
    const lead = better === null || better === undefined ? ""
      : `Federated is ${metric.higher_is_better ? "higher" : "lower"}, which is better, at <b>${better} of ${metric.per_hospital.length}</b> hospitals. `;
    return `<p class="fl-subhead">Hospital by hospital</p>
      ${picker}
      <h3 class="fl-chart-title">${esc(metric.label)}</h3>
      ${legendHtml()}
      <div class="fl-chart" id="cmp-dumbbell">${dumbbellSvg(metric, chartWidth($("#cmp-detail")))}</div>
      <p class="fl-chart-note">${lead}Each hospital's figure rests on its own test patients, so a lead at a few hospitals can be chance. The axis does not start at zero: read the numbers, not the line lengths.</p>
      ${hospitalTable(metric)}`;
  }

  function wireChart(task) {
    const wrap = $("#cmp-dumbbell");
    if (!wrap) return;
    const metric = task.metrics.find(m => m.id === metricFor[task.id]);
    const rows = metric.per_hospital;
    const show = (event, i) => {
      const r = rows[i];
      showTip(wrap, event, [
        { key: "federated", value: fmt(r.federated, metric.format), label: "Federated" },
        { key: "original", value: fmt(r.original, metric.format), label: "Original" },
        { key: "", value: fmtDiff(r.federated - r.original, metric.format), label: "Difference" },
      ], `${hospitalName(r.hospital)} · ${r.rows.toLocaleString()} test patients`);
    };
    wrap.querySelectorAll(".fl-row").forEach(g => {
      const i = Number(g.dataset.i);
      g.addEventListener("pointermove", e => show(e, i));
      g.addEventListener("pointerleave", () => hideTip(wrap));
      g.addEventListener("focus", () => show(focusPoint(g), i));
      g.addEventListener("blur", () => hideTip(wrap));
    });
  }

  // ---------------------------------------------------------------- one model
  function factsTable(task) {
    return `<p class="fl-subhead">How the two were built</p>
      <table class="data-table"><thead><tr><th scope="col">Detail</th><th scope="col"><i class="cmp-key original"></i>Original</th><th scope="col"><i class="cmp-key federated"></i>Federated</th></tr></thead>
      <tbody>${task.facts.map(f => `<tr><th scope="row">${esc(f.label)}</th><td>${esc(f.original)}</td><td>${esc(f.federated)}</td></tr>`).join("")}</tbody></table>`;
  }

  function recordedTable(task) {
    const rows = task.rows.map(r => {
      const same = r.original === r.federated;
      return `<tr><th scope="row">${esc(r.label)}</th><td class="num">${fmt(r.original, r.format)}</td><td class="num">${fmt(r.federated, r.format)}</td><td>${same ? "identical" : "differs"}</td></tr>`;
    }).join("");
    const days = task.test_days ? ` <span class="surv-day">${esc(task.test_days[0])} to ${esc(task.test_days[1])}</span>` : "";
    return `<p class="fl-subhead">Held-out results${days}</p>
      <div class="heat-wrap"><table class="data-table cmp-table"><thead><tr><th scope="col">Measure</th><th scope="col" class="num"><i class="cmp-key original"></i>Original</th><th scope="col" class="num"><i class="cmp-key federated"></i>Federated</th><th scope="col"></th></tr></thead><tbody>${rows}</tbody></table></div>`;
  }

  function renderTask() {
    const task = data.tasks.find(t => t.id === taskId) || data.tasks[0];
    taskId = task.id;
    renderTaskButtons();
    const middle = task.kind === "scored" ? hospitalChart(task) : recordedTable(task);
    const facts = factsTable(task);
    $("#cmp-detail").innerHTML = `<h2 class="fl-question">${esc(task.question)}</h2>
      <p class="fl-model">${esc(task.model)}${task.test_rows ? ` · scored on ${task.test_rows.toLocaleString()} held-out patients` : ""}</p>
      ${middle}${facts}
      <details class="fl-notes" open><summary>Read before trusting the numbers</summary><ul>${task.notes.map(n => `<li>${esc(n)}</li>`).join("")}</ul></details>`;
    wireChart(task);
  }

  function showError(message) {
    const el = $("#cmp-error");
    el.textContent = message || "";
    el.classList.toggle("hidden", !message);
  }

  async function load() {
    try {
      const response = await apiFetch("/api/federated/comparison");
      const body = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(typeof body.detail === "string" ? body.detail : "The comparison could not be loaded.");
      data = body;
      hospitalById = Object.fromEntries(data.hospitals.map(h => [h.id, h]));
      showError("");
      if (!data.available) {
        $("#cmp-summary").innerHTML = `<p class="empty-state">${esc(data.hint)}</p>`;
        $("#cmp-tasks").innerHTML = "";
        $("#cmp-detail").innerHTML = "";
        return;
      }
      $("#cmp-scope").textContent = "Both sets of models are read from disk and scored here on synthetic data; nothing is retrained. "
        + "The federated models were trained in a simulation, with all hospitals on one machine.";
      renderSummary();
      renderTask();
    } catch (error) {
      showError(error instanceof TypeError ? "Could not reach the API. Make sure the backend is running at this address." : error.message);
    }
  }

  // ---------------------------------------------------------------- wiring
  $("#cmp-tasks").addEventListener("click", event => {
    const button = event.target.closest("[data-task]");
    if (!button || button.dataset.task === taskId) return;
    taskId = button.dataset.task;
    renderTask();
  });
  $("#cmp-detail").addEventListener("click", event => {
    const button = event.target.closest("[data-metric]");
    if (!button) return;
    metricFor[taskId] = button.dataset.metric;
    renderTask();
  });
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      if (data && data.available && !$("#comparison-view").classList.contains("hidden")) renderTask();
    }, 150);
  });

  window.carepathTabs.register("comparison", "#comparison-view", load);
  if (location.hash === "#comparison") document.querySelector('.topnav-tab[data-tab="comparison"]').click();
})();
