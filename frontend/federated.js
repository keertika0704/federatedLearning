// Federated learning tab (objective C): how the hospitals learn together, what the results are, and what is
// and is not protected. Reads /api/federated/overview (results the federated training already wrote).
// Relies on apiFetch() and escapeHtml() from app.js and on window.carepathTabs from surveillance.js.
(() => {
  const $ = selector => document.querySelector(selector);
  const esc = escapeHtml;
  const COLORS = { fed: "#1d5fa4", alone: "#7b8791", ink: "#3b4651", grid: "#e4e8ec", surface: "#ffffff" };
  let resizeTimer;

  let data = null;
  let taskId = "disease";
  const metricFor = {};               // task id -> metric id chosen in the results
  const replay = { metric: null, turn: 0, timer: null };
  let hospitalById = {};

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

  function fmtDelta(delta, format) {
    const sign = delta > 0 ? "+" : delta < 0 ? "−" : "±";
    const abs = Math.abs(delta);
    const body = format === "pct" ? `${(abs * 100).toFixed(1)} pts` : format === "days" ? `${abs.toFixed(2)} days`
      : abs.toFixed(format === "num3" ? 3 : 1);
    return sign + body;
  }

  const rollLen = () => data.hospitals.length;     // one full round: every hospital has had exactly one turn

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

  // ---------------------------------------------------------------- header numbers
  function renderKpis() {
    const trained = data.tasks.filter(t => t.available).length;
    $("#fl-kpis").innerHTML = `<table class="data-table fl-summary-table">
      <thead><tr><th scope="col">Measure</th><th scope="col" class="num">Value</th><th scope="col">Detail</th></tr></thead>
      <tbody>${[
        [data.hospitals.length, "Hospitals take part", "Each keeps its own patients"],
        [`${trained} of ${data.tasks.length}`, "Models trained across them", "Disease, treatment, atypical cases, outbreaks"],
        ["0", "Patient rows leave a hospital", "Only models and summary statistics travel, see what is and isn't protected below"],
      ].map(([value, label, note]) => `<tr><th scope="row">${esc(label)}</th><td class="num">${esc(value)}</td><td>${esc(note)}</td></tr>`).join("")}</tbody></table>`;
  }

  function renderTaskButtons() {
    $("#fl-tasks").innerHTML = data.tasks.map(t => `
      <button class="fl-task" type="button" data-task="${t.id}" aria-pressed="${t.id === taskId}">
        ${esc(t.title)}<small>${t.available ? esc(t.kind === "turns" ? "hospitals take turns" : "hospitals send summaries") : "not trained yet"}</small>
      </button>`).join("");
  }

  // ---------------------------------------------------------------- network diagram
  function renderNetwork(task) {
    const hospitals = data.hospitals;
    const C = 200, R = 148;
    const pos = hospitals.map((h, i) => {
      const a = -Math.PI / 2 + (i * 2 * Math.PI) / hospitals.length;
      return { h, x: C + R * Math.cos(a), y: C + R * Math.sin(a) };
    });
    const links = pos.map(p => `<line class="fl-link" data-h="${esc(p.h.id)}" x1="${C}" y1="${C}" x2="${p.x.toFixed(1)}" y2="${p.y.toFixed(1)}"/>`).join("");
    const nodes = pos.map(p => `<g class="fl-node" data-h="${esc(p.h.id)}" tabindex="0" role="img" aria-label="${esc(p.h.name)}">
        <title>${esc(p.h.name)}${p.h.city ? ", " + esc(p.h.city) : ""}</title>
        <circle cx="${p.x.toFixed(1)}" cy="${p.y.toFixed(1)}" r="19"/>
        <text x="${p.x.toFixed(1)}" y="${p.y.toFixed(1)}">${Number(p.h.id.slice(1))}</text></g>`).join("");
    const hub = task.kind === "turns" ? ["Shared", "model"] : ["Combined", "statistics"];
    $("#fl-network").className = "fl-network" + (task.kind === "turns" ? "" : " all");
    $("#fl-network").innerHTML = `
      <svg viewBox="0 0 400 400" role="img" aria-label="${hospitals.length} hospitals around one server. Only the ${esc(hub.join(" ").toLowerCase())} travels, never patient records.">
        ${links}
        <g class="fl-hub"><circle cx="${C}" cy="${C}" r="44"/><text x="${C}" y="${C - 7}">${hub[0]}</text><text x="${C}" y="${C + 9}">${hub[1]}</text></g>
        ${nodes}
      </svg>
      <figcaption>${esc(task.method)}</figcaption>`;
  }

  function setActiveHospital(id) {
    document.querySelectorAll("#fl-network .fl-node, #fl-network .fl-link").forEach(el => {
      el.classList.toggle("active", Boolean(id) && el.dataset.h === id);
    });
  }

  // ---------------------------------------------------------------- task detail
  function settingsLine(task) {
    const s = task.settings || {};
    if (task.id === "atypical") {
      return s.fit_rows ? `Built from ${s.fit_rows.toLocaleString()} patients' statistics in ${s.em_rounds} shared rounds, never their records.` : "";
    }
    if (task.id === "outbreak") {
      return s.hospitals ? `${s.hospitals} hospitals sent daily counts and ${s.labelled_outbreaks} labelled outbreaks (${s.linked_outbreaks} spread between hospitals).` : "";
    }
    const iterations = s.iterations && typeof s.iterations === "object"
      ? Object.entries(s.iterations).map(([k, v]) => `${k} ${v}`).join(" · ") : s.iterations;
    return s.turns ? `${s.turns} turns, ${s.local_iters} boosting steps each (${iterations} in the final model).` : "";
  }

  function renderDetail(task) {
    const updated = task.updated ? new Date(task.updated).toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" }) : "";
    $("#fl-detail").innerHTML = `
      <p class="result-kicker">${esc(task.title)}${updated ? ` <span class="surv-day">trained ${esc(updated)}</span>` : ""}</p>
      <h2 class="fl-question">${esc(task.question)}</h2>
      <p class="fl-model">${esc(task.model)}</p>
      <div class="fl-lists">
        <div><p class="alert-label">What travels</p><ul class="fl-list travels">${task.travels.map(t => `<li>${esc(t)}</li>`).join("")}</ul></div>
        <div><p class="alert-label">What never leaves a hospital</p><ul class="fl-list stays"><li>Patient rows</li><li>Names and identifiers</li></ul></div>
      </div>
      ${task.available ? `<p class="fl-settings">${esc(settingsLine(task))}</p>` : ""}
      <details class="fl-notes"><summary>Read before trusting the numbers</summary><ul>${task.notes.map(n => `<li>${esc(n)}</li>`).join("")}</ul></details>`;
  }

  // ---------------------------------------------------------------- results
  function legendHtml(withCentral = true) {
    return `<div class="fl-legend" aria-label="Legend">
      <span><i class="fl-key alone"></i>Each hospital alone</span>
      <span><i class="fl-key fed"></i>Federated</span>
      ${withCentral ? '<span><i class="fl-key central"></i>All data pooled in one place (simulation only)</span>' : ""}</div>`;
  }

  function compareCells(m) {
    const better = m.higher_is_better === null ? null : (m.higher_is_better ? m.federated > m.alone : m.federated < m.alone);
    const delta = m.federated !== null && m.alone !== null ? m.federated - m.alone : null;
    const deltaHtml = delta !== null && better !== null
      ? `<span class="fl-delta ${better ? "good" : "bad"}">${fmtDelta(delta, m.format)} vs alone, ${better ? "better" : "worse"}</span>` : "";
    const row = (cls, label, value, extra = "") => `<tr${cls === "fed" ? ' class="top"' : ""}><th scope="row"><i class="fl-key ${cls}"></i>${label}</th><td class="num">${fmt(value, m.format)}</td><td>${extra}</td></tr>`;
    const direction = m.higher_is_better === null ? "" : m.higher_is_better ? "Higher is better." : "Lower is better.";
    const wins = m.higher_is_better !== null && m.hospitals_improved !== null && m.hospitals_improved !== undefined
      ? `Federated beats training alone at <b>${m.hospitals_improved} of ${m.hospitals}</b> hospitals. ` : "";
    return `<table class="data-table compare-table">
        <thead><tr><th scope="col">Training setup</th><th scope="col" class="num">${esc(m.label)}</th><th scope="col">Note</th></tr></thead>
        <tbody>
        ${row("alone", "Each hospital alone", m.alone)}
        ${row("fed", "Federated", m.federated, deltaHtml)}
        ${row("central", "Pooled in one place", m.centralized, "Simulation only: real hospitals cannot do this")}
        </tbody></table>
      <p class="fl-wins">${wins}${direction} Averages are weighted by each hospital's test patients.</p>`;
  }

  function dumbbellSvg(m, W) {
    const rows = m.per_hospital;
    const narrow = W < 520;
    const left = narrow ? 112 : 168, right = 24, top = 30, rowH = 31;
    const H = top + rows.length * rowH + 10;
    const vals = rows.flatMap(r => [r.alone, r.federated, r.centralized]).filter(Number.isFinite);
    let lo = Math.min(...vals), hi = Math.max(...vals);
    const pad = (hi - lo) * 0.12 || 0.01;
    lo -= pad; hi += pad;
    const { ticks, step } = niceTicks(lo, hi);
    const x = v => left + ((v - lo) / (hi - lo)) * (W - left - right);
    const grid = ticks.map(t => `<line x1="${x(t).toFixed(1)}" x2="${x(t).toFixed(1)}" y1="${top - 6}" y2="${H - 6}" stroke="${COLORS.grid}" stroke-width="1"/>
        <text x="${x(t).toFixed(1)}" y="${top - 12}" class="fl-axis" text-anchor="middle">${tickLabel(t, m.format, step)}</text>`).join("");
    const body = rows.map((r, i) => {
      const y = top + i * rowH + rowH / 2;
      const improved = m.higher_is_better === null ? null : (m.higher_is_better ? r.federated > r.alone : r.federated < r.alone);
      const link = Number.isFinite(r.alone) && Number.isFinite(r.federated)
        ? `<line x1="${x(r.alone).toFixed(1)}" x2="${x(r.federated).toFixed(1)}" y1="${y}" y2="${y}" stroke="${improved === false ? "#e9a99d" : "#a9c4e2"}" stroke-width="3" stroke-linecap="round"/>` : "";
      const central = Number.isFinite(r.centralized) ? `<circle cx="${x(r.centralized).toFixed(1)}" cy="${y}" r="5" fill="${COLORS.surface}" stroke="${COLORS.ink}" stroke-width="2"/>` : "";
      const alone = Number.isFinite(r.alone) ? `<circle cx="${x(r.alone).toFixed(1)}" cy="${y}" r="5" fill="${COLORS.alone}" stroke="${COLORS.surface}" stroke-width="2"/>` : "";
      const fed = Number.isFinite(r.federated) ? `<circle cx="${x(r.federated).toFixed(1)}" cy="${y}" r="5.5" fill="${COLORS.fed}" stroke="${COLORS.surface}" stroke-width="2"/>` : "";
      return `<g class="fl-row" data-i="${i}" tabindex="0" role="img" aria-label="${esc(hospitalName(r.hospital))}: alone ${fmt(r.alone, m.format)}, federated ${fmt(r.federated, m.format)}">
          <rect x="0" y="${top + i * rowH}" width="${W}" height="${rowH}" fill="transparent"/>
          <text x="${left - 14}" y="${y}" class="fl-rowlabel" text-anchor="end" dominant-baseline="central">${esc(shortName(hospitalName(r.hospital), narrow ? 15 : 24))}</text>
          ${link}${central}${alone}${fed}</g>`;
    }).join("");
    return `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" class="fl-dumbbell" role="group" aria-label="${esc(m.label)} for each hospital">${grid}${body}</svg>`;
  }

  function tableHtml(m) {
    const body = m.per_hospital.map(r => `<tr><th scope="row">${esc(hospitalName(r.hospital))}</th><td>${fmt(r.alone, m.format)}</td><td>${fmt(r.federated, m.format)}</td><td>${fmt(r.centralized, m.format)}</td></tr>`).join("");
    return `<details class="fl-table"><summary>Show as a table</summary>
      <table><thead><tr><th scope="col">Hospital</th><th scope="col">Alone</th><th scope="col">Federated</th><th scope="col">Pooled (simulation)</th></tr></thead><tbody>${body}</tbody></table></details>`;
  }

  function comparisonTable(c) {
    const rows = c.rows.map(r => {
      const same = r.federated === r.centralized;
      return `<tr><th scope="row">${esc(r.label)}</th><td>${fmt(r.federated, r.format)}</td><td>${fmt(r.centralized, r.format)}</td><td>${same ? "identical" : "differs"}</td></tr>`;
    }).join("");
    return `<p class="result-kicker">Held-out results ${c.days ? `<span class="surv-day">${esc(c.days[0])} to ${esc(c.days[1])}</span>` : ""}</p>
      <div class="heat-wrap"><table class="fl-compare-table"><thead><tr><th scope="col">Measure</th><th scope="col">Federated</th><th scope="col">Centralised pipeline</th><th scope="col"></th></tr></thead><tbody>${rows}</tbody></table></div>
      <p class="fl-wins">The detector already works on daily counts, so federating its calibration gives the same alerts as pooling the records would.</p>`;
  }

  const chartWidth = el => Math.max(300, Math.floor(el.clientWidth - 26));   // 1:1 pixels, so chart text matches the page

  function renderResults(task) {
    const el = $("#fl-results");
    if (!task.available) {
      el.innerHTML = `<p class="result-kicker">Results</p><p class="empty-state">${esc(task.hint || "No results yet.")}</p>`;
      return;
    }
    if (task.comparison) {
      el.innerHTML = comparisonTable(task.comparison);
      return;
    }
    const metric = task.metrics.find(m => m.id === metricFor[task.id]) || task.metrics[0];
    metricFor[task.id] = metric.id;
    const picker = task.metrics.length > 1
      ? `<div class="fl-metrics" role="group" aria-label="Measure">${task.metrics.map(m => `<button type="button" class="fl-metric" data-metric="${m.id}" aria-pressed="${m.id === metric.id}">${esc(m.label)}</button>`).join("")}</div>` : "";
    el.innerHTML = `<p class="result-kicker">Results on patients the models never saw</p>
      ${picker}
      <h3 class="fl-chart-title">${esc(metric.label)}</h3>
      ${compareCells(metric)}
      <p class="fl-subhead">Hospital by hospital</p>
      ${legendHtml()}
      <div class="fl-chart" id="fl-dumbbell">${dumbbellSvg(metric, chartWidth(el))}</div>
      <p class="fl-chart-note">Each line runs from a hospital training alone to the federated model${metric.higher_is_better === null ? "" : ` (blue when federated is ${metric.higher_is_better ? "higher" : "lower"}, which is better, red when it is worse)`}. The axis does not start at zero, so read the numbers, not the line lengths.</p>
      ${tableHtml(metric)}`;
    const wrap = $("#fl-dumbbell");
    const rows = metric.per_hospital;
    const show = (event, i) => {
      const r = rows[i];
      const delta = Number.isFinite(r.alone) && Number.isFinite(r.federated) && metric.higher_is_better !== null
        ? ` (${fmtDelta(r.federated - r.alone, metric.format)} vs alone)` : "";
      showTip(wrap, event, [
        { key: "fed", value: fmt(r.federated, metric.format) + delta, label: "Federated" },
        { key: "alone", value: fmt(r.alone, metric.format), label: "Each hospital alone" },
        { key: "central", value: fmt(r.centralized, metric.format), label: "Pooled (simulation)" },
      ], `${hospitalName(r.hospital)} · ${r.rows ? r.rows.toLocaleString() + " test patients" : ""}`);
    };
    wrap.querySelectorAll(".fl-row").forEach(g => {
      const i = Number(g.dataset.i);
      g.addEventListener("pointermove", e => show(e, i));
      g.addEventListener("pointerleave", () => hideTip(wrap));
      g.addEventListener("focus", () => show(focusPoint(g), i));
      g.addEventListener("blur", () => hideTip(wrap));
    });
  }

  // ---------------------------------------------------------------- replay of the recorded training
  // Average over the last full round of turns (every hospital once), so differences between hospitals cancel out;
  // undefined until the first full round has happened.
  const rolling = values => {
    const r = rollLen();
    return values.map((_, i) => {
      if (i < r - 1) return NaN;
      const window = values.slice(i - r + 1, i + 1).filter(Number.isFinite);
      return window.length ? window.reduce((a, b) => a + b, 0) / window.length : NaN;
    });
  };

  function stopReplay() {
    clearInterval(replay.timer);
    replay.timer = null;
    const button = $("#fl-play");
    if (button) button.textContent = "Play";
  }

  function replayState(task) {
    const history = task.history;
    const metric = task.metrics.find(m => m.id === replay.metric) || task.metrics[0];
    const values = history.map(h => h.metrics[metric.id]);
    return { history, metric, values, mean: rolling(values) };
  }

  function replayChart(task, W) {
    const { history, metric, values, mean } = replayState(task);
    const H = 270, ml = 52, mr = 18, mt = 16, mb = 52;
    const finite = [...values, ...mean].filter(Number.isFinite);
    let lo = Math.min(...finite), hi = Math.max(...finite);
    const pad = (hi - lo) * 0.1 || 0.01;
    lo -= pad; hi += pad;
    const { ticks, step } = niceTicks(lo, hi);
    const n = history.length;
    const x = i => ml + (n === 1 ? 0 : (i / (n - 1)) * (W - ml - mr));
    const y = v => mt + (1 - (v - lo) / (hi - lo)) * (H - mt - mb);
    const grid = ticks.map(t => `<line x1="${ml}" x2="${W - mr}" y1="${y(t).toFixed(1)}" y2="${y(t).toFixed(1)}" stroke="${COLORS.grid}"/>
      <text x="${ml - 8}" y="${y(t).toFixed(1)}" class="fl-axis" text-anchor="end" dominant-baseline="central">${tickLabel(t, metric.format, step)}</text>`).join("");
    const xTicks = [0, Math.round((n - 1) / 2), n - 1].filter((v, i, a) => a.indexOf(v) === i)
      .map(i => `<text x="${x(i).toFixed(1)}" y="${H - 28}" class="fl-axis" text-anchor="middle">${history[i].turn}</text>`).join("");
    return `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" class="fl-line" id="fl-line" role="img" aria-label="${esc(metric.label)} after each turn of the recorded training">
        ${grid}${xTicks}<text x="${(ml + W - mr) / 2}" y="${H - 8}" class="fl-axis fl-axis-title" text-anchor="middle">Turn (one hospital per turn)</text>
        <path id="fl-raw" fill="none" stroke="#c3cdc9" stroke-width="1" stroke-linejoin="round"/>
        <path id="fl-mean" fill="none" stroke="${COLORS.fed}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>
        <line id="fl-cursor" y1="${mt}" y2="${H - mb}" stroke="#c3cdc9" stroke-width="1"/>
        <circle id="fl-dot" r="5" fill="${COLORS.fed}" stroke="${COLORS.surface}" stroke-width="2"/>
        <line id="fl-hover" y1="${mt}" y2="${H - mb}" stroke="${COLORS.ink}" stroke-width="1" class="hidden"/>
        <rect id="fl-hit" x="${ml}" y="${mt}" width="${W - ml - mr}" height="${H - mt - mb}" fill="transparent"/>
      </svg>`;
  }

  function drawReplay(task) {
    const { history, metric, values, mean } = replayState(task);
    const n = history.length;
    const t = Math.min(Math.max(replay.turn, 1), n);
    const svg = $("#fl-line");
    const vb = svg.viewBox.baseVal;
    const ml = 52, mr = 18, mt = 16, mb = 52;
    const finite = [...values, ...mean].filter(Number.isFinite);
    let lo = Math.min(...finite), hi = Math.max(...finite);
    const pad = (hi - lo) * 0.1 || 0.01;
    lo -= pad; hi += pad;
    const x = i => ml + (n === 1 ? 0 : (i / (n - 1)) * (vb.width - ml - mr));
    const y = v => mt + (1 - (v - lo) / (hi - lo)) * (vb.height - mt - mb);
    const path = series => series.slice(0, t).map((v, i) => Number.isFinite(v) ? `${i && Number.isFinite(series[i - 1]) ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}` : "").join("");
    $("#fl-raw").setAttribute("d", path(values));
    $("#fl-mean").setAttribute("d", path(mean));
    const cx = x(t - 1);
    $("#fl-cursor").setAttribute("x1", cx); $("#fl-cursor").setAttribute("x2", cx);
    const dotValue = Number.isFinite(mean[t - 1]) ? mean[t - 1] : values[t - 1];
    $("#fl-dot").setAttribute("cx", cx); $("#fl-dot").setAttribute("cy", Number.isFinite(dotValue) ? y(dotValue) : y(lo));
    $("#fl-dot").setAttribute("fill", Number.isFinite(mean[t - 1]) ? COLORS.fed : COLORS.alone);
    $("#fl-scrub").value = t;
    const turn = history[t - 1];
    const iters = task.settings.local_iters;
    $("#fl-readout").innerHTML = `<b>Turn ${turn.turn} of ${n}</b> · ${esc(hospitalName(turn.hospital))} gets the shared model, adds ${iters} boosting step${iters === 1 ? "" : "s"} using only its own patients, and passes it on. ${Number.isFinite(mean[t - 1])
      ? `<span>${esc(metric.label)} now: <b>${fmt(mean[t - 1], metric.format)}</b> <small>(average of the last ${rollLen()} turns, one per hospital)</small></span>`
      : `<span>The smoothed line starts at turn ${rollLen()}, once every hospital has had a turn.</span>`}`;
    setActiveHospital(turn.hospital);
  }

  function renderReplay(task) {
    stopReplay();
    const section = $("#fl-replay");
    if (!task.available || !task.history) {
      section.classList.add("hidden");
      return;
    }
    section.classList.remove("hidden");
    const options = task.metrics.filter(m => (task.replay_metrics || []).includes(m.id));
    replay.metric = options.some(m => m.id === replay.metric) ? replay.metric : options[0].id;
    const n = task.history.length;
    replay.turn = n;
    const picker = options.length > 1
      ? `<div class="fl-metrics" role="group" aria-label="Measure">${options.map(m => `<button type="button" class="fl-metric" data-replay-metric="${m.id}" aria-pressed="${m.id === replay.metric}">${esc(m.label)}</button>`).join("")}</div>` : "";
    section.innerHTML = `<p class="result-kicker">Watch it learn <span class="surv-day">replay of the recorded training run</span></p>
      ${picker}
      <div class="fl-controls">
        <button class="secondary-button" id="fl-play" type="button">Play</button>
        <label class="fl-scrub-wrap"><span class="sr-only">Turn</span><input id="fl-scrub" type="range" min="1" max="${n}" value="${n}"></label>
      </div>
      <p class="fl-readout" id="fl-readout" aria-live="off"></p>
      <div class="fl-legend"><span><i class="fl-line-key fed"></i>Average of the last ${rollLen()} turns (every hospital once)</span><span><i class="fl-line-key raw"></i>Each turn on its own (a different hospital's patients each time)</span></div>
      <div class="fl-chart" id="fl-linewrap">${replayChart(task, chartWidth(section))}</div>
      <p class="fl-chart-note">After each turn the shared model is scored on the held-out patients of the hospital that just took its turn. Hospitals differ in how hard they are, so single turns jump around; the green line averages one full round (every hospital once), so it shows the model, not whose turn it was. These per-turn scores are not weighted by hospital size, so they differ slightly from the results above.</p>`;
    wireControls(task);
    wireChart(task);
    drawReplay(task);
  }

  function redrawReplayChart(task) {
    $("#fl-linewrap").innerHTML = replayChart(task, chartWidth($("#fl-replay")));
    wireChart(task);
    drawReplay(task);
  }

  function wireControls(task) {
    const n = task.history.length;
    $("#fl-play").addEventListener("click", () => {
      if (replay.timer) { stopReplay(); return; }
      if (replay.turn >= n) replay.turn = 0;
      $("#fl-play").textContent = "Pause";
      replay.timer = setInterval(() => {
        replay.turn += 1;
        drawReplay(task);
        if (replay.turn >= n) stopReplay();
      }, 350);
    });
    $("#fl-scrub").addEventListener("input", e => { stopReplay(); replay.turn = Number(e.target.value); drawReplay(task); });
  }

  function wireChart(task) {
    const n = task.history.length;
    const wrap = $("#fl-linewrap");
    const { history, metric, values, mean } = replayState(task);
    const hit = $("#fl-hit"), hover = $("#fl-hover"), svg = $("#fl-line");
    const nearest = event => {
      const box = svg.getBoundingClientRect();
      const px = ((event.clientX - box.left) / box.width) * svg.viewBox.baseVal.width;
      return Math.min(n - 1, Math.max(0, Math.round(((px - 52) / (svg.viewBox.baseVal.width - 52 - 18)) * (n - 1))));
    };
    hit.addEventListener("pointermove", event => {
      const i = nearest(event);
      const cx = 52 + (n === 1 ? 0 : (i / (n - 1)) * (svg.viewBox.baseVal.width - 52 - 18));
      hover.setAttribute("x1", cx); hover.setAttribute("x2", cx); hover.classList.remove("hidden");
      showTip(wrap, event, [
        { key: "fed", value: fmt(mean[i], metric.format), label: `Average of the last ${rollLen()} turns` },
        { key: "raw", value: fmt(values[i], metric.format), label: "This turn alone" },
      ], `Turn ${history[i].turn} · ${hospitalName(history[i].hospital)}`);
    });
    hit.addEventListener("pointerleave", () => { hover.classList.add("hidden"); hideTip(wrap); });
    hit.addEventListener("click", event => { stopReplay(); replay.turn = nearest(event) + 1; drawReplay(task); });
  }

  // ---------------------------------------------------------------- privacy
  function renderPrivacy() {
    const table = (items, head) => `<table class="data-table">
      <thead><tr><th scope="col">${head}</th><th scope="col">Detail</th></tr></thead>
      <tbody>${items.map(i => `<tr><th scope="row">${esc(i.label)}</th><td>${esc(i.detail)}</td></tr>`).join("")}</tbody></table>`;
    $("#fl-privacy").innerHTML = `<p class="result-kicker">Privacy and security</p>
      <div class="fl-privacy-grid">
        <div><h3 class="fl-chart-title">In place</h3>${table(data.protections, "Protection")}</div>
        <div><h3 class="fl-chart-title">Not covered</h3>${table(data.limits, "Gap")}</div>
      </div>`;
    $("#fl-scope").textContent = data.scope_note;
  }

  // ---------------------------------------------------------------- page
  function renderTask() {
    const task = data.tasks.find(t => t.id === taskId) || data.tasks[0];
    taskId = task.id;
    renderTaskButtons();
    renderNetwork(task);
    renderDetail(task);
    renderResults(task);
    renderReplay(task);
  }

  function showError(message) {
    const el = $("#fl-error");
    el.textContent = message || "";
    el.classList.toggle("hidden", !message);
  }

  async function load() {
    try {
      const response = await apiFetch("/api/federated/overview");
      const body = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(typeof body.detail === "string" ? body.detail : "The results could not be loaded.");
      data = body;
      hospitalById = Object.fromEntries(data.hospitals.map(h => [h.id, h]));
      showError("");
      renderKpis();
      renderPrivacy();
      renderTask();
    } catch (error) {
      showError(error instanceof TypeError ? "Could not reach the API. Make sure the backend is running at this address." : error.message);
    }
  }

  // ---------------------------------------------------------------- wiring
  $("#fl-tasks").addEventListener("click", event => {
    const button = event.target.closest("[data-task]");
    if (!button || button.dataset.task === taskId) return;
    taskId = button.dataset.task;
    renderTask();
  });
  $("#fl-results").addEventListener("click", event => {
    const button = event.target.closest("[data-metric]");
    if (!button) return;
    metricFor[taskId] = button.dataset.metric;
    renderResults(data.tasks.find(t => t.id === taskId));
  });
  $("#fl-replay").addEventListener("click", event => {
    const button = event.target.closest("[data-replay-metric]");
    if (!button) return;
    replay.metric = button.dataset.replayMetric;
    renderReplay(data.tasks.find(t => t.id === taskId));
  });

  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      if (!data || $("#federated-view").classList.contains("hidden")) return;
      const task = data.tasks.find(t => t.id === taskId);
      renderResults(task);
      if (task.available && task.history) redrawReplayChart(task);
    }, 150);
  });

  window.carepathTabs.register("federated", "#federated-view", load);
  document.querySelectorAll(".topnav-tab").forEach(button => button.addEventListener("click", () => {
    if (button.dataset.tab !== "federated") stopReplay();     // never keep animating a hidden tab
  }));
  if (location.hash === "#federated") document.querySelector('.topnav-tab[data-tab="federated"]').click();
})();
