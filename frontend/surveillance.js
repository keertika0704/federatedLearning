// Surveillance tab (objective B): outbreak alerts across hospitals, a hospital x disease grid, and a topbar badge.
// Relies on apiFetch() and escapeHtml() from app.js (same cookie session, same escaping).
(() => {
  const BASE = "/api/outbreak";
  const POLL_MS = 60000;
  const $ = selector => document.querySelector(selector);

  let previousView = "#intake-view";   // the assessment step to return to when leaving this tab
  let lastOpenCount = null;            // null until the first poll, so the first load never toasts
  let loaded = false;
  let toastTimer;

  async function api(path, options) {
    const response = await apiFetch(BASE + path, options);
    const body = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(typeof body.detail === "string" ? body.detail : "The request could not be completed.");
    return body;
  }

  const post = (path, payload) => api(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: payload === undefined ? undefined : JSON.stringify(payload),
  });

  function toast(message) {
    const el = $("#toast");
    el.textContent = message;
    el.classList.remove("hidden");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => el.classList.add("hidden"), 7000);
  }

  function showError(message) {
    const el = $("#surv-error");
    el.textContent = message || "";
    el.classList.toggle("hidden", !message);
  }

  // ---------------------------------------------------------------- tabs
  // "assessment" is every view that is not a tab view (the intake / severity / results steps). Other tabs register
  // themselves with window.carepathTabs.register(name, viewSelector, onEnter), e.g. the federated learning tab.
  const tabViews = { surveillance: { view: "#surveillance-view", enter: () => (loaded ? refreshAlerts() : loadPage()) } };
  let activeTab = "assessment";
  window.carepathTabs = { register: (name, view, enter) => { tabViews[name] = { view, enter }; } };

  function showTab(tab) {
    if (tab === activeTab || (tab !== "assessment" && !tabViews[tab])) return;
    const views = [...document.querySelectorAll(".shell > .view")];
    if (activeTab === "assessment") {
      const open = views.find(v => !v.classList.contains("hidden"));
      if (open) previousView = "#" + open.id;
    }
    views.forEach(v => v.classList.add("hidden"));
    if (tab === "assessment") {
      ($(previousView) || $("#intake-view")).classList.remove("hidden");
    } else {
      $(tabViews[tab].view).classList.remove("hidden");
      tabViews[tab].enter();
    }
    activeTab = tab;
    document.querySelectorAll(".topnav-tab").forEach(button => {
      const active = button.dataset.tab === tab;
      if (active) $("#crumb").textContent = button.dataset.title;
      button.classList.toggle("active", active);
      if (active) button.setAttribute("aria-current", "page"); else button.removeAttribute("aria-current");
    });
    window.scrollTo({ top: 0 });
  }

  // ---------------------------------------------------------------- alerts
  const SEVERITY_LABEL = { watch: "Watch", warning: "Warning", critical: "Critical" };
  const STATUS_LABEL = { open: "Open", acknowledged: "Acknowledged", resolved: "Resolved" };

  const sevCell = a => `<span class="sev sev-${a.severity}">${SEVERITY_LABEL[a.severity] || a.severity}</span>`;
  const siteList = a => `<ul class="site-list">${a.sites.map(s => `<li title="p = ${s.p_value}">${escapeHtml(s.hospital_name)}, ${escapeHtml(s.region)}: <b>${s.recent_count}</b> vs about ${s.expected}</li>`).join("")}</ul>`;
  const titleCell = (a, extra = "") => `<strong>${escapeHtml(a.title)}</strong>
    ${a.kind === "possible_novel" ? '<span class="sub-line flag-text">New-disease signal</span>' : ""}
    <span class="sub-line">${a.scope === "multi-region" ? "Multi-region" : "Regional"}${extra}</span>
    ${a.signature.length ? `<span class="sub-line">Excess symptoms: ${a.signature.map(s => escapeHtml(s.symptom)).join(", ")}</span>` : ""}`;

  function alertRow(a) {
    return `<tr data-id="${a.id}">
      <td>${sevCell(a)}</td>
      <td>${titleCell(a, `, sites agree ${Math.round(a.similarity * 100)}%`)}</td>
      <td>${siteList(a)}</td>
      <td>${escapeHtml(a.first_seen)}<span class="sub-line">latest ${escapeHtml(a.last_seen)}</span></td>
      <td><span class="status st-${a.status}">${STATUS_LABEL[a.status] || a.status}</span></td>
      <td class="num">${a.status === "open" ? `<button class="ack-button" type="button" data-ack="${a.id}">Acknowledge</button>` : ""}</td>
    </tr>`;
  }

  const alertsTable = list => `<div class="table-scroll"><table class="data-table alert-table">
    <thead><tr><th scope="col">Severity</th><th scope="col">Alert</th><th scope="col">Hospitals: recent vs expected cases</th><th scope="col">First seen</th><th scope="col">Status</th><th scope="col"><span class="sr-only">Action</span></th></tr></thead>
    <tbody>${list.map(alertRow).join("")}</tbody></table></div>`;

  async function refreshAlerts() {
    try {
      const { alerts } = await api("/alerts?limit=500");
      const active = alerts.filter(a => a.status !== "resolved");
      active.sort((x, y) => (x.status === "open" ? 0 : 1) - (y.status === "open" ? 0 : 1));
      $("#alert-list").innerHTML = active.length
        ? alertsTable(active)
        : '<p class="empty-state">No active alerts. Choose a day and run a scan, or see Past outbreaks for what has been dealt with.</p>';
      updateBadge(active.filter(a => a.status === "open").length);
      pastOutbreaks = alerts.filter(a => a.status === "resolved");
      renderHistory();
    } catch (error) {
      showError(error.message);
    }
  }

  function updateBadge(count) {
    const badge = $("#alert-badge");
    badge.textContent = count;
    badge.classList.toggle("hidden", count === 0);
    lastOpenCount = count;
  }

  // Lightweight poll so a new alert raised elsewhere (a node push, a scheduled scan) shows up without a click.
  async function poll() {
    try {
      const { count, alerts } = await api("/alerts?status=open");
      if (lastOpenCount !== null && count > lastOpenCount) toast(`New outbreak alert: ${alerts[0].title}`);
      updateBadge(count);
      if (!$("#surveillance-view").classList.contains("hidden")) refreshAlerts();
    } catch {
      /* the badge simply stays as it was */
    }
  }

  // ---------------------------------------------------------------- past outbreaks
  let pastOutbreaks = [];
  const DAY_MS = 86400000;
  const utc = day => Date.parse(day + "T00:00:00Z");
  const fmtDay = (ms, year) => new Date(ms).toLocaleDateString("en-GB", { day: "numeric", month: "short", ...(year ? { year: "numeric" } : {}), timeZone: "UTC" });
  const spanDays = a => Math.round((utc(a.last_seen) - utc(a.first_seen)) / DAY_MS) + 1;
  const fmtSpan = a => {
    const days = spanDays(a);
    return `${a.first_seen === a.last_seen ? fmtDay(utc(a.first_seen), true) : `${fmtDay(utc(a.first_seen))} to ${fmtDay(utc(a.last_seen), true)}`}, ${days} day${days === 1 ? "" : "s"}`;
  };

  function historyRow(a) {
    return `<tr>
      <td>${sevCell(a)}</td>
      <td>${titleCell(a)}</td>
      <td>${siteList(a)}</td>
      <td>${fmtSpan(a)}</td>
      <td>No new signal for ${detectorResolveDays} days after <span class="nowrap">${escapeHtml(a.last_seen)}</span></td>
    </tr>`;
  }

  const historyTable = list => `<div class="table-scroll"><table class="data-table alert-table">
    <thead><tr><th scope="col">Severity</th><th scope="col">Outbreak</th><th scope="col">Hospitals: recent vs expected cases</th><th scope="col">Period</th><th scope="col">Resolution</th></tr></thead>
    <tbody>${list.map(historyRow).join("")}</tbody></table></div>`;

  let detectorResolveDays = 14;

  function renderTimeline(list) {
    if (!list.length) { $("#history-timeline").innerHTML = ""; return; }
    const first = new Date(Math.min(...list.map(a => utc(a.first_seen))));
    const last = new Date(Math.max(...list.map(a => utc(a.last_seen))));
    const start = Date.UTC(first.getUTCFullYear(), first.getUTCMonth(), 1);
    const end = Date.UTC(last.getUTCFullYear(), last.getUTCMonth() + 1, 1);
    const pct = ms => (100 * (ms - start)) / (end - start);
    const months = [];
    for (let m = new Date(start); m.getTime() < end; m = new Date(Date.UTC(m.getUTCFullYear(), m.getUTCMonth() + 1, 1))) {
      const label = m.toLocaleDateString("en-GB", { month: "short", timeZone: "UTC" });
      months.push(`<span class="tl-month" style="left:${pct(m.getTime())}%">${label}${m.getUTCMonth() === 0 || months.length === 0 ? ` ${m.getUTCFullYear()}` : ""}</span>`);
    }
    const names = [...new Set(list.map(a => a.disease_name))];
    const rows = names.map(name => {
      const bars = list.filter(a => a.disease_name === name).map(a => {
        const left = pct(utc(a.first_seen)), width = Math.max(pct(utc(a.last_seen) + DAY_MS) - left, 0.8);
        return `<i class="tl-bar sev-${a.severity}" style="left:${left}%;width:${width}%" title="${escapeHtml(`${a.title} (${fmtSpan(a)})`)}"></i>`;
      }).join("");
      return `<div class="tl-row"><span class="tl-name">${escapeHtml(name)}</span><div class="tl-track">${bars}</div></div>`;
    }).join("");
    $("#history-timeline").innerHTML = `<div class="tl-row tl-axis"><span class="tl-name"></span><div class="tl-track">${months.join("")}</div></div>${rows}`;
  }

  function renderHistory() {
    const select = $("#history-disease");
    const names = [...new Set(pastOutbreaks.map(a => a.disease_name))].sort();
    if ([...select.options].slice(1).map(o => o.value).join("|") !== names.join("|")) {
      const chosen = select.value;
      select.innerHTML = '<option value="">All diseases</option>' + names.map(n => `<option>${escapeHtml(n)}</option>`).join("");
      select.value = names.includes(chosen) ? chosen : "";
    }
    const count = $("#history-count");
    count.textContent = pastOutbreaks.length;
    count.classList.toggle("hidden", pastOutbreaks.length === 0);

    const list = pastOutbreaks.filter(a => !select.value || a.disease_name === select.value)
      .sort((x, y) => (y.first_seen > x.first_seen) - (y.first_seen < x.first_seen));
    const critical = list.filter(a => a.severity === "critical").length;
    const days = list.reduce((sum, a) => sum + spanDays(a), 0);
    $("#history-summary").textContent = list.length
      ? `${list.length} past outbreak${list.length === 1 ? "" : "s"} dealt with, ${critical} critical, ${days} days with an active signal in total.`
      : "";
    renderTimeline([...list].reverse());
    $("#history-list").innerHTML = list.length
      ? historyTable(list)
      : '<p class="empty-state">No past outbreaks yet. An alert moves here once it has had no new signal for two weeks.</p>';
  }

  function showSubtab(name) {
    document.querySelectorAll(".surv-subtab").forEach(b => {
      const on = b.dataset.subtab === name;
      b.classList.toggle("active", on);
      b.setAttribute("aria-selected", on);
    });
    $("#surv-live").classList.toggle("hidden", name !== "live");
    $("#surv-history").classList.toggle("hidden", name !== "history");
    if (name === "history") refreshAlerts();
  }

  // ---------------------------------------------------------------- hospital x disease grid
  function level(cell) {
    if (cell.signal) return 2;
    return cell.ratio !== null && cell.ratio >= 1.3 && cell.count >= 3 ? 1 : 0;
  }

  function renderTrends(data) {
    $("#trend-day").textContent = `week ending ${data.day}`;
    const head = `<thead><tr><th scope="col">Hospital</th>${data.diseases.map(d => `<th scope="col">${escapeHtml(d.disease_name)}</th>`).join("")}<th scope="col" title="Share of cases flagged atypical by objective A">Atypical</th></tr></thead>`;
    const rows = data.hospitals.map(h => {
      const cells = data.diseases.map(d => {
        const c = h.cells[d.disease_id];
        const text = c.count === null ? "–" : c.count;
        const tip = c.count === null ? "Not enough history" : `${c.count} cases vs ~${c.expected} normal (p = ${c.p_value})`;
        return `<td class="heat-cell lvl-${level(c)}" title="${escapeHtml(tip)}">${text}</td>`;
      }).join("");
      const a = h.atypical;
      const share = a.of ? `${Math.round(100 * a.count / a.of)}%` : "–";
      const atip = a.of ? `${a.count} of ${a.of} cases vs ~${Math.round(100 * a.expected_share)}% normal (p = ${a.p_value})` : "Not enough history";
      return `<tr class="${h.reported ? "" : "no-report"}"><th scope="row">${escapeHtml(h.hospital_name)} <small>${escapeHtml(h.region)}</small></th>${cells}<td class="heat-cell lvl-${a.signal ? 2 : 0}" title="${escapeHtml(atip)}">${share}</td></tr>`;
    }).join("");
    $("#heat-grid").innerHTML = `${head}<tbody>${rows}</tbody>`;
  }

  async function refreshTrends(day) {
    try {
      renderTrends(await api(`/trends?as_of=${encodeURIComponent(day)}`));
    } catch (error) {
      showError(error.message);
    }
  }

  // ---------------------------------------------------------------- scanning
  function setBusy(busy) {
    $("#surv-run").disabled = busy;
    $("#surv-next").disabled = busy;
    $("#surv-run").textContent = busy ? "Scanning…" : "Run scan";
  }

  async function runScan(day) {
    showError("");
    setBusy(true);
    try {
      const result = await post("/scan", { as_of: day });
      const fresh = result.alerts.filter(a => a.change === "new" || a.change === "escalated");
      $("#surv-status").textContent = result.alerts.length
        ? `As of ${result.as_of}: ${result.alerts.length} active alert${result.alerts.length === 1 ? "" : "s"}${result.new ? `, ${result.new} new` : ""}.`
        : `As of ${result.as_of}: nothing unusual across the hospitals.`;
      if (fresh.length) toast(`${fresh[0].change === "new" ? "New" : "Escalated"} outbreak alert: ${fresh[0].title}`);
      await Promise.all([refreshAlerts(), refreshTrends(result.as_of)]);
    } catch (error) {
      showError(error.message);
    } finally {
      setBusy(false);
    }
  }

  function shiftDay(day, delta) {
    const d = new Date(day + "T00:00:00Z");
    d.setUTCDate(d.getUTCDate() + delta);
    return d.toISOString().slice(0, 10);
  }

  async function loadPage() {
    loaded = true;
    try {
      const health = await api("/health");
      const { from, to, hospitals } = health.summaries;
      detectorResolveDays = (health.thresholds && health.thresholds.resolve_after_days) || detectorResolveDays;
      const input = $("#surv-date");
      if (!from) {
        showError("No hospital summaries yet. Build them with: python models/objectiveB/node.py --build");
        await refreshAlerts();
        return;
      }
      input.min = from;
      input.max = to;
      input.value = input.value || to;
      $("#surv-status").textContent = `${hospitals} hospitals reporting, ${from} to ${to}.`;
      const m = health.metrics && health.metrics.test;
      $("#surv-note").textContent = m
        ? `On a held-out period the detector sent ${m.notifications} notifications (${Math.round(100 * m.notification_precision)}% correct) and caught ${Math.round(100 * m.linked_alert_recall)}% of outbreaks that spread between hospitals, a median ${m.median_delay_days} days after they began. Small or slow outbreaks can go unnoticed.`
        : "The detector is running on default thresholds. Train it with python models/objectiveB/train.py.";
      await Promise.all([refreshAlerts(), refreshTrends(input.value)]);
    } catch (error) {
      loaded = false;
      showError(error instanceof TypeError ? "Could not reach the API. Make sure the backend is running at this address." : error.message);
    }
  }

  // ---------------------------------------------------------------- wiring
  document.querySelectorAll(".topnav-tab").forEach(button => button.addEventListener("click", () => showTab(button.dataset.tab)));
  $("#surv-run").addEventListener("click", () => { if ($("#surv-date").value) runScan($("#surv-date").value); });
  $("#surv-next").addEventListener("click", () => {
    const input = $("#surv-date");
    if (!input.value) return;
    const next = shiftDay(input.value, 1);
    if (input.max && next > input.max) { showError(`No data after ${input.max}.`); return; }
    input.value = next;
    runScan(next);
  });
  document.querySelectorAll(".surv-subtab").forEach(b => b.addEventListener("click", () => showSubtab(b.dataset.subtab)));
  $("#history-disease").addEventListener("change", renderHistory);
  $("#alert-list").addEventListener("click", async event => {
    const button = event.target.closest("[data-ack]");
    if (!button) return;
    button.disabled = true;
    try {
      await post(`/alerts/${button.dataset.ack}/ack`);
      await refreshAlerts();
    } catch (error) {
      button.disabled = false;
      showError(error.message);
    }
  });

  poll();
  setInterval(poll, POLL_MS);
})();
