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
      button.classList.toggle("active", active);
      if (active) button.setAttribute("aria-current", "page"); else button.removeAttribute("aria-current");
    });
    window.scrollTo({ top: 0 });
  }

  // ---------------------------------------------------------------- alerts
  const SEVERITY_LABEL = { watch: "Watch", warning: "Warning", critical: "Critical" };
  const STATUS_LABEL = { open: "Open", acknowledged: "Acknowledged", resolved: "Resolved" };

  function alertCard(a) {
    const sites = a.sites.map(s => `<span class="site-chip" title="p = ${s.p_value}">${escapeHtml(s.hospital_name)} <small>${escapeHtml(s.region)}</small><b>${s.recent_count}</b><small>vs ~${s.expected}</small></span>`).join("");
    const symptoms = a.signature.map(s => `<span class="atypical-chip"><i>+</i>${escapeHtml(s.symptom)}</span>`).join("");
    const novel = a.kind === "possible_novel";
    return `<article class="alert-card sev-${a.severity}${a.status === "resolved" ? " is-resolved" : ""}" data-id="${a.id}">
      <div class="alert-head">
        <span class="sev-pill sev-${a.severity}">${SEVERITY_LABEL[a.severity] || a.severity}</span>
        ${novel ? '<span class="atypical-badge novel">New-disease signal</span>' : ""}
        <span class="alert-scope">${a.scope === "multi-region" ? "Multi-region" : "Regional"}</span>
        <span class="alert-status st-${a.status}">${STATUS_LABEL[a.status] || a.status}</span>
      </div>
      <h3 class="alert-title">${escapeHtml(a.title)}</h3>
      <div class="alert-sites">${sites}</div>
      ${symptoms ? `<div class="alert-symptoms"><span class="alert-label">Excess symptoms</span>${symptoms}</div>` : ""}
      <div class="alert-foot">
        <span>Sites agree ${Math.round(a.similarity * 100)}% · first seen ${escapeHtml(a.first_seen)} · latest ${escapeHtml(a.last_seen)}</span>
        ${a.status === "open" ? `<button class="ack-button" type="button" data-ack="${a.id}">Acknowledge</button>` : ""}
      </div>
    </article>`;
  }

  async function refreshAlerts() {
    try {
      const { alerts } = await api("/alerts");
      const rank = { open: 0, acknowledged: 1, resolved: 2 };
      alerts.sort((x, y) => rank[x.status] - rank[y.status]);
      $("#alert-list").innerHTML = alerts.length
        ? alerts.map(alertCard).join("")
        : '<p class="empty-state">No alerts yet. Choose a day and run a scan.</p>';
      updateBadge(alerts.filter(a => a.status === "open").length);
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

  // ---------------------------------------------------------------- hospital x disease grid
  function level(cell) {
    if (cell.signal) return 2;
    return cell.ratio !== null && cell.ratio >= 1.3 && cell.count >= 3 ? 1 : 0;
  }

  function renderTrends(data) {
    $("#trend-day").textContent = `· week ending ${data.day}`;
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
