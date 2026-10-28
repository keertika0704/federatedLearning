// Past assessments tab: every finished assessment with what the atypical-presentation model found unusual, plus a card on
// the surveillance tab showing how this clinic's own cases feed outbreak detection.
// Relies on apiFetch(), escapeHtml(), atypicalChips(), sessionName(), diseaseNames and severityNames from app.js and on
// window.carepathTabs from surveillance.js.
(() => {
  const $ = selector => document.querySelector(selector);
  const esc = escapeHtml;
  const diseaseName = id => diseaseNames[id] || id;
  let rows = [];

  async function get(path) {
    const response = await apiFetch(path);
    const body = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(typeof body.detail === "string" ? body.detail : "The request could not be completed.");
    return body;
  }

  const unreachable = error => error instanceof TypeError ? "Could not reach the API. Make sure the backend is running at this address." : error.message;

  function whenText(iso) {
    const d = new Date(iso);
    return Number.isNaN(d.getTime()) ? iso : d.toLocaleString("en-GB", { day: "numeric", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit" });
  }

  // What was unusual, in short: unexpected symptoms and out-of-range vitals, or the model's overall flag when it names none.
  function atypicalCell(a) {
    const chips = atypicalChips(a.reasons);
    if (!a.atypical_flag && !a.novelty_flag) return '<span class="muted-line">None</span>';
    const lines = chips.slice(0, 4).map(c => `<span class="sub-line">${esc(c.text)} <small>${esc(c.kind)}</small></span>`).join("");
    const head = a.novelty_flag ? '<span class="flag-text">New-disease signal</span>' : '<span class="flag-text">Atypical presentation</span>';
    return head + (lines || '<span class="sub-line">No single finding stands out.</span>');
  }

  function patientCell(a) {
    const name = sessionName(a.id);
    return `<strong>${name ? esc(name) : `Assessment #${a.id}`}</strong><span class="sub-line">${a.age} y, ${a.gender === "F" ? "female" : "male"}</span>`;
  }

  function render() {
    const onlyAtypical = $("#past-filter").value === "atypical";
    const list = rows.filter(a => !onlyAtypical || a.atypical_flag || a.novelty_flag);
    const flagged = rows.filter(a => a.atypical_flag || a.novelty_flag).length;
    $("#past-summary").textContent = rows.length
      ? `${rows.length} assessment${rows.length === 1 ? "" : "s"}, ${flagged} with atypical findings.`
      : "";
    if (!list.length) {
      $("#past-list").innerHTML = `<p class="empty-state">${rows.length ? "No assessments with atypical findings." : "No assessments yet. Finish one under Patient assessment and it appears here."}</p>`;
      return;
    }
    $("#past-list").innerHTML = `<div class="table-scroll"><table class="data-table past-table">
      <thead><tr><th scope="col">When</th><th scope="col">Patient</th><th scope="col">Predicted disease</th><th scope="col">Severity</th><th scope="col">Top treatment</th><th scope="col">Atypical findings</th></tr></thead>
      <tbody>${list.map(a => `<tr${a.atypical_flag || a.novelty_flag ? ' class="flagged"' : ""}>
        <td class="nowrap">${esc(whenText(a.created_at))}</td>
        <td>${patientCell(a)}</td>
        <td>${esc(diseaseName(a.disease_id))}<span class="sub-line">${Math.min(99, Math.round(a.confidence * 100))}% confidence</span></td>
        <td>${a.severity_id ? esc(severityNames[a.severity_id] || a.severity_id) : "–"}</td>
        <td>${a.treatment ? esc(a.treatment) : "–"}</td>
        <td>${atypicalCell(a)}</td>
      </tr>`).join("")}</tbody></table></div>`;
  }

  async function load() {
    const error = $("#past-error");
    try {
      rows = (await get("/api/assessments")).assessments;
      error.classList.add("hidden");
      render();
    } catch (cause) {
      error.textContent = unreachable(cause);
      error.classList.remove("hidden");
    }
  }

  // ---------------------------------------------------------------- surveillance card
  async function loadClinicFeed() {
    const box = $("#clinic-feed");
    try {
      const s = await get("/api/assessments/clinic-summary");
      const byDisease = s.by_disease.length
        ? `<ul class="plain-list">${s.by_disease.map(d => `<li>${esc(diseaseName(d.disease_id))}: <b>${d.cases}</b></li>`).join("")}</ul>`
        : '<p class="muted-line">No assessments in the last 7 days.</p>';
      box.innerHTML = `<h2 class="section-title">This clinic's own cases <span class="surv-day">${esc(s.since)} to ${esc(s.until)}</span></h2>
        <div class="clinic-feed">
          <div><p class="feed-number">${s.cases}</p><p class="feed-label">assessments</p></div>
          <div><p class="feed-number${s.atypical ? " flagged" : ""}">${s.atypical}</p><p class="feed-label">with atypical findings</p></div>
          <div class="feed-diseases">${byDisease}</div>
        </div>
        <p class="surv-note">Shared with the detector as the hospital node ${esc(s.hospital_id)} (${esc(s.region)}), as daily counts only and never patient records. The detector compares a site with its own earlier weeks, so this clinic only starts to count towards alerts once it has about two months of daily reports.</p>`;
    } catch {
      box.innerHTML = "";
    }
  }

  $("#past-filter").addEventListener("change", render);
  document.querySelectorAll('.topnav-tab[data-tab="surveillance"]').forEach(b => b.addEventListener("click", loadClinicFeed));
  window.carepathTabs.register("assessments", "#assessments-view", load);
})();
