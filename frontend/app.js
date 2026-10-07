const symptoms = [
  ["General", [["S001", "Fever"], ["S003", "Fatigue"], ["S004", "Chills"], ["S005", "Sweating"], ["S023", "Weight loss"], ["S024", "Night sweats"], ["S027", "Dehydration"]]],
  ["Respiratory", [["S006", "Dry cough"], ["S007", "Persistent cough"], ["S008", "Sore throat"], ["S009", "Runny nose"], ["S010", "Breathlessness"], ["S011", "Chest pain"]]],
  ["Pain & sensory", [["S002", "Headache"], ["S012", "Loss of taste"], ["S013", "Loss of smell"], ["S014", "Muscle pain"], ["S015", "Joint pain"], ["S016", "Retro-orbital pain"]]],
  ["Gastrointestinal", [["S018", "Nausea"], ["S019", "Vomiting"], ["S020", "Diarrhoea"], ["S021", "Abdominal pain"], ["S022", "Loss of appetite"]]],
  ["Skin & blood", [["S017", "Rash"], ["S025", "Bleeding"], ["S026", "Anaemia"], ["S028", "High fever"]]],
];

const diseaseNames = {
  D001: "Dengue", D002: "Malaria", D003: "Chikungunya", D004: "Influenza",
  D005: "COVID-19", D006: "Tuberculosis", D007: "Typhoid fever", D008: "Acute gastroenteritis",
};
const severityNames = { SV001: "Mild", SV002: "Moderate", SV003: "Severe" };
const form = document.querySelector("#patient-form");
const symptomList = document.querySelector("#symptom-list");
const severityForm = document.querySelector("#severity-form");
let toastTimer;
let currentAssessment = null;
let sessionPromise = null;
const selectedSymptomIds = new Set();

function renderSymptoms(filter = "") {
  const query = filter.trim().toLowerCase();
  const groups = symptoms.map(([group, items]) => {
    const matches = items.filter(([, name]) => name.toLowerCase().includes(query));
    if (!matches.length) return "";
    return `<div class="symptom-group"><p class="symptom-group-title">${group}</p>${matches.map(([id, name]) => `
      <label class="symptom-choice"><input type="checkbox" name="${id}" value="1" ${selectedSymptomIds.has(id) ? "checked" : ""}><span>${name}</span></label>`).join("")}</div>`;
  }).join("");
  symptomList.innerHTML = groups;
  document.querySelector("#no-symptoms").classList.toggle("hidden", Boolean(groups));
  updateSelectedCount();
}

function updateSelectedCount() {
  const count = symptomList.querySelectorAll("input:checked").length;
  document.querySelector("#selected-count").textContent = `${count} selected`;
}

function selectedSymptoms() {
  return symptoms.flatMap(([, items]) => items)
    .filter(([id]) => selectedSymptomIds.has(id))
    .map(([, name]) => name);
}

function confidencePercent(probability) {
  return Math.min(99, Math.round(probability * 100));
}

const RING = 2 * Math.PI * 34;

// Diagnosis: confidence ring + a bar for each of the top predictions (no text paragraphs).
function renderDiseaseCard(container, prediction, alternatives = []) {
  const name = id => diseaseNames[id] || `Disease ${id}`;
  const pct = confidencePercent(prediction.probability);
  const others = alternatives.filter(item => confidencePercent(item.probability) > 0);
  container.innerHTML = `
    <div class="diagnosis-visual">
      <div class="ring" role="img" aria-label="${pct}% confidence">
        <svg viewBox="0 0 80 80"><circle class="ring-track" cx="40" cy="40" r="34"/><circle class="ring-value" cx="40" cy="40" r="34" stroke-dasharray="${(RING * pct / 100).toFixed(1)} ${RING.toFixed(1)}"/></svg>
        <strong>${pct}<small>%</small></strong>
      </div>
      <div class="diagnosis-main">
        <span class="diagnosis-label">MOST LIKELY</span>
        <span class="diagnosis-name">${name(prediction.disease_id)}</span>
        ${others.length ? `<div class="alt-bars">${others.map(item => `
          <div class="alt-row"><span>${name(item.disease_id)}</span><div class="alt-track"><div class="alt-fill" style="width:${confidencePercent(item.probability)}%"></div></div><small>${confidencePercent(item.probability)}%</small></div>`).join("")}</div>` : ""}
      </div>
    </div>`;
}

function setLoading(loading) {
  const button = document.querySelector("#diagnose-button");
  button.disabled = loading;
  button.querySelector("span:first-child").textContent = loading ? "Assessing patient…" : "Diagnose patient";
}

function ensureSession() {
  if (!sessionPromise) {
    sessionPromise = fetch("/ui/session", { method: "POST", credentials: "same-origin" })
      .then(response => {
        if (!response.ok) throw new Error("The backend could not initialize a secure session.");
      })
      .catch(error => {
        sessionPromise = null;
        throw error;
      });
  }
  return sessionPromise;
}

async function apiFetch(url, options) {
  await ensureSession();
  let response = await fetch(url, options);
  if (response.status === 401) {
    sessionPromise = null;
    await ensureSession();
    response = await fetch(url, options);
  }
  return response;
}

function valueOrUnknown(formData, key) {
  const value = formData.get(key);
  return value === "" || value === null ? null : Number(value);
}

async function diagnose(event) {
  event.preventDefault();
  const error = document.querySelector("#form-error");
  error.classList.add("hidden");
  if (!form.reportValidity()) return;

  const data = new FormData(form);
  const patient = {
    age: Number(data.get("age")),
    gender: data.get("gender"),
    temperature: valueOrUnknown(data, "temperature"),
    heart_rate: valueOrUnknown(data, "heart_rate"),
    respiratory_rate: valueOrUnknown(data, "respiratory_rate"),
    spo2: valueOrUnknown(data, "spo2"),
    systolic_bp: valueOrUnknown(data, "systolic_bp"),
    diastolic_bp: valueOrUnknown(data, "diastolic_bp"),
  };
  for (const [, items] of symptoms) {
    for (const [id] of items) patient[id] = selectedSymptomIds.has(id) ? 1 : 0;
  }

  setLoading(true);
  try {
    const response = await apiFetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(patient),
    });
    const result = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(typeof result.detail === "string" ? result.detail : "The assessment could not be completed.");
    }
    const atypical = await checkAtypical(patient, result.predicted_disease.disease_id);
    currentAssessment = {
      atypical,
      name: data.get("patientName").trim(),
      age: patient.age,
      gender: patient.gender,
      vitals: patient,
      symptoms: selectedSymptoms(),
      modelPatient: patient,
      prediction: result.predicted_disease,
      alternatives: result.other_predictions || [],
    };
    renderSeverityStep();
    window.location.hash = "severity";
    document.querySelector("#intake-view").classList.add("hidden");
    document.querySelector("#severity-view").classList.remove("hidden");
    window.scrollTo({ top: 0, behavior: "smooth" });
  } catch (cause) {
    const message = cause instanceof TypeError
      ? "Could not reach the API. Make sure the backend is running at this address."
      : cause.message;
    error.textContent = message;
    error.classList.remove("hidden");
  } finally {
    setLoading(false);
  }
}

// Flags presentations that do not fit the predicted disease (possible new disease / variant).
// Never blocks the diagnosis: if this call fails, no atypical card is shown.
async function checkAtypical(patient, diseaseId) {
  try {
    const response = await apiFetch("/api/atypical/assess", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ patient, disease_id: diseaseId }),
    });
    return response.ok ? await response.json() : null;
  } catch {
    return null;
  }
}

function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

const vitalLabels = { temperature: "Temperature", heart_rate: "Heart rate", respiratory_rate: "Resp. rate", spo2: "SpO₂", systolic_bp: "Systolic BP", diastolic_bp: "Diastolic BP" };

// Turns the API's reason sentences into short chips. Only unexpected symptoms and unusual vitals are kept.
function atypicalChips(reasons) {
  const chips = [];
  for (const reason of reasons || []) {
    let m = reason.match(/^unexpected symptom: (.+?) \(seen/);
    if (m) { chips.push({ text: m[1], icon: "+" }); continue; }
    m = reason.match(/^unusual (\w+): ([\d.]+) \(typical ([\d.]+)/);
    if (m) chips.push({ text: `${vitalLabels[m[1]] || m[1]} ${m[2]}`, icon: Number(m[2]) > Number(m[3]) ? "↑" : "↓" });
  }
  return chips;
}

// Shown only when the patient has concrete atypical findings; otherwise nothing at all.
function renderAtypicalCard(container, atypical) {
  const chips = atypical && (atypical.atypical_flag || atypical.novelty_flag) ? atypicalChips(atypical.reasons) : [];
  if (!chips.length) {
    container.innerHTML = "";
    return;
  }
  container.innerHTML = `<div class="atypical-card">
    <span class="atypical-badge"><b>!</b> Atypical</span>
    ${atypical.novelty_flag ? '<span class="atypical-badge novel">New-disease signal</span>' : ""}
    <div class="atypical-chips">${chips.slice(0, 6).map(c => `<span class="atypical-chip"><i>${c.icon}</i>${escapeHtml(c.text)}</span>`).join("")}</div>
  </div>`;
}

async function buildTreatmentPlan(event) {
  event.preventDefault();
  const error = document.querySelector("#severity-error");
  error.classList.add("hidden");
  if (!severityForm.reportValidity()) return;

  const severityId = new FormData(severityForm).get("severity_id");
  const button = document.querySelector("#plan-button");
  button.disabled = true;
  button.querySelector("span:first-child").textContent = "Building treatment plan…";
  try {
    const response = await apiFetch("/recommend-treatment", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        disease_id: currentAssessment.prediction.disease_id,
        severity_id: severityId,
        patient: currentAssessment.modelPatient,
        top_k: 3,
      }),
    });
    const recommendation = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(typeof recommendation.detail === "string" ? recommendation.detail : "The treatment plan could not be generated.");
    }
    currentAssessment.severity = severityId;
    renderResults(recommendation);
    document.querySelector("#severity-view").classList.add("hidden");
    document.querySelector("#results-view").classList.remove("hidden");
    window.location.hash = "results";
    window.scrollTo({ top: 0, behavior: "smooth" });
  } catch (cause) {
    error.textContent = cause instanceof TypeError
      ? "Could not reach the API. Make sure the backend is running at this address."
      : cause.message;
    error.classList.remove("hidden");
  } finally {
    button.disabled = false;
    button.querySelector("span:first-child").textContent = "Build treatment plan";
  }
}

function renderSeverityStep() {
  const patient = currentAssessment;
  const prediction = patient.prediction;
  document.querySelector("#severity-patient-name").textContent = patient.name;
  renderDiseaseCard(document.querySelector("#step-diagnosis"), prediction, patient.alternatives);
  renderAtypicalCard(document.querySelector("#step-atypical"), patient.atypical);
}

function renderResults(recommendation) {
  const patient = currentAssessment;
  const prediction = patient.prediction;
  document.querySelector("#result-patient-name").textContent = patient.name;
  const severity = document.querySelector("#severity-result");
  severity.textContent = `${severityNames[patient.severity]} severity`;
  severity.className = `severity-result sev-${patient.severity}`;

  renderDiseaseCard(document.querySelector("#primary-diagnosis"), prediction, patient.alternatives);
  renderAtypicalCard(document.querySelector("#result-atypical"), patient.atypical);

  const treatments = recommendation.treatments || [];
  const maxDays = Math.max(1, ...treatments.map(t => Number(t.expected_recovery_days) || 0));
  document.querySelector("#treatment-list").innerHTML = treatments.length ? treatments.map(treatment => {
    const success = confidencePercent(treatment.success_probability);
    const days = Number(treatment.expected_recovery_days);
    return `
    <article class="treatment-card${treatment.rank === 1 ? " top" : ""}">
      <span class="treatment-rank">${treatment.rank}</span>
      <div class="treatment-body">
        <div class="treatment-head"><h3 class="treatment-name">${escapeHtml(treatment.treatment_name)}</h3>
          ${treatment.first_line ? '<span class="treatment-tag first-line">★ First line</span>' : ""}
          ${treatment.inpatient_required ? '<span class="treatment-tag">Inpatient</span>' : ""}</div>
        <div class="meter"><span class="meter-label">Success</span><div class="meter-track"><div class="meter-fill success" style="width:${success}%"></div></div><strong>${success}%</strong></div>
        <div class="meter"><span class="meter-label">Recovery</span><div class="meter-track"><div class="meter-fill days" style="width:${Math.max(6, days / maxDays * 100)}%"></div></div><strong>${days} d</strong></div>
      </div>
    </article>`;
  }).join("") : `<div class="no-recommendations">${escapeHtml(recommendation.message || "No treatment options for this disease and severity.")}</div>`;

  // vitals as gauges: [label, key, unit, scale min, scale max, normal low, normal high]
  const gauges = [
    ["Temperature", "temperature", "°C", 35, 41, 36.1, 37.8], ["Heart rate", "heart_rate", "bpm", 40, 160, 60, 100],
    ["Resp. rate", "respiratory_rate", "/min", 8, 40, 12, 20], ["SpO₂", "spo2", "%", 80, 100, 95, 100],
    ["Systolic BP", "systolic_bp", "mmHg", 70, 200, 90, 120], ["Diastolic BP", "diastolic_bp", "mmHg", 40, 120, 60, 80],
  ];
  const pos = (v, lo, hi) => Math.min(100, Math.max(0, (v - lo) / (hi - lo) * 100));
  document.querySelector("#snapshot-vitals").innerHTML = gauges.map(([label, key, unit, min, max, lo, hi]) => {
    const v = patient.vitals[key];
    const known = v !== null && v !== undefined;
    const out = known && (v < lo || v > hi);
    return `<div class="gauge${known ? "" : " unknown"}${out ? " out" : ""}">
      <div class="gauge-top"><span>${label}</span><strong>${known ? `${v}<small> ${unit}</small>` : "—"}</strong></div>
      <div class="gauge-track"><div class="gauge-normal" style="left:${pos(lo, min, max)}%;width:${pos(hi, min, max) - pos(lo, min, max)}%"></div>${known ? `<div class="gauge-dot" style="left:${pos(v, min, max)}%"></div>` : ""}</div>
    </div>`;
  }).join("");
  document.querySelector("#snapshot-symptoms").innerHTML = patient.symptoms.length
    ? patient.symptoms.map(name => `<span class="symptom-chip">${name}</span>`).join("")
    : '<span class="symptom-chip">None selected</span>';
}

function showIntake() {
  document.querySelector("#severity-view").classList.add("hidden");
  document.querySelector("#results-view").classList.add("hidden");
  document.querySelector("#intake-view").classList.remove("hidden");
  currentAssessment = null;
  window.location.hash = "intake";
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function showSeverity() {
  document.querySelector("#results-view").classList.add("hidden");
  document.querySelector("#severity-view").classList.remove("hidden");
  window.location.hash = "severity";
  window.scrollTo({ top: 0, behavior: "smooth" });
}

document.querySelector("#symptom-search").addEventListener("input", event => renderSymptoms(event.target.value));
symptomList.addEventListener("change", event => {
  if (event.target.matches('input[type="checkbox"]')) {
    if (event.target.checked) selectedSymptomIds.add(event.target.name);
    else selectedSymptomIds.delete(event.target.name);
  }
  updateSelectedCount();
});
form.addEventListener("submit", diagnose);
severityForm.addEventListener("submit", buildTreatmentPlan);
document.querySelector("#back-to-intake").addEventListener("click", showIntake);
document.querySelector("#back-to-severity").addEventListener("click", showSeverity);
document.querySelector("#new-assessment").addEventListener("click", () => {
  form.reset();
  severityForm.reset();
  selectedSymptomIds.clear();
  document.querySelector("#symptom-search").value = "";
  renderSymptoms(document.querySelector("#symptom-search").value);
  showIntake();
});
document.addEventListener("keydown", event => {
  if (event.key === "/" && !["INPUT", "TEXTAREA"].includes(document.activeElement.tagName)) {
    event.preventDefault();
    document.querySelector("#symptom-search").focus();
  }
});

renderSymptoms();
ensureSession().catch(() => {});