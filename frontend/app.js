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

// Diagnosis: a ranked table of the top predictions with the model's confidence in each.
function renderDiseaseCard(container, prediction, alternatives = []) {
  const name = id => diseaseNames[id] || `Disease ${id}`;
  const rows = [prediction, ...alternatives.filter(item => confidencePercent(item.probability) > 0)];
  container.innerHTML = `
    <h2 class="section-title">Predicted disease</h2>
    <div class="table-scroll"><table class="data-table dx-table">
      <thead><tr><th scope="col" class="col-rank">Rank</th><th scope="col">Disease</th><th scope="col" class="col-bar">Confidence</th></tr></thead>
      <tbody>${rows.map((item, i) => {
        const pct = confidencePercent(item.probability);
        return `<tr${i === 0 ? ' class="top"' : ""}><td>${i + 1}</td><th scope="row">${name(item.disease_id)}${i === 0 ? ' <small>Most likely</small>' : ""}</th>
          <td><span class="bar-cell"><span class="bar" aria-hidden="true"><i style="width:${pct}%"></i></span><span class="bar-value">${pct}%</span></span></td></tr>`;
      }).join("")}</tbody>
    </table></div>`;
}

function setLoading(loading) {
  const button = document.querySelector("#diagnose-button");
  button.disabled = loading;
  button.querySelector("span:first-child").textContent = loading ? "Running prediction…" : "Run prediction";
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

// Turns the API's reason sentences into short findings. Only unexpected symptoms and unusual vitals are kept.
function atypicalChips(reasons) {
  const chips = [];
  for (const reason of reasons || []) {
    let m = reason.match(/^unexpected symptom: (.+?) \(seen/);
    if (m) { chips.push({ text: m[1], kind: "Unexpected symptom" }); continue; }
    m = reason.match(/^unusual (\w+): ([\d.]+) \(typical ([\d.]+)/);
    if (m) chips.push({ text: `${vitalLabels[m[1]] || m[1]} ${m[2]}`, kind: Number(m[2]) > Number(m[3]) ? "Above typical range" : "Below typical range" });
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
  container.innerHTML = `<div class="notice">
    <h2 class="section-title">Atypical findings${atypical.novelty_flag ? ' <span class="flag-text">New-disease signal</span>' : ""}</h2>
    <table class="data-table">
      <thead><tr><th scope="col">Finding</th><th scope="col">Why it is flagged</th></tr></thead>
      <tbody>${chips.slice(0, 6).map(c => `<tr><th scope="row">${escapeHtml(c.text)}</th><td>${c.kind}</td></tr>`).join("")}</tbody>
    </table>
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
    recordAssessment(recommendation);
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

// ---- assessment log: every finished assessment is saved (never the patient name) for the Past assessments tab, and
// the server turns it into counts-only summaries for outbreak surveillance.
const NAME_KEY = "carepath_names";

function rememberName(id, name) {
  try {
    const names = JSON.parse(sessionStorage.getItem(NAME_KEY) || "{}");
    names[id] = name;
    sessionStorage.setItem(NAME_KEY, JSON.stringify(names));
  } catch { /* names are a convenience; the list falls back to the assessment number */ }
}

function sessionName(id) {
  try { return JSON.parse(sessionStorage.getItem(NAME_KEY) || "{}")[id] || ""; } catch { return ""; }
}

async function postJson(url, payload) {
  const response = await apiFetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(typeof body.detail === "string" ? body.detail : "The request failed.");
  return body;
}

async function recordAssessment(recommendation) {
  const a = currentAssessment;
  const treatment = ((recommendation.treatments || [])[0] || {}).treatment_name || null;
  try {
    if (a.logId) {                       // plan rebuilt after "Change severity": update, do not count the patient twice
      await postJson(`/api/assessments/${a.logId}/severity`, { severity_id: a.severity, treatment });
      return;
    }
    const flags = a.atypical || {};
    const vitals = Object.fromEntries(Object.keys(vitalLabels).map(key => [key, a.vitals[key]]));
    const { id } = await postJson("/api/assessments", {
      age: a.age, gender: a.gender, vitals, symptoms: [...selectedSymptomIds],
      disease_id: a.prediction.disease_id, confidence: a.prediction.probability,
      atypical_flag: Boolean(flags.atypical_flag), novelty_flag: Boolean(flags.novelty_flag), reasons: flags.reasons || [],
      severity_id: a.severity, treatment,
    });
    a.logId = id;
    rememberName(id, a.name);
  } catch (cause) {                      // the assessment on screen is unaffected, but say it was not saved
    const toast = document.querySelector("#toast");
    toast.textContent = `This assessment was not saved to Past assessments: ${cause instanceof TypeError ? "the API could not be reached." : cause.message}`;
    toast.classList.remove("hidden");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => toast.classList.add("hidden"), 8000);
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
  document.querySelector("#treatment-list").innerHTML = treatments.length ? `
    <div class="table-scroll"><table class="data-table treatment-table">
      <thead><tr><th scope="col" class="col-rank">Rank</th><th scope="col">Treatment</th><th scope="col">Notes</th><th scope="col" class="col-bar">Success rate</th><th scope="col" class="num">Recovery</th></tr></thead>
      <tbody>${treatments.map(treatment => {
        const success = confidencePercent(treatment.success_probability);
        const notes = [treatment.first_line ? "First line" : "", treatment.inpatient_required ? "Inpatient" : ""].filter(Boolean).join(", ");
        return `<tr${treatment.rank === 1 ? ' class="top"' : ""}><td>${treatment.rank}</td><th scope="row">${escapeHtml(treatment.treatment_name)}</th><td>${notes || "None"}</td>
          <td><span class="bar-cell"><span class="bar" aria-hidden="true"><i style="width:${success}%"></i></span><span class="bar-value">${success}%</span></span></td>
          <td class="num">${Number(treatment.expected_recovery_days)} days</td></tr>`;
      }).join("")}</tbody>
    </table></div>` : `<p class="empty-state">${escapeHtml(recommendation.message || "No treatment options for this disease and severity.")}</p>`;

  // vitals against normal: [label, key, unit, normal low, normal high, scale low, scale high]
  const vitals = [
    ["Temperature", "temperature", "°C", 36.1, 37.8, 34, 42], ["Heart rate", "heart_rate", "bpm", 60, 100, 30, 180],
    ["Resp. rate", "respiratory_rate", "/min", 12, 20, 5, 40], ["SpO₂", "spo2", "%", 95, 100, 70, 100],
    ["Systolic BP", "systolic_bp", "mmHg", 90, 120, 60, 200], ["Diastolic BP", "diastolic_bp", "mmHg", 60, 80, 30, 120],
  ];
  document.querySelector("#snapshot-vitals").innerHTML = `<ul class="vital-list">${vitals.map(([label, key, unit, lo, hi, min, max]) => {
    const v = patient.vitals[key];
    const known = v !== null && v !== undefined;
    const out = known && (v < lo || v > hi);
    const at = x => Math.max(0, Math.min(100, (100 * (x - min)) / (max - min)));
    const state = !known ? "Not recorded" : `${v} ${unit}${out ? ` <small>${v < lo ? "low" : "high"}</small>` : ""}`;
    const spoken = known ? `${label} ${v} ${unit}, ${out ? (v < lo ? "below" : "above") : "within"} the normal range` : `${label} not recorded`;
    return `<li class="vital-row${out ? " out" : ""}${known ? "" : " unknown"}">
      <div class="vital-head"><span class="vital-name">${label}</span><span class="vital-value">${state}</span></div>
      <div class="vital-track" role="img" aria-label="${spoken}, normal ${lo} to ${hi}">
        <i class="vital-band" style="left:${at(lo)}%;width:${at(hi) - at(lo)}%"></i>${known ? `<i class="vital-marker" style="left:${at(v)}%"></i>` : ""}
      </div>
      <div class="vital-foot"><span>${min}</span><span>Normal ${lo} to ${hi}</span><span>${max}</span></div>
    </li>`;
  }).join("")}</ul>`;
  document.querySelector("#snapshot-symptoms").innerHTML = patient.symptoms.length
    ? `<ul class="plain-list">${patient.symptoms.map(name => `<li>${name}</li>`).join("")}</ul>`
    : '<p class="muted-line">None selected</p>';
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
document.querySelector("#print-results").addEventListener("click", () => {
  document.querySelector("#print-meta").textContent = `Carepath assessment summary, printed ${new Date().toLocaleString("en-GB", { dateStyle: "long", timeStyle: "short" })}`;
  window.print();
});
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