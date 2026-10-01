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

function renderDiseaseCard(container, prediction, alternatives = []) {
  const diseaseName = diseaseNames[prediction.disease_id] || `Disease ${prediction.disease_id}`;
  const visibleAlternatives = alternatives.filter(item => confidencePercent(item.probability) > 0);
  const cardContent = `
    <span class="diagnosis-icon" aria-hidden="true">+</span>
    <span class="diagnosis-copy"><span class="diagnosis-name">${diseaseName}</span><span class="diagnosis-meta">${prediction.disease_id} · Most likely condition</span></span>
    <span class="diagnosis-confidence"><strong>${confidencePercent(prediction.probability)}%</strong><small>model confidence</small></span>
    ${visibleAlternatives.length ? '<span class="diagnosis-expand" aria-hidden="true">⌄</span>' : ""}`;
  const card = visibleAlternatives.length
    ? `<button class="primary-diagnosis diagnosis-toggle" type="button" aria-expanded="false">${cardContent}</button>`
    : `<div class="primary-diagnosis">${cardContent}</div>`;
  const alternativeContent = visibleAlternatives.length
    ? `<div class="other-predictions hidden">${visibleAlternatives.map(item => `
        <div class="prediction-row"><span>${diseaseNames[item.disease_id] || `Disease ${item.disease_id}`}</span><small>${confidencePercent(item.probability)}%</small></div>`).join("")}</div>`
    : "";
  container.innerHTML = card + alternativeContent;
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
    currentAssessment = {
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
}

function renderResults(recommendation) {
  const patient = currentAssessment;
  const prediction = patient.prediction;
  document.querySelector("#result-patient-name").textContent = patient.name;
  const severity = document.querySelector("#severity-result");
  severity.textContent = `${severityNames[patient.severity]} severity`;
  severity.className = `severity-result sev-${patient.severity}`;

  renderDiseaseCard(document.querySelector("#primary-diagnosis"), prediction, patient.alternatives);

  document.querySelector("#treatment-context").textContent = recommendation.disclaimer || "Ranked by predicted success and expected recovery; clinician review required.";
  const treatments = recommendation.treatments || [];
  document.querySelector("#treatment-list").innerHTML = treatments.length ? treatments.map(treatment => `
    <article class="treatment-card">
      <span class="treatment-rank">${treatment.rank}</span>
      <div><h3 class="treatment-name">${treatment.treatment_name}</h3><div class="treatment-tags">
        ${treatment.first_line ? '<span class="treatment-tag first-line">First line</span>' : ""}
        ${treatment.treatment_category ? `<span class="treatment-tag">${treatment.treatment_category}</span>` : ""}
        ${treatment.inpatient_required ? '<span class="treatment-tag">Inpatient care</span>' : ""}
        ${treatment.guideline_priority ? `<span class="treatment-tag">Guideline priority ${treatment.guideline_priority}</span>` : ""}
      </div></div>
      <div class="treatment-metrics"><strong>${confidencePercent(treatment.success_probability)}%</strong><small>predicted success</small><strong class="recovery-label">${treatment.expected_recovery_days} days</strong><small>expected recovery</small></div>
    </article>`).join("") : `<div class="no-recommendations">${recommendation.message || "No treatment options were returned for this disease and severity."}</div>`;

  const vitalData = [
    ["Age", `${patient.age} years`], ["Gender", patient.gender === "F" ? "Female" : "Male"],
    ["Temperature", patient.vitals.temperature === null ? "Not recorded" : `${patient.vitals.temperature} °C`],
    ["Heart rate", patient.vitals.heart_rate === null ? "Not recorded" : `${patient.vitals.heart_rate} bpm`],
    ["Respiratory rate", patient.vitals.respiratory_rate === null ? "Not recorded" : `${patient.vitals.respiratory_rate} /min`],
    ["SpO₂", patient.vitals.spo2 === null ? "Not recorded" : `${patient.vitals.spo2}%`],
    ["Blood pressure", patient.vitals.systolic_bp === null || patient.vitals.diastolic_bp === null ? "Not recorded" : `${patient.vitals.systolic_bp}/${patient.vitals.diastolic_bp}`],
  ];
  document.querySelector("#snapshot-vitals").innerHTML = vitalData.map(([label, value]) => `
    <div class="snapshot-metric"><small>${label}</small><strong>${value}</strong></div>`).join("");
  document.querySelector("#snapshot-symptoms").innerHTML = patient.symptoms.length
    ? patient.symptoms.map(name => `<span class="symptom-chip">${name}</span>`).join("")
    : '<span class="symptom-chip">No symptoms selected</span>';
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
document.addEventListener("click", event => {
  const toggle = event.target.closest(".diagnosis-toggle");
  if (!toggle) return;
  const alternatives = toggle.nextElementSibling;
  const expanded = toggle.getAttribute("aria-expanded") === "true";
  toggle.setAttribute("aria-expanded", String(!expanded));
  alternatives.classList.toggle("hidden", expanded);
});
document.addEventListener("keydown", event => {
  if (event.key === "/" && !["INPUT", "TEXTAREA"].includes(document.activeElement.tagName)) {
    event.preventDefault();
    document.querySelector("#symptom-search").focus();
  }
});

renderSymptoms();
ensureSession().catch(() => {});