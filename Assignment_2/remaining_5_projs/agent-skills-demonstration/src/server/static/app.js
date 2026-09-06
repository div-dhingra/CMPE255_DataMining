/**
 * Agent Skills Demonstration Engine - Client Application Logic
 */

let allSkills = [];
let currentCategoryFilter = "all";

// Tab Switching
function switchTab(tabId) {
  document.querySelectorAll(".tab-btn").forEach(btn => btn.classList.remove("active"));
  document.querySelectorAll(".tab-content").forEach(content => content.classList.remove("active"));

  const targetTab = document.getElementById(`tab-${tabId}`);
  if (targetTab) {
    targetTab.classList.add("active");
  }

  // Set active button
  event.target.classList.add("active");
}

// Modal Handling
function openModal(title, content) {
  document.getElementById("modal-title").innerText = title;
  document.getElementById("modal-body-content").innerText = typeof content === "object" ? JSON.stringify(content, null, 2) : content;
  document.getElementById("modal").style.display = "flex";
}

function closeModal() {
  document.getElementById("modal").style.display = "none";
}

// Load CRISP-DM Phases
async function loadPhases() {
  try {
    const res = await fetch("/api/crisp/phases");
    const data = await res.json();
    const container = document.getElementById("phases-container");
    container.innerHTML = "";

    data.phases.forEach(phase => {
      const card = document.createElement("div");
      card.className = "phase-card";
      card.innerHTML = `
        <div>
          <div class="phase-header">
            <span class="phase-name">${phase.name}</span>
            <span class="badge badge-green">✓ COMPLETE</span>
          </div>
          <p style="font-size: 0.85rem; color: var(--text-secondary); margin-bottom: 0.5rem;">${phase.description}</p>
          <div class="skill-chips">
            ${phase.skills.map(s => `<span class="chip">${s}</span>`).join("")}
          </div>
        </div>
        <div style="display: flex; gap: 0.5rem; margin-top: 1rem;">
          <button class="btn btn-sm btn-secondary" onclick="viewPhaseArtifacts('${phase.phase_id}', 'md')">📄 View Report</button>
          <button class="btn btn-sm btn-secondary" onclick="viewPhaseArtifacts('${phase.phase_id}', 'json')">⚙️ View JSON</button>
        </div>
      `;
      container.appendChild(card);
    });
  } catch (err) {
    console.error("Error loading phases:", err);
  }
}

async function viewPhaseArtifacts(phaseId, format) {
  try {
    const res = await fetch(`/api/crisp/${phaseId}`);
    const data = await res.json();
    if (format === "md") {
      openModal(`${data.metadata.name} - Markdown Report`, data.markdown_report || "No markdown report found.");
    } else {
      openModal(`${data.metadata.name} - JSON Artifact Data`, data.artifact_data || {});
    }
  } catch (err) {
    alert("Error fetching phase artifact: " + err);
  }
}

// Load Skills Catalog
async function loadSkills() {
  try {
    const res = await fetch("/api/skills");
    const data = await res.json();
    allSkills = data.skills;
    renderSkillsTable();
  } catch (err) {
    console.error("Error loading skills:", err);
  }
}

function setCategoryFilter(cat) {
  currentCategoryFilter = cat;
  document.querySelectorAll(".filter-btn").forEach(btn => btn.classList.remove("active"));
  event.target.classList.add("active");
  renderSkillsTable();
}

function filterSkills() {
  renderSkillsTable();
}

function renderSkillsTable() {
  const query = document.getElementById("skill-search").value.toLowerCase();
  const tbody = document.getElementById("skills-table-body");
  tbody.innerHTML = "";

  const filtered = allSkills.filter(skill => {
    const matchCat = currentCategoryFilter === "all" || skill.category === currentCategoryFilter;
    const matchQuery = skill.name.toLowerCase().includes(query) ||
                       skill.id.toLowerCase().includes(query) ||
                       skill.description.toLowerCase().includes(query) ||
                       skill.author.toLowerCase().includes(query);
    return matchCat && matchQuery;
  });

  document.getElementById("skill-count-badge").innerText = `${filtered.length} Skills Shown`;

  filtered.forEach(skill => {
    const tr = document.createElement("tr");
    const catBadgeClass = skill.author === "param087" ? "badge-blue" : "badge-purple";

    tr.innerHTML = `
      <td>
        <div style="font-weight: 600; color: var(--text-primary)">${skill.name}</div>
        <div style="font-size: 0.75rem; color: var(--text-secondary); font-family: monospace;">${skill.id}</div>
      </td>
      <td><span class="badge ${catBadgeClass}">${skill.category}</span></td>
      <td><span style="font-size: 0.8rem; color: var(--text-secondary)">${skill.crisp_dm_phase}</span></td>
      <td><span class="badge" style="background-color: #334155">${skill.author}</span></td>
      <td>
        <div style="display: flex; gap: 0.4rem;">
          <button class="btn btn-sm btn-secondary" onclick="viewSkillSpec('${skill.id}')">📖 Spec</button>
          <button class="btn btn-sm" onclick="runSkillLive('${skill.id}')">▶ Run</button>
        </div>
      </td>
    `;
    tbody.appendChild(tr);
  });
}

async function viewSkillSpec(skillId) {
  try {
    const res = await fetch(`/api/skills/${skillId}`);
    const data = await res.json();
    openModal(`Skill Spec: ${data.name} (${data.id})`, data.markdown_specification);
  } catch (err) {
    alert("Error fetching skill spec: " + err);
  }
}

async function runSkillLive(skillId) {
  try {
    const res = await fetch(`/api/skills/${skillId}/run`, { method: "POST" });
    const data = await res.json();
    openModal(`Live Skill Run: ${data.name}`, data);
  } catch (err) {
    alert("Error executing skill: " + err);
  }
}

// Preset Archetypes
function loadArchetype(type) {
  if (type === "high_risk") {
    document.getElementById("sim-tenure").value = 2;
    document.getElementById("tenure-val").innerText = "2";
    document.getElementById("sim-monthly").value = 89.5;
    document.getElementById("monthly-val").innerText = "89.50";
    document.getElementById("sim-contract").value = "Month-to-month";
    document.getElementById("sim-internet").value = "Fiber optic";
    document.getElementById("sim-payment").value = "Electronic check";
    document.getElementById("sim-techsupport").value = "No";
    document.getElementById("sim-security").value = "No";
    document.getElementById("sim-backup").value = "No";
    document.getElementById("sim-paperless").value = "Yes";
  } else if (type === "loyal_multiplay") {
    document.getElementById("sim-tenure").value = 48;
    document.getElementById("tenure-val").innerText = "48";
    document.getElementById("sim-monthly").value = 65.0;
    document.getElementById("monthly-val").innerText = "65.00";
    document.getElementById("sim-contract").value = "Two year";
    document.getElementById("sim-internet").value = "DSL";
    document.getElementById("sim-payment").value = "Bank transfer (automatic)";
    document.getElementById("sim-techsupport").value = "Yes";
    document.getElementById("sim-security").value = "Yes";
    document.getElementById("sim-backup").value = "Yes";
    document.getElementById("sim-paperless").value = "No";
  } else if (type === "moderate_dsl") {
    document.getElementById("sim-tenure").value = 14;
    document.getElementById("tenure-val").innerText = "14";
    document.getElementById("sim-monthly").value = 54.0;
    document.getElementById("monthly-val").innerText = "54.00";
    document.getElementById("sim-contract").value = "One year";
    document.getElementById("sim-internet").value = "DSL";
    document.getElementById("sim-payment").value = "Credit card (automatic)";
    document.getElementById("sim-techsupport").value = "No";
    document.getElementById("sim-security").value = "Yes";
    document.getElementById("sim-backup").value = "No";
    document.getElementById("sim-paperless").value = "Yes";
  }
  runPrediction();
}

// Real-Time Inference
async function runPrediction() {
  const payload = {
    gender: "Female",
    SeniorCitizen: 0,
    Partner: "No",
    Dependents: "No",
    tenure: parseFloat(document.getElementById("sim-tenure").value),
    PhoneService: "Yes",
    MultipleLines: "No",
    InternetService: document.getElementById("sim-internet").value,
    OnlineSecurity: document.getElementById("sim-security").value,
    OnlineBackup: document.getElementById("sim-backup").value,
    DeviceProtection: "No",
    TechSupport: document.getElementById("sim-techsupport").value,
    StreamingTV: "Yes",
    StreamingMovies: "Yes",
    Contract: document.getElementById("sim-contract").value,
    PaperlessBilling: document.getElementById("sim-paperless").value,
    PaymentMethod: document.getElementById("sim-payment").value,
    MonthlyCharges: parseFloat(document.getElementById("sim-monthly").value),
    TotalCharges: parseFloat(document.getElementById("sim-tenure").value) * parseFloat(document.getElementById("sim-monthly").value)
  };

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const data = await res.json();

    // Update UI
    document.getElementById("sim-score-pct").innerText = `${data.churn_score_pct}%`;
    const circle = document.getElementById("sim-score-circle");
    circle.style.borderColor = data.badge_color;

    const badge = document.getElementById("sim-risk-badge");
    badge.innerText = `${data.risk_tier} RISK`;
    badge.style.backgroundColor = data.badge_color;
    badge.style.color = "#0f172a";

    // Top drivers
    const driversList = document.getElementById("sim-risk-drivers");
    driversList.innerHTML = data.top_contributing_risk_factors.map(d => `<li>${d}</li>`).join("");

    // Recommended action
    const actionBox = document.getElementById("sim-action-box");
    const act = data.recommended_retention_action;
    actionBox.innerHTML = `
      <strong>${act.campaign_action}</strong><br>
      <span style="color: var(--text-secondary)">${act.offer_details}</span><br>
      <span style="font-size: 0.75rem; color: var(--accent-blue)">Channel: ${act.recommended_channel} | Impact: ${act.projected_churn_reduction}</span>
    `;

  } catch (err) {
    console.error("Error predicting churn:", err);
  }
}

// Initialize on page load
window.addEventListener("DOMContentLoaded", () => {
  loadPhases();
  loadSkills();
  runPrediction();
});
