// Autoresearch Studio & Literature Alignment Controller
const AutoresearchView = {
  trajectoryChart: null,
  isStreaming: false,

  init() {
    this.setupControls();
  },

  setupControls() {
    const btnStep = document.getElementById("btn-auto-step");
    if (btnStep) {
      btnStep.addEventListener("click", () => this.runSingleStep());
    }

    const btnBatch = document.getElementById("btn-auto-batch");
    if (btnBatch) {
      btnBatch.addEventListener("click", () => this.runBatchSearch());
    }

    const btnApply = document.getElementById("btn-auto-apply");
    if (btnApply) {
      btnApply.addEventListener("click", () => this.applyBestConfig());
    }

    const btnReset = document.getElementById("btn-auto-reset");
    if (btnReset) {
      btnReset.addEventListener("click", () => this.resetOptimizer());
    }
  },

  async render() {
    await this.updateTrajectory();
    await this.updateLeaderboard();
    await this.loadLiterature();
  },

  async updateTrajectory() {
    try {
      const res = await fetch("/api/v1/autoresearch/trajectory");
      if (!res.ok) return;
      const data = await res.json();

      this.renderTrajectoryChart(data);
    } catch (e) {
      console.error("Failed to update trajectory", e);
    }
  },

  renderTrajectoryChart(data) {
    const canvas = document.getElementById("auto-trajectory-canvas");
    if (!canvas || typeof Chart === "undefined") return;

    if (this.trajectoryChart) {
      this.trajectoryChart.destroy();
    }

    const labels = data.iterations || [];
    const bestFit = data.best_fitness || [];
    const candFit = data.candidate_fitness || [];

    this.trajectoryChart = new Chart(canvas.getContext("2d"), {
      type: "line",
      data: {
        labels: labels.map(i => `Step ${i}`),
        datasets: [
          {
            label: "Global Best Fitness",
            data: bestFit,
            borderColor: "#10b981", // emerald
            backgroundColor: "rgba(16, 185, 129, 0.1)",
            borderWidth: 2.5,
            fill: true,
            tension: 0.2
          },
          {
            label: "Candidate Fitness",
            data: candFit,
            borderColor: "#38bdf8", // cyan
            backgroundColor: "transparent",
            borderWidth: 1.5,
            borderDash: [4, 4],
            pointRadius: 4,
            pointBackgroundColor: (ctx) => {
              const idx = ctx.dataIndex;
              return (data.accepted && data.accepted[idx]) ? "#38bdf8" : "rgba(244, 63, 94, 0.5)";
            }
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: "top",
            labels: { color: "#9ca3af" }
          }
        },
        scales: {
          x: {
            grid: { color: "rgba(255, 255, 255, 0.05)" },
            ticks: { color: "#9ca3af" }
          },
          y: {
            title: { display: true, text: "Composite Fitness F(theta)", color: "#9ca3af" },
            grid: { color: "rgba(255, 255, 255, 0.05)" },
            ticks: { color: "#9ca3af" },
            min: 0,
            max: 1.0
          }
        }
      }
    });
  },

  async runSingleStep() {
    const btn = document.getElementById("btn-auto-step");
    if (btn) btn.disabled = true;

    try {
      const res = await fetch("/api/v1/autoresearch/step", { method: "POST" });
      if (!res.ok) throw new Error("Step execution failed");
      const stepData = await res.json();

      App.toast(`Step ${stepData.iteration}: ${stepData.action} (Fitness: ${stepData.candidate_fitness.toFixed(4)})`);
      await this.render();
      await App.refreshGlobalStats();
    } catch (err) {
      App.toast(`Error: ${err.message}`, "error");
    } finally {
      if (btn) btn.disabled = false;
    }
  },

  async runBatchSearch() {
    const stepsInput = document.getElementById("auto-batch-steps");
    const steps = parseInt(stepsInput?.value || 12);
    const btn = document.getElementById("btn-auto-batch");
    if (btn) {
      btn.disabled = true;
      btn.textContent = "Running Autonomous Search...";
    }

    try {
      const res = await fetch("/api/v1/autoresearch/trigger", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ max_steps: steps })
      });
      if (!res.ok) throw new Error("Batch search failed");
      const data = await res.json();

      App.toast(`Completed ${data.completed_iterations} iterations! Best fitness: ${data.best_fitness}`);
      await this.render();
      await App.refreshGlobalStats();
    } catch (err) {
      App.toast(`Error: ${err.message}`, "error");
    } finally {
      if (btn) {
        btn.disabled = false;
        btn.textContent = "Run Batch Autoresearch";
      }
    }
  },

  async applyBestConfig() {
    try {
      const res = await fetch("/api/v1/autoresearch/apply-best", { method: "POST" });
      if (!res.ok) throw new Error("Failed to apply configuration");
      const data = await res.json();
      App.toast(data.message);
      await App.refreshGlobalStats();
      if (window.ScatterMatrixView) ScatterMatrixView.render();
      if (window.NetworkGraphView) NetworkGraphView.render();
    } catch (err) {
      App.toast(`Error: ${err.message}`, "error");
    }
  },

  async resetOptimizer() {
    try {
      const res = await fetch("/api/v1/autoresearch/reset", { method: "POST" });
      if (!res.ok) throw new Error("Failed to reset");
      App.toast("Autoresearch ledger and state reset.");
      await this.render();
      await App.refreshGlobalStats();
    } catch (err) {
      App.toast(`Error: ${err.message}`, "error");
    }
  },

  async updateLeaderboard() {
    const tableBody = document.getElementById("auto-leaderboard-tbody");
    if (!tableBody) return;

    try {
      const res = await fetch("/api/v1/autoresearch/leaderboard?top_n=8");
      if (!res.ok) return;
      const leaderboard = await res.json();

      tableBody.innerHTML = "";
      leaderboard.forEach((entry, idx) => {
        const cfg = entry.configuration;
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td style="font-family:var(--font-mono); color:var(--text-muted)">#${idx + 1}</td>
          <td><span class="badge" style="background:rgba(59,130,246,0.2); color:#60a5fa;">${cfg.algorithm.toUpperCase()}</span></td>
          <td style="font-family:var(--font-mono); font-weight:700; color:#34d399;">${entry.candidate_fitness.toFixed(4)}</td>
          <td style="font-family:var(--font-mono)">${cfg.min_support.toFixed(4)}</td>
          <td style="font-family:var(--font-mono)">${cfg.min_confidence.toFixed(2)}</td>
          <td style="font-family:var(--font-mono)">${cfg.min_lift.toFixed(2)}x</td>
          <td style="font-family:var(--font-mono)">${entry.rule_count}</td>
          <td style="font-family:var(--font-mono); color:var(--text-muted)">${entry.execution_time_ms.toFixed(1)}ms</td>
        `;
        tableBody.appendChild(tr);
      });
    } catch (e) {
      console.error("Failed to update leaderboard", e);
    }
  },

  async loadLiterature() {
    const container = document.getElementById("literature-cards-container");
    if (!container) return;

    try {
      const res = await fetch("/api/v1/autoresearch/literature");
      if (!res.ok) return;
      const papers = await res.json();

      container.innerHTML = "";
      papers.forEach(p => {
        const card = document.createElement("div");
        card.className = "glass-panel";
        card.style.padding = "18px";
        card.style.display = "flex";
        card.style.flexDirection = "column";
        card.style.gap = "10px";

        card.innerHTML = `
          <div style="display:flex; justify-content:space-between; align-items:flex-start;">
            <h4 style="font-size:0.95rem; color:#f8fafc; font-weight:600;">${p.title}</h4>
            <span class="badge" style="background:rgba(139,92,246,0.2); color:#c084fc;">${p.venue} (${p.year})</span>
          </div>
          <div style="font-size:0.775rem; color:var(--text-muted);">${p.authors}</div>
          <div style="background:rgba(0,0,0,0.3); padding:10px; border-radius:6px; font-family:var(--font-mono); font-size:0.75rem; color:#38bdf8;">
            ${p.core_concept}
          </div>
          <p style="font-size:0.8rem; color:var(--text-secondary); line-height:1.4;">
            <strong>System Mapping:</strong> ${p.system_implementation_mapping}
          </p>
          <div style="font-size:0.75rem; color:#f59e0b; border-left:2px solid #f59e0b; padding-left:8px;">
            <strong>Empirical Trade-off:</strong> ${p.empirical_tradeoff}
          </div>
        `;
        container.appendChild(card);
      });
    } catch (e) {
      console.error("Failed to load literature", e);
    }
  }
};

window.AutoresearchView = AutoresearchView;
