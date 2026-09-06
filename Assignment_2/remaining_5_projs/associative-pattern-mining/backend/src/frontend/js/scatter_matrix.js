// Rules Matrix Scatter Plot & Mining Studio View
const ScatterMatrixView = {
  chart: null,
  cachedMatrixData: [],

  init() {
    this.setupMiningForm();
    this.setupFilters();
  },

  setupMiningForm() {
    const btn = document.getElementById("btn-run-mining");
    if (!btn) return;

    btn.addEventListener("click", async () => {
      const alg = document.getElementById("mine-alg").value;
      const supp = parseFloat(document.getElementById("mine-supp").value) || 0.03;
      const conf = parseFloat(document.getElementById("mine-conf").value) || 0.40;
      const lift = parseFloat(document.getElementById("mine-lift").value) || 1.10;
      const maxLen = parseInt(document.getElementById("mine-maxlen").value) || 3;
      const pruneRedundant = document.getElementById("mine-prune").checked;

      btn.disabled = true;
      btn.textContent = "Mining...";

      try {
        const res = await fetch("/api/v1/mine", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            algorithm: alg,
            min_support: supp,
            min_confidence: conf,
            min_lift: lift,
            max_length: maxLen,
            prune_redundant: pruneRedundant,
            limit: 100
          })
        });

        if (!res.ok) throw new Error("Mining failed");
        const data = await res.json();
        App.toast(`Mined ${data.total_rules_found} rules using ${data.algorithm} in ${data.execution_time_ms}ms!`);

        await App.refreshGlobalStats();
        await this.render();
      } catch (err) {
        App.toast(`Error: ${err.message}`, "error");
      } finally {
        btn.disabled = false;
        btn.textContent = "Run Mining Execution";
      }
    });
  },

  setupFilters() {
    const searchInput = document.getElementById("matrix-search");
    const sortSelect = document.getElementById("matrix-sort");

    if (searchInput) {
      searchInput.addEventListener("input", () => this.filterAndRenderTable());
    }
    if (sortSelect) {
      sortSelect.addEventListener("change", () => this.filterAndRenderTable());
    }
  },

  async render() {
    try {
      const res = await fetch("/api/v1/mine/matrix?limit=200");
      if (!res.ok) return;
      const json = await res.json();
      this.cachedMatrixData = json.data || [];

      this.renderScatterChart();
      this.filterAndRenderTable();
    } catch (e) {
      console.error("Failed to render scatter matrix", e);
    }
  },

  renderScatterChart() {
    const canvas = document.getElementById("scatter-chart-canvas");
    if (!canvas || typeof Chart === "undefined") return;

    if (this.chart) {
      this.chart.destroy();
    }

    const points = this.cachedMatrixData.map(r => ({
      x: r.support * 100,
      y: r.confidence * 100,
      r: Math.max(5, Math.min(18, (r.lift - 0.8) * 3.5)),
      rule: r.rule,
      lift: r.lift,
      kulczynski: r.kulczynski,
      conviction: r.conviction,
      zhang: r.zhang
    }));

    this.chart = new Chart(canvas.getContext("2d"), {
      type: "bubble",
      data: {
        datasets: [{
          label: "Association Rules",
          data: points,
          backgroundColor: (ctx) => {
            const raw = ctx.raw;
            if (!raw) return "rgba(56, 189, 248, 0.6)";
            // Color based on Lift
            if (raw.lift >= 3.0) return "rgba(244, 63, 94, 0.75)"; // rose
            if (raw.lift >= 2.0) return "rgba(168, 85, 247, 0.75)"; // purple
            if (raw.lift >= 1.4) return "rgba(56, 189, 248, 0.75)"; // blue
            return "rgba(16, 185, 129, 0.75)"; // emerald
          },
          borderColor: "rgba(255, 255, 255, 0.2)",
          borderWidth: 1
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              title: (items) => items[0].raw.rule,
              label: (ctx) => {
                const r = ctx.raw;
                return [
                  `Support: ${r.x.toFixed(2)}%`,
                  `Confidence: ${r.y.toFixed(1)}%`,
                  `Lift: ${r.lift.toFixed(2)}x`,
                  `Kulczynski: ${r.kulczynski.toFixed(3)}`,
                  `Conviction: ${r.conviction.toFixed(2)}`
                ];
              }
            }
          }
        },
        scales: {
          x: {
            title: { display: true, text: "Support (%)", color: "#9ca3af" },
            grid: { color: "rgba(255, 255, 255, 0.05)" },
            ticks: { color: "#9ca3af" }
          },
          y: {
            title: { display: true, text: "Confidence (%)", color: "#9ca3af" },
            grid: { color: "rgba(255, 255, 255, 0.05)" },
            ticks: { color: "#9ca3af" },
            min: 0,
            max: 100
          }
        }
      }
    });
  },

  filterAndRenderTable() {
    const tableBody = document.getElementById("rules-table-tbody");
    if (!tableBody) return;

    const searchTerm = (document.getElementById("matrix-search")?.value || "").toLowerCase().trim();
    const sortBy = document.getElementById("matrix-sort")?.value || "lift";

    let rules = [...this.cachedMatrixData];

    if (searchTerm) {
      rules = rules.filter(r => r.rule.toLowerCase().includes(searchTerm));
    }

    rules.sort((a, b) => {
      if (sortBy === "lift") return b.lift - a.lift;
      if (sortBy === "confidence") return b.confidence - a.confidence;
      if (sortBy === "support") return b.support - a.support;
      if (sortBy === "kulczynski") return b.kulczynski - a.kulczynski;
      if (sortBy === "conviction") return b.conviction - a.conviction;
      if (sortBy === "zhang") return b.zhang - a.zhang;
      return 0;
    });

    tableBody.innerHTML = "";
    rules.slice(0, 50).forEach(r => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td style="font-weight:600; font-family:var(--font-mono); color:#38bdf8;">${r.rule}</td>
        <td style="font-family:var(--font-mono)">${(r.support * 100).toFixed(2)}%</td>
        <td style="font-family:var(--font-mono); color:#60a5fa;">${(r.confidence * 100).toFixed(1)}%</td>
        <td><span class="badge badge-lift">${r.lift.toFixed(2)}x</span></td>
        <td><span class="badge badge-kulc">${r.kulczynski.toFixed(3)}</span></td>
        <td style="font-family:var(--font-mono); color:var(--text-muted)">${r.conviction.toFixed(2)}</td>
        <td style="font-family:var(--font-mono); color:var(--text-muted)">${r.zhang.toFixed(3)}</td>
        <td style="font-family:var(--font-mono); color:var(--text-muted)">${r.imbalance_ratio.toFixed(2)}</td>
      `;
      tableBody.appendChild(tr);
    });
  }
};

window.ScatterMatrixView = ScatterMatrixView;
