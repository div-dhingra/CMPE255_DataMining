// CRISP-DM View Controller
const CrispDmView = {
  async init() {
    await this.refresh();
    this.setupSyntheticGenerator();
  },

  async refresh() {
    await this.loadLifecycle();
    await this.loadTopItems();
  },

  async loadLifecycle() {
    const container = document.getElementById("crisp-phases-container");
    if (!container) return;

    try {
      const res = await fetch("/api/v1/crisp-dm");
      if (!res.ok) return;
      const data = await res.json();

      container.innerHTML = "";
      data.phases.forEach(phase => {
        const card = document.createElement("div");
        card.className = "glass-panel phase-card";

        let metricsHtml = "";
        for (const [k, v] of Object.entries(phase.key_metrics || {})) {
          const displayVal = Array.isArray(v) ? v.join(", ") : v;
          metricsHtml += `<div><span style="color:var(--text-muted)">${k}:</span> <span style="color:var(--accent-cyan)">${displayVal}</span></div>`;
        }

        let artifactsHtml = (phase.artifacts || []).map(a => `<code style="font-size:0.75rem; color:#a78bfa;">${a}</code>`).join(", ");

        card.innerHTML = `
          <div class="phase-header">
            <span class="phase-num">Phase 0${phase.phase_id}</span>
            <span class="badge-completed">● ${phase.status}</span>
          </div>
          <div class="phase-title">${phase.name}</div>
          <p class="phase-desc">${phase.description}</p>
          <div class="phase-metrics">
            ${metricsHtml}
            <div style="margin-top:6px; border-top:1px solid rgba(255,255,255,0.06); padding-top:4px;">
              <span style="color:var(--text-muted)">Artifact:</span> ${artifactsHtml}
            </div>
          </div>
        `;
        container.appendChild(card);
      });
    } catch (e) {
      console.error("Failed to load CRISP-DM data", e);
    }
  },

  async loadTopItems() {
    const tableBody = document.getElementById("top-items-tbody");
    if (!tableBody) return;

    try {
      const res = await fetch("/api/v1/data/summary");
      if (!res.ok) return;
      const data = await res.json();

      tableBody.innerHTML = "";
      (data.top_items || []).slice(0, 10).forEach((item, idx) => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td style="font-family:var(--font-mono); color:var(--text-muted)">#${idx + 1}</td>
          <td style="font-weight:600; color:var(--text-primary)">${item.item}</td>
          <td style="font-family:var(--font-mono)">${item.count.toLocaleString()}</td>
          <td style="font-family:var(--font-mono); color:var(--accent-cyan)">${(item.support * 100).toFixed(1)}%</td>
          <td>
            <div style="background:rgba(255,255,255,0.08); border-radius:4px; height:8px; width:120px; overflow:hidden;">
              <div style="background:linear-gradient(90deg, var(--accent-cyan), var(--accent-blue)); height:100%; width:${Math.min(100, item.support * 250)}%"></div>
            </div>
          </td>
        `;
        tableBody.appendChild(tr);
      });
    } catch (e) {
      console.error("Failed to load top items", e);
    }
  },

  setupSyntheticGenerator() {
    const btn = document.getElementById("btn-generate-synthetic");
    if (!btn) return;
    btn.addEventListener("click", async () => {
      const txCount = parseInt(document.getElementById("synth-tx-count").value) || 800;
      const avgBasket = parseFloat(document.getElementById("synth-avg-basket").value) || 4.0;
      const noise = parseFloat(document.getElementById("synth-noise").value) || 0.20;

      btn.disabled = true;
      btn.textContent = "Generating...";
      try {
        const res = await fetch("/api/v1/data/generate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            num_transactions: txCount,
            avg_basket_size: avgBasket,
            noise_level: noise,
            seed: Math.floor(Math.random() * 10000)
          })
        });
        if (!res.ok) throw new Error("Generation failed");
        App.toast(`Generated ${txCount} synthetic transactions with Zipf affinities!`);
        await App.refreshGlobalStats();
        await this.refresh();
      } catch (err) {
        App.toast(`Error: ${err.message}`, "error");
      } finally {
        btn.disabled = false;
        btn.textContent = "Generate Synthetic Transactions";
      }
    });
  }
};

window.CrispDmView = CrispDmView;
