// Central Dashboard Application Controller
const App = {
  activeDataset: "retail",
  currentTab: "crisp-dm",

  async init() {
    this.setupTabs();
    this.setupDatasetSelector();
    await this.refreshGlobalStats();

    // Initialize individual views
    if (window.CrispDmView) CrispDmView.init();
    if (window.NetworkGraphView) NetworkGraphView.init();
    if (window.ScatterMatrixView) ScatterMatrixView.init();
    if (window.AutoresearchView) AutoresearchView.init();
    if (window.CartSimulatorView) CartSimulatorView.init();
  },

  setupTabs() {
    const tabBtns = document.querySelectorAll(".tab-btn");
    tabBtns.forEach(btn => {
      btn.addEventListener("click", () => {
        tabBtns.forEach(b => b.classList.remove("active"));
        document.querySelectorAll(".view-panel").forEach(p => p.classList.remove("active"));

        btn.classList.add("active");
        const targetViewId = btn.getAttribute("data-view");
        const targetPanel = document.getElementById(targetViewId);
        if (targetPanel) {
          targetPanel.classList.add("active");
          this.currentTab = targetViewId;
          this.handleViewActivated(targetViewId);
        }
      });
    });
  },

  handleViewActivated(viewId) {
    if (viewId === "network-graph-view" && window.NetworkGraphView) {
      NetworkGraphView.render();
    } else if (viewId === "scatter-matrix-view" && window.ScatterMatrixView) {
      ScatterMatrixView.render();
    } else if (viewId === "autoresearch-view" && window.AutoresearchView) {
      AutoresearchView.render();
    } else if (viewId === "cart-simulator-view" && window.CartSimulatorView) {
      CartSimulatorView.render();
    }
  },

  setupDatasetSelector() {
    const sel = document.getElementById("dataset-select");
    if (!sel) return;
    sel.addEventListener("change", async (e) => {
      const val = e.target.value;
      this.activeDataset = val;
      App.toast(`Switching dataset to ${val}...`);
      try {
        const res = await fetch(`/api/v1/data/load-preset/${val}`, { method: "POST" });
        if (!res.ok) throw new Error("Failed to load preset");
        await this.refreshGlobalStats();
        App.toast(`Loaded ${val} dataset successfully!`);

        // Refresh all views
        if (window.CrispDmView) CrispDmView.refresh();
        if (window.NetworkGraphView) NetworkGraphView.render();
        if (window.ScatterMatrixView) ScatterMatrixView.render();
        if (window.CartSimulatorView) CartSimulatorView.refreshCatalog();
      } catch (err) {
        App.toast(`Error: ${err.message}`, "error");
      }
    });
  },

  async refreshGlobalStats() {
    try {
      const res = await fetch("/api/v1/data/summary");
      if (!res.ok) return;
      const data = await res.json();

      document.getElementById("stat-total-tx").textContent = data.num_transactions.toLocaleString();
      document.getElementById("stat-total-items").textContent = data.num_unique_items.toLocaleString();
      document.getElementById("stat-avg-basket").textContent = data.avg_basket_size;
      document.getElementById("stat-density").textContent = data.density_percent + "%";

      const rulesRes = await fetch("/api/v1/mine/rules?limit=1");
      if (rulesRes.ok) {
        const rulesData = await rulesRes.json();
        document.getElementById("stat-active-rules").textContent = rulesData.total_active_rules;
      }
    } catch (e) {
      console.warn("Global stats refresh error", e);
    }
  },

  toast(msg, type = "info") {
    const container = document.getElementById("toast-container");
    if (!container) return;
    const toast = document.createElement("div");
    toast.className = "toast";
    if (type === "error") toast.style.borderLeftColor = "var(--accent-rose)";
    toast.textContent = msg;
    container.appendChild(toast);
    setTimeout(() => {
      toast.style.opacity = "0";
      setTimeout(() => toast.remove(), 250);
    }, 3200);
  }
};

window.addEventListener("DOMContentLoaded", () => App.init());
