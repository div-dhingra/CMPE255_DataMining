// Association Rule Network Graph (Vis.js integration)
const NetworkGraphView = {
  network: null,
  nodesDataSet: null,
  edgesDataSet: null,
  rawGraphData: null,

  init() {
    this.setupControls();
  },

  setupControls() {
    const liftSlider = document.getElementById("graph-min-lift");
    const liftValDisplay = document.getElementById("graph-lift-val");
    if (liftSlider && liftValDisplay) {
      liftSlider.addEventListener("input", (e) => {
        liftValDisplay.textContent = parseFloat(e.target.value).toFixed(2);
      });
      liftSlider.addEventListener("change", () => this.render());
    }

    const maxRulesSelect = document.getElementById("graph-max-rules");
    if (maxRulesSelect) {
      maxRulesSelect.addEventListener("change", () => this.render());
    }

    const physicsToggle = document.getElementById("graph-toggle-physics");
    if (physicsToggle) {
      physicsToggle.addEventListener("change", (e) => {
        if (this.network) {
          this.network.setOptions({ physics: { enabled: e.target.checked } });
        }
      });
    }

    const searchInput = document.getElementById("graph-item-search");
    if (searchInput) {
      searchInput.addEventListener("input", (e) => {
        this.highlightItem(e.target.value.trim().toLowerCase());
      });
    }
  },

  async render() {
    const container = document.getElementById("vis-network-container");
    if (!container) return;

    const minLift = parseFloat(document.getElementById("graph-min-lift")?.value || 1.10);
    const maxRules = parseInt(document.getElementById("graph-max-rules")?.value || 35);

    try {
      const res = await fetch(`/api/v1/mine/graph?min_lift=${minLift}&max_rules=${maxRules}`);
      if (!res.ok) return;
      const data = await res.json();
      this.rawGraphData = data;

      document.getElementById("graph-stat-nodes").textContent = data.node_count;
      document.getElementById("graph-stat-edges").textContent = data.rule_count;
      document.getElementById("graph-stat-avg-lift").textContent = data.avg_lift + "x";

      // Transform nodes for Vis.js styling
      const visNodes = data.nodes.map(n => ({
        id: n.id,
        label: n.label,
        value: n.value,
        title: n.title,
        color: {
          background: "#1e293b",
          border: "#38bdf8",
          highlight: { background: "#0284c7", border: "#38bdf8" }
        },
        font: { color: "#f8fafc", size: 12, face: "system-ui" },
        shape: "dot"
      }));

      // Transform edges
      const visEdges = data.edges.map(e => ({
        from: e.from,
        to: e.to,
        value: e.value,
        title: e.title,
        label: e.label,
        font: { color: "#94a3b8", size: 9, strokeWidth: 0, align: "middle" },
        color: { color: "rgba(56, 189, 248, 0.45)", highlight: "#f43f5e" },
        arrows: { to: { enabled: true, scaleFactor: 0.7 } },
        smooth: { type: "curvedCW", roundness: 0.15 }
      }));

      if (typeof vis === "undefined") {
        container.innerHTML = `<div style="padding:40px; text-align:center; color:var(--text-muted)">Network visualization library loading...</div>`;
        return;
      }

      this.nodesDataSet = new vis.DataSet(visNodes);
      this.edgesDataSet = new vis.DataSet(visEdges);

      const options = {
        nodes: {
          scaling: { min: 14, max: 32 }
        },
        edges: {
          scaling: { min: 1.5, max: 7.0 }
        },
        physics: {
          enabled: true,
          solver: "forceAtlas2Based",
          forceAtlas2Based: {
            gravitationalConstant: -38,
            centralGravity: 0.015,
            springLength: 120,
            springConstant: 0.08,
            damping: 0.4
          },
          stabilization: { iterations: 150 }
        },
        interaction: {
          hover: true,
          tooltipDelay: 100,
          zoomView: true,
          dragView: true
        }
      };

      this.network = new vis.Network(container, { nodes: this.nodesDataSet, edges: this.edgesDataSet }, options);

      this.network.on("selectNode", (params) => {
        if (params.nodes.length > 0) {
          this.displayNodeInspector(params.nodes[0]);
        }
      });

      this.network.on("selectEdge", (params) => {
        if (params.edges.length > 0) {
          const edgeId = params.edges[0];
          const edgeObj = this.edgesDataSet.get(edgeId);
          this.displayEdgeInspector(edgeObj);
        }
      });
    } catch (e) {
      console.error("Failed to render network graph", e);
    }
  },

  highlightItem(term) {
    if (!this.network || !this.nodesDataSet) return;
    if (!term) {
      // Reset colors
      this.nodesDataSet.forEach(node => {
        this.nodesDataSet.update({ id: node.id, color: { background: "#1e293b", border: "#38bdf8" } });
      });
      return;
    }

    const matches = [];
    this.nodesDataSet.forEach(node => {
      if (node.id.toLowerCase().includes(term)) {
        matches.push(node.id);
        this.nodesDataSet.update({
          id: node.id,
          color: { background: "#f59e0b", border: "#f59e0b" }
        });
      } else {
        this.nodesDataSet.update({
          id: node.id,
          color: { background: "#0f172a", border: "#334155" }
        });
      }
    });

    if (matches.length > 0) {
      this.network.focus(matches[0], { scale: 1.2, animation: true });
      this.displayNodeInspector(matches[0]);
    }
  },

  displayNodeInspector(nodeId) {
    const inspector = document.getElementById("graph-inspector-panel");
    if (!inspector) return;

    const connectedEdges = this.network.getConnectedEdges(nodeId);
    let incomingCount = 0;
    let outgoingCount = 0;

    connectedEdges.forEach(eId => {
      const e = this.edgesDataSet.get(eId);
      if (e.from === nodeId) outgoingCount++;
      if (e.to === nodeId) incomingCount++;
    });

    inspector.innerHTML = `
      <div style="border-bottom:1px solid var(--bg-card-border); padding-bottom:10px; margin-bottom:12px;">
        <span style="font-size:0.75rem; color:var(--accent-cyan); text-transform:uppercase;">Selected Product Node</span>
        <h3 style="font-size:1.1rem; color:#fff; margin-top:4px;">${nodeId}</h3>
      </div>
      <div style="display:flex; gap:16px; margin-bottom:12px;">
        <div style="background:rgba(0,0,0,0.3); padding:8px 12px; border-radius:6px; flex:1;">
          <div style="font-size:0.7rem; color:var(--text-muted)">Rules as Antecedent</div>
          <div style="font-size:1.2rem; font-weight:700; color:#38bdf8;">${outgoingCount}</div>
        </div>
        <div style="background:rgba(0,0,0,0.3); padding:8px 12px; border-radius:6px; flex:1;">
          <div style="font-size:0.7rem; color:var(--text-muted)">Rules as Consequent</div>
          <div style="font-size:1.2rem; font-weight:700; color:#a78bfa;">${incomingCount}</div>
        </div>
      </div>
      <p style="font-size:0.8rem; color:var(--text-secondary); line-height:1.4;">
        This item participates in ${outgoingCount + incomingCount} strong retail affinity patterns. Click on connecting edges to inspect exact confidence, lift, and Kulczynski metrics.
      </p>
    `;
  },

  displayEdgeInspector(edge) {
    const inspector = document.getElementById("graph-inspector-panel");
    if (!inspector || !edge) return;

    inspector.innerHTML = `
      <div style="border-bottom:1px solid var(--bg-card-border); padding-bottom:10px; margin-bottom:12px;">
        <span style="font-size:0.75rem; color:var(--accent-emerald); text-transform:uppercase;">Association Rule Edge</span>
        <h4 style="font-size:1rem; color:#fff; margin-top:4px;">${edge.from} &rarr; ${edge.to}</h4>
      </div>
      <div style="font-size:0.85rem; color:var(--text-secondary); line-height:1.6;">
        ${edge.title}
      </div>
    `;
  }
};

window.NetworkGraphView = NetworkGraphView;
