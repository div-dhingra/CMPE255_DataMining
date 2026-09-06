/**
 * SOTA LLM Chatbot & Autoresearch Studio Front-End Engine.
 * Handles live SSE streaming, Chart.js trajectory visualization,
 * KV-cache calculations, and REST interactions.
 */

// Global State
let currentTab = "chat";
let chatMessages = [];
let trajectoryChart = null;
let totalTokensGenerated = 0;

// Initialize when DOM is loaded
document.addEventListener("DOMContentLoaded", () => {
  if (window.lucide) {
    window.lucide.createIcons();
  }
  initTelemetry();
  initArchitectureExplorer();
  initAutoresearchChart();
  fetchAutoresearchStatus();
  fetchBenchmarkMatrix();
  fetchCrispDmSummary();
  runKVSimulation();
});

// =====================================================================
// Tab Navigation
// =====================================================================
function switchTab(tabId) {
  currentTab = tabId;
  document.querySelectorAll(".tab-panel").forEach((el) => el.classList.add("hidden"));
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.classList.remove("bg-blue-600", "text-white");
    btn.classList.add("text-slate-400");
  });

  const activePanel = document.getElementById(`tab-content-${tabId}`);
  if (activePanel) activePanel.classList.remove("hidden");

  const activeBtn = document.getElementById(`nav-${tabId}`);
  if (activeBtn) {
    activeBtn.classList.add("bg-blue-600", "text-white");
    activeBtn.classList.remove("text-slate-400");
  }

  if (window.lucide) {
    window.lucide.createIcons();
  }

  if (tabId === "autoresearch" && trajectoryChart) {
    setTimeout(() => trajectoryChart.resize(), 100);
  }
}

function updateParamVal(param, val) {
  const el = document.getElementById(`val-${param}`);
  if (el) el.textContent = val;
}

// =====================================================================
// Telemetry & Hardware Info
// =====================================================================
async function initTelemetry() {
  try {
    const res = await fetch("/api/model/telemetry");
    if (!res.ok) return;
    const data = await res.json();
    const devEl = document.getElementById("header-device");
    if (devEl) {
      devEl.textContent = `${data.device} · ${data.model_memory_mb} MB`;
    }
  } catch (err) {
    console.error("Telemetry error:", err);
  }
}

// =====================================================================
// Chatbot Playground & Live SSE Streaming
// =====================================================================
function sendPromptPreset(promptText) {
  const input = document.getElementById("chat-input");
  if (input) {
    input.value = promptText;
    handleChatSubmit(new Event("submit"));
  }
}

function appendUserMessage(text) {
  const container = document.getElementById("chat-messages");
  const msgEl = document.createElement("div");
  msgEl.className = "flex items-start justify-end space-x-3";
  msgEl.innerHTML = `
    <div class="bg-blue-600 rounded-2xl rounded-tr-none p-4 max-w-[85%] text-sm text-white shadow-md leading-relaxed">
      ${escapeHtml(text)}
    </div>
    <div class="w-8 h-8 rounded-lg bg-slate-800 border border-slate-700 flex items-center justify-center shrink-0">
      <i data-lucide="user" class="w-4 h-4 text-slate-300"></i>
    </div>
  `;
  container.appendChild(msgEl);
  container.scrollTop = container.scrollHeight;
  if (window.lucide) window.lucide.createIcons();
}

function createAssistantMessagePlaceholder() {
  const container = document.getElementById("chat-messages");
  const msgEl = document.createElement("div");
  msgEl.className = "flex items-start space-x-3";
  const id = `assistant-msg-${Date.now()}`;
  msgEl.innerHTML = `
    <div class="w-8 h-8 rounded-lg bg-blue-600/20 border border-blue-500/30 flex items-center justify-center shrink-0">
      <i data-lucide="cpu" class="w-4 h-4 text-blue-400"></i>
    </div>
    <div id="${id}" class="bg-slate-900 border border-slate-800 rounded-2xl rounded-tl-none p-4 max-w-[85%] text-sm text-slate-200 shadow-sm leading-relaxed whitespace-pre-wrap">
      <span class="inline-block animate-pulse text-slate-500">Generating with RoPE & SwiGLU...</span>
    </div>
  `;
  container.appendChild(msgEl);
  container.scrollTop = container.scrollHeight;
  if (window.lucide) window.lucide.createIcons();
  return document.getElementById(id);
}

async function handleChatSubmit(e) {
  e.preventDefault();
  const input = document.getElementById("chat-input");
  const text = input.value.trim();
  if (!text) return;

  input.value = "";
  appendUserMessage(text);
  chatMessages.push({ role: "user", content: text });

  const msgContentEl = createAssistantMessagePlaceholder();
  let firstTokenReceived = false;

  const temp = parseFloat(document.getElementById("input-temp").value);
  const topp = parseFloat(document.getElementById("input-topp").value);
  const topk = parseInt(document.getElementById("input-topk").value);
  const reppen = parseFloat(document.getElementById("input-reppen").value);
  const maxtok = parseInt(document.getElementById("input-maxtok").value);

  try {
    const response = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        messages: chatMessages,
        temperature: temp,
        top_p: topp,
        top_k: topk,
        repetition_penalty: reppen,
        max_tokens: maxtok,
        stream: true,
      }),
    });

    if (!response.ok) {
      msgContentEl.innerHTML = `<span class="text-red-400">Error generating response (${response.statusText})</span>`;
      return;
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let accumulatedText = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const rawChunk = decoder.decode(value, { stream: true });
      const lines = rawChunk.split("\n");

      for (const line of lines) {
        if (line.startsWith("data: ")) {
          const dataStr = line.replace("data: ", "").trim();
          if (dataStr === "[DONE]") continue;

          try {
            const data = JSON.parse(dataStr);
            if (!firstTokenReceived) {
              msgContentEl.innerHTML = "";
              firstTokenReceived = true;
            }

            accumulatedText += data.token;
            msgContentEl.textContent = accumulatedText;

            // Update live telemetry HUD
            if (data.ttft_ms) document.getElementById("hud-ttft").textContent = `${data.ttft_ms} ms`;
            if (data.tokens_per_sec) document.getElementById("hud-tps").textContent = `${data.tokens_per_sec} tps`;
            if (data.kv_memory_mb) document.getElementById("hud-kv-mem").textContent = `${data.kv_memory_mb} MB`;

            totalTokensGenerated += 1;
            document.getElementById("hud-total-tokens").textContent = totalTokensGenerated;

            const chatBox = document.getElementById("chat-messages");
            chatBox.scrollTop = chatBox.scrollHeight;
          } catch (pErr) {
            // Ignore partial SSE chunk parse error
          }
        }
      }
    }

    chatMessages.push({ role: "assistant", content: accumulatedText });
  } catch (err) {
    console.error("Streaming error:", err);
    msgContentEl.innerHTML = `<span class="text-red-400">Connection error: ${escapeHtml(err.message)}</span>`;
  }
}

async function clearChat() {
  await fetch("/api/chat/reset", { method: "POST" });
  chatMessages = [];
  const container = document.getElementById("chat-messages");
  container.innerHTML = `
    <div class="flex items-start space-x-3">
      <div class="w-8 h-8 rounded-lg bg-blue-600/20 border border-blue-500/30 flex items-center justify-center shrink-0">
        <i data-lucide="cpu" class="w-4 h-4 text-blue-400"></i>
      </div>
      <div class="bg-slate-900 border border-slate-800 rounded-2xl rounded-tl-none p-4 max-w-[85%] text-sm text-slate-200 shadow-sm leading-relaxed">
        <p class="font-medium text-blue-400 mb-1">Session Cleared</p>
        The conversation history and KV-cache buffers have been reset. How can I assist you now?
      </div>
    </div>
  `;
  if (window.lucide) window.lucide.createIcons();
}

// =====================================================================
// Architecture Explorer & KV-Cache Calculator
// =====================================================================
async function initArchitectureExplorer() {
  try {
    const res = await fetch("/api/model/architecture");
    if (!res.ok) return;
    const data = await res.json();

    const tbody = document.getElementById("layer-table-body");
    if (!tbody) return;

    tbody.innerHTML = data.layer_breakdown
      .map(
        (layer) => `
        <tr class="hover:bg-slate-800/40 transition">
          <td class="px-4 py-2.5 font-semibold text-white">${escapeHtml(layer.name)}</td>
          <td class="px-4 py-2.5 text-blue-400">${escapeHtml(layer.type)}</td>
          <td class="px-4 py-2.5 text-purple-400 font-mono">${(layer.params || 0).toLocaleString()}</td>
          <td class="px-4 py-2.5 text-slate-400">${escapeHtml(layer.details || layer.attention_heads || "")}</td>
        </tr>
      `
      )
      .join("");
  } catch (err) {
    console.error("Architecture fetch error:", err);
  }
}

async function runKVSimulation() {
  const batch = parseInt(document.getElementById("kv-batch-input").value);
  const seq = parseInt(document.getElementById("kv-seq-input").value);
  const dim = parseInt(document.getElementById("kv-dim-input").value);

  document.getElementById("kv-batch-val").textContent = batch;
  document.getElementById("kv-seq-val").textContent = seq;
  document.getElementById("kv-dim-val").textContent = dim;

  try {
    const res = await fetch("/api/model/kv-cache-calculator", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        batch_size: batch,
        seq_len: seq,
        dim: dim,
        n_layers: 8,
        n_heads: 8,
        precision_bytes: 2,
      }),
    });
    if (!res.ok) return;
    const data = await res.json();

    document.getElementById("kv-res-mha").textContent = `${data.footprints_mb["MHA (Full Heads)"]} MB`;
    document.getElementById("kv-res-gqa4").textContent = `${data.footprints_mb["GQA-4 (Half KV Heads)"]} MB`;
    document.getElementById("kv-res-gqa2").textContent = `${data.footprints_mb["GQA-2 (Quarter KV Heads)"]} MB`;
    document.getElementById("kv-res-mqa").textContent = `${data.footprints_mb["MQA (Single KV Head)"]} MB`;

    document.getElementById("kv-savings-summary").textContent =
      `Calculated savings: ${data.savings.GQA_vs_MHA_reduction} (${data.savings.MQA_vs_MHA_reduction} for MQA)`;
  } catch (err) {
    console.error("KV simulation error:", err);
  }
}

async function fetchAttentionMatrix() {
  const prompt = document.getElementById("attn-prompt-input").value;
  const container = document.getElementById("attn-matrix-container");
  container.innerHTML = `<p class="text-xs text-blue-400 py-4 text-center">Computing forward pass attention weights...</p>`;

  try {
    const res = await fetch("/api/model/attention-map", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt: prompt, layer_idx: 0 }),
    });
    if (!res.ok) return;
    const data = await res.json();

    const tokens = data.tokens;
    const matrix = data.matrix;
    const n = tokens.length;

    let html = `
      <div class="text-xs text-slate-400 mb-2">Layer 0, Attention Head 0 (${n} tokens)</div>
      <div class="grid gap-1" style="grid-template-columns: repeat(${n + 1}, minmax(0, 1fr));">
        <div class="text-[10px] text-slate-500 font-mono text-center">Q \\ K</div>
    `;

    // Header row with token labels
    for (let k = 0; k < n; k++) {
      html += `<div class="text-[9px] text-slate-400 font-mono truncate text-center" title="${escapeHtml(tokens[k])}">${escapeHtml(tokens[k])}</div>`;
    }

    // Rows
    for (let i = 0; i < n; i++) {
      html += `<div class="text-[9px] text-slate-400 font-mono truncate text-right pr-1" title="${escapeHtml(tokens[i])}">${escapeHtml(tokens[i])}</div>`;
      for (let j = 0; j < n; j++) {
        const val = matrix[i] && matrix[i][j] !== undefined ? matrix[i][j] : 0;
        const opacity = Math.min(1.0, val * 3.5);
        const bg = `rgba(59, 130, 246, ${opacity})`;
        const textColor = opacity > 0.5 ? "text-white" : "text-slate-400";
        html += `
          <div class="attention-cell rounded ${textColor} cursor-pointer" style="background-color: ${bg};" title="Q: '${escapeHtml(tokens[i])}' attends to K: '${escapeHtml(tokens[j])}' (Weight: ${val.toFixed(3)})">
            ${val > 0.05 ? val.toFixed(2) : ""}
          </div>
        `;
      }
    }

    html += `</div>`;
    container.innerHTML = html;
  } catch (err) {
    container.innerHTML = `<p class="text-xs text-red-400 py-4 text-center">Error: ${escapeHtml(err.message)}</p>`;
  }
}

// =====================================================================
// Autoresearch Studio & Trajectory Chart
// =====================================================================
function initAutoresearchChart() {
  const ctx = document.getElementById("trajectoryChart");
  if (!ctx) return;

  trajectoryChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: [],
      datasets: [
        {
          label: "Composite Utility Score",
          data: [],
          borderColor: "#3b82f6",
          backgroundColor: "rgba(59, 130, 246, 0.1)",
          borderWidth: 2.5,
          tension: 0.25,
          fill: true,
          yAxisID: "yScore",
        },
        {
          label: "Throughput (Tokens / Sec)",
          data: [],
          borderColor: "#10b981",
          borderWidth: 2,
          borderDash: [5, 5],
          tension: 0.2,
          fill: false,
          yAxisID: "yTps",
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      scales: {
        x: {
          grid: { color: "rgba(51, 65, 85, 0.4)" },
          ticks: { color: "#94a3b8", font: { size: 10 } },
        },
        yScore: {
          type: "linear",
          display: true,
          position: "left",
          grid: { color: "rgba(51, 65, 85, 0.4)" },
          ticks: { color: "#3b82f6", font: { size: 10 } },
          title: { display: true, text: "Utility Score", color: "#3b82f6", font: { size: 11 } },
        },
        yTps: {
          type: "linear",
          display: true,
          position: "right",
          grid: { drawOnChartArea: false },
          ticks: { color: "#10b981", font: { size: 10 } },
          title: { display: true, text: "Throughput (TPS)", color: "#10b981", font: { size: 11 } },
        },
      },
      plugins: {
        legend: { labels: { color: "#cbd5e1", font: { size: 11 } } },
      },
    },
  });
}

async function fetchAutoresearchStatus() {
  try {
    const res = await fetch("/api/autoresearch/ledger");
    if (!res.ok) return;
    const data = await res.json();
    renderAutoresearchData(data);
  } catch (err) {
    console.error("Autoresearch ledger fetch error:", err);
  }
}

function renderAutoresearchData(data) {
  const summary = data.summary;
  const history = data.history || [];

  // Update KPI cards
  document.getElementById("stat-total-steps").textContent = summary.total_steps || 0;
  document.getElementById("stat-accepted-steps").textContent = summary.accepted_steps || 0;
  document.getElementById("stat-accept-rate").textContent = `${Math.round((summary.acceptance_rate || 0) * 100)}%`;
  document.getElementById("stat-baseline-score").textContent = summary.baseline_score ? summary.baseline_score.toFixed(3) : "--";
  document.getElementById("stat-best-score").textContent = summary.best_score ? summary.best_score.toFixed(3) : "--";
  document.getElementById("stat-improvement-pct").textContent = `+${summary.score_improvement_pct || 0}%`;

  document.getElementById("chart-step-counter").textContent = `${history.length} iterations recorded`;

  // Update Trajectory Chart
  if (trajectoryChart) {
    trajectoryChart.data.labels = history.map((e) => `Iter ${e.iteration}`);
    trajectoryChart.data.datasets[0].data = history.map((e) => e.composite_score);
    trajectoryChart.data.datasets[1].data = history.map((e) => e.tokens_per_sec);
    trajectoryChart.update();
  }

  // Update Ledger Table
  const tbody = document.getElementById("ledger-table-body");
  if (tbody && history.length > 0) {
    tbody.innerHTML = history
      .slice()
      .reverse()
      .map((entry) => {
        let badgeClass = "badge-accepted";
        if (entry.decision === "rejected") badgeClass = "badge-rejected";
        else if (entry.decision === "baseline") badgeClass = "badge-baseline";
        else if (entry.step_type === "restart") badgeClass = "badge-restart";

        const deltaColor = entry.delta_score > 0 ? "text-emerald-400" : (entry.delta_score < 0 ? "text-red-400" : "text-slate-400");
        const deltaPrefix = entry.delta_score > 0 ? "+" : "";

        return `
          <tr class="hover:bg-slate-800/40 transition">
            <td class="px-3 py-2 font-bold text-white">#${entry.iteration}</td>
            <td class="px-3 py-2 text-slate-400">${escapeHtml(entry.step_type)}</td>
            <td class="px-3 py-2"><span class="px-2 py-0.5 rounded text-[10px] uppercase font-semibold ${badgeClass}">${escapeHtml(entry.decision)}</span></td>
            <td class="px-3 py-2 font-bold text-blue-400">${entry.composite_score.toFixed(3)}</td>
            <td class="px-3 py-2 ${deltaColor}">${deltaPrefix}${entry.delta_score.toFixed(3)}</td>
            <td class="px-3 py-2 text-emerald-400">${entry.tokens_per_sec.toFixed(1)} tps</td>
            <td class="px-3 py-2 text-purple-400">${entry.memory_mb.toFixed(1)} MB</td>
          </tr>
        `;
      })
      .join("");
  }

  // Update Latest Proposal Delta Inspector
  const lastEntry = history.length > 0 ? history[history.length - 1] : null;
  const inspector = document.getElementById("delta-inspector-body");
  const litTag = document.getElementById("delta-literature-tag");

  if (lastEntry && inspector) {
    let deltaHtml = `
      <div class="flex justify-between py-1 border-b border-slate-800">
        <span class="text-slate-400">Step Transition:</span>
        <span class="text-white font-bold">#${lastEntry.iteration} (${lastEntry.step_type})</span>
      </div>
      <div class="flex justify-between py-1 border-b border-slate-800">
        <span class="text-slate-400">Decision Status:</span>
        <span class="font-bold ${lastEntry.decision === 'accepted' ? 'text-emerald-400' : 'text-red-400'}">${lastEntry.decision.toUpperCase()}</span>
      </div>
      <div class="flex justify-between py-1 border-b border-slate-800">
        <span class="text-slate-400">Score Delta (Δ):</span>
        <span class="font-bold ${lastEntry.delta_score > 0 ? 'text-emerald-400' : 'text-red-400'}">${lastEntry.delta_score > 0 ? '+' : ''}${lastEntry.delta_score.toFixed(4)}</span>
      </div>
    `;

    const deltas = lastEntry.parameter_deltas || {};
    if (Object.keys(deltas).length > 0) {
      deltaHtml += `<div class="pt-2 text-slate-300 font-semibold">Mutated Parameters:</div>`;
      for (const [k, v] of Object.entries(deltas)) {
        if (typeof v === "object" && v !== null && "from" in v && "to" in v) {
          deltaHtml += `
            <div class="flex justify-between text-[11px] py-0.5">
              <span class="text-slate-400">${k}:</span>
              <span class="text-indigo-300">${v.from} &rarr; <strong class="text-emerald-400">${v.to}</strong></span>
            </div>
          `;
        } else {
          deltaHtml += `
            <div class="flex justify-between text-[11px] py-0.5">
              <span class="text-slate-400">${k}:</span>
              <span class="text-emerald-400">${JSON.stringify(v)}</span>
            </div>
          `;
        }
      }
    }

    inspector.innerHTML = deltaHtml;
    if (litTag) litTag.textContent = lastEntry.literature_reference || "Standard Literature";
  }
}

async function executeHillClimbStep() {
  try {
    const res = await fetch("/api/autoresearch/step", { method: "POST" });
    if (!res.ok) return;
    await fetchAutoresearchStatus();
  } catch (err) {
    console.error("Step execution error:", err);
  }
}

async function runMultipleSteps(numSteps) {
  try {
    const res = await fetch("/api/autoresearch/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ steps: numSteps }),
    });
    if (!res.ok) return;
    await fetchAutoresearchStatus();
  } catch (err) {
    console.error("Multi-step error:", err);
  }
}

async function resetAutoresearch() {
  if (confirm("Reset the hill-climbing search ledger to initial baseline?")) {
    await fetch("/api/autoresearch/reset", { method: "POST" });
    await fetchAutoresearchStatus();
  }
}

// =====================================================================
// Research Benchmarks & Literature Alignment
// =====================================================================
async function fetchBenchmarkMatrix() {
  try {
    const res = await fetch("/api/benchmarks/matrix");
    if (!res.ok) return;
    const data = await res.json();

    // Render Literature Papers Grid
    const papersGrid = document.getElementById("papers-grid");
    if (papersGrid && data.papers) {
      papersGrid.innerHTML = Object.values(data.papers)
        .map(
          (p) => `
          <div class="glass-card p-4 rounded-xl flex flex-col justify-between border-t-2 border-indigo-500">
            <div>
              <div class="flex items-center justify-between text-[10px] text-indigo-400 font-mono">
                <span>${escapeHtml(p.venue)}</span>
                <span>${p.year}</span>
              </div>
              <h4 class="text-sm font-bold text-white mt-1">${escapeHtml(p.title)}</h4>
              <p class="text-[11px] text-slate-400 mt-1">${escapeHtml(p.authors)}</p>
              <div class="text-xs text-slate-300 mt-2 bg-slate-900/60 p-2.5 rounded-lg border border-slate-800">
                <strong>Key Contribution:</strong> ${escapeHtml(p.key_contribution)}
              </div>
            </div>
            <div class="mt-3 pt-2 border-t border-slate-800 text-[11px] text-blue-400 font-mono">
              ${escapeHtml(p.doi_or_arxiv)}
            </div>
          </div>
        `
        )
        .join("");
    }

    // Render Benchmark Matrix Table
    const benchBody = document.getElementById("benchmark-matrix-body");
    if (benchBody && data.benchmark_matrix) {
      benchBody.innerHTML = data.benchmark_matrix
        .map((row) => {
          const isOurs = row.architecture.includes("Ours");
          const rowClass = isOurs ? "bg-blue-950/30 font-bold border-l-2 border-blue-500" : "hover:bg-slate-800/40";
          return `
            <tr class="${rowClass} transition">
              <td class="px-4 py-3 text-white">${escapeHtml(row.architecture)}</td>
              <td class="px-4 py-3 text-emerald-400">${escapeHtml(row.norm_type)}</td>
              <td class="px-4 py-3 text-blue-400">${escapeHtml(row.positional_emb)}</td>
              <td class="px-4 py-3 text-purple-400">${escapeHtml(row.activation)}</td>
              <td class="px-4 py-3 text-amber-400">${escapeHtml(row.attention_type)}</td>
              <td class="px-4 py-3 font-mono">${row.kv_cache_mb_1k.toFixed(1)} MB</td>
              <td class="px-4 py-3 font-mono text-emerald-400">${row.throughput_tps.toFixed(1)} tps</td>
              <td class="px-4 py-3 font-mono text-indigo-300">${row.relative_ppl.toFixed(3)}</td>
            </tr>
          `;
        })
        .join("");
    }

    // Render Ablation Studies
    const ablationsContainer = document.getElementById("ablations-container");
    if (ablationsContainer && data.ablations) {
      ablationsContainer.innerHTML = data.ablations
        .map(
          (ab) => `
          <div class="glass-card p-4 rounded-xl">
            <h4 class="text-xs font-bold text-white uppercase tracking-wider mb-2 flex items-center space-x-1.5">
              <i data-lucide="layers" class="w-3.5 h-3.5 text-blue-400"></i>
              <span>${escapeHtml(ab.ablation_target)}</span>
            </h4>
            <div class="space-y-2 text-xs">
              ${ab.variants
                .map(
                  (v) => `
                <div class="flex items-center justify-between p-2 rounded bg-slate-900/60 border border-slate-800/80">
                  <div>
                    <div class="font-semibold text-slate-200">${escapeHtml(v.name)}</div>
                    <div class="text-[10px] text-slate-400">${escapeHtml(v.note)}</div>
                  </div>
                  <div class="text-right font-mono">
                    <span class="text-blue-400 font-bold">${v.perplexity ? `PPL: ${v.perplexity}` : v.kv_memory_mb}</span>
                    <div class="text-[10px] text-emerald-400">${v.speedup || `${v.decode_tps} tps`}</div>
                  </div>
                </div>
              `
                )
                .join("")}
            </div>
          </div>
        `
        )
        .join("");
      if (window.lucide) window.lucide.createIcons();
    }
  } catch (err) {
    console.error("Benchmark fetch error:", err);
  }
}

// =====================================================================
// CRISP-DM Summary
// =====================================================================
async function fetchCrispDmSummary() {
  try {
    const res = await fetch("/api/crisp-dm/summary");
    if (!res.ok) return;
    const data = await res.json();

    const container = document.getElementById("crisp-phases-container");
    if (!container || !data.phases) return;

    container.innerHTML = data.phases
      .map(
        (phase) => `
        <div class="glass-card p-4 rounded-xl border-l-4 border-blue-500 hover:border-indigo-400 transition">
          <div class="flex items-center justify-between">
            <h3 class="text-sm font-bold text-white flex items-center space-x-2">
              <span class="w-6 h-6 rounded-full bg-blue-600/30 text-blue-400 border border-blue-500/30 flex items-center justify-center text-xs font-mono font-bold">${phase.phase_id}</span>
              <span>Phase ${phase.phase_id}: ${escapeHtml(phase.name)}</span>
            </h3>
            <span class="text-[10px] px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">Verified</span>
          </div>
          <p class="text-xs text-slate-300 mt-2 leading-relaxed">${escapeHtml(phase.focus)}</p>
          <div class="mt-3 flex flex-wrap gap-1.5">
            ${phase.deliverables
              .map((d) => `<span class="px-2 py-0.5 rounded bg-slate-900 text-slate-400 text-[10px] border border-slate-800 font-mono">${escapeHtml(d)}</span>`)
              .join("")}
          </div>
        </div>
      `
      )
      .join("");
  } catch (err) {
    console.error("CRISP-DM summary error:", err);
  }
}

// =====================================================================
// Utilities
// =====================================================================
function escapeHtml(str) {
  if (typeof str !== "string") return str;
  return str.replace(/[&<>'"]/g, (tag) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" }[tag] || tag));
}
