// Market Basket Recommendation Playground Controller
const CartSimulatorView = {
  cart: [],
  catalog: [],

  init() {
    this.setupControls();
  },

  setupControls() {
    const clearBtn = document.getElementById("btn-clear-cart");
    if (clearBtn) {
      clearBtn.addEventListener("click", () => {
        this.cart = [];
        this.updateCartUI();
        this.fetchRecommendations();
      });
    }

    const itemSelect = document.getElementById("catalog-item-select");
    const addBtn = document.getElementById("btn-add-to-cart");

    if (addBtn && itemSelect) {
      addBtn.addEventListener("click", () => {
        const selected = itemSelect.value;
        if (selected && !this.cart.includes(selected)) {
          this.cart.push(selected);
          this.updateCartUI();
          this.fetchRecommendations();
        }
      });
    }
  },

  async render() {
    await this.refreshCatalog();
    await this.fetchRecommendations();
  },

  async refreshCatalog() {
    try {
      const res = await fetch("/api/v1/cart/catalog?limit=40");
      if (!res.ok) return;
      const data = await res.json();
      this.catalog = data.items || [];

      const select = document.getElementById("catalog-item-select");
      if (!select) return;

      select.innerHTML = `<option value="">-- Select product to add to cart --</option>`;
      this.catalog.forEach(item => {
        const opt = document.createElement("option");
        opt.value = item.name;
        opt.textContent = `${item.name} (${item.frequency_pct}% freq)`;
        select.appendChild(opt);
      });
    } catch (e) {
      console.error("Failed to load catalog", e);
    }
  },

  updateCartUI() {
    const container = document.getElementById("active-cart-items");
    const countBadge = document.getElementById("cart-count-badge");
    if (!container) return;

    if (countBadge) countBadge.textContent = `${this.cart.length} items`;

    if (this.cart.length === 0) {
      container.innerHTML = `<div style="color:var(--text-muted); font-size:0.85rem; padding:12px 0;">Your basket is empty. Select products above to simulate basket associations.</div>`;
      return;
    }

    container.innerHTML = "";
    this.cart.forEach(item => {
      const tag = document.createElement("div");
      tag.className = "glass-panel";
      tag.style.display = "inline-flex";
      tag.style.alignItems = "center";
      tag.style.gap = "8px";
      tag.style.padding = "6px 12px";
      tag.style.borderRadius = "9999px";
      tag.style.fontSize = "0.825rem";
      tag.style.border = "1px solid rgba(56, 189, 248, 0.3)";

      tag.innerHTML = `
        <span style="color:#f8fafc; font-weight:500;">${item}</span>
        <button style="background:transparent; border:none; color:var(--accent-rose); cursor:pointer; font-weight:bold; font-size:0.9rem;" title="Remove">&times;</button>
      `;

      tag.querySelector("button").addEventListener("click", () => {
        this.cart = this.cart.filter(i => i !== item);
        this.updateCartUI();
        this.fetchRecommendations();
      });

      container.appendChild(tag);
    });
  },

  async fetchRecommendations() {
    const grid = document.getElementById("recommendations-grid");
    const upliftDisplay = document.getElementById("projected-uplift-pct");
    const matchedRulesDisplay = document.getElementById("matched-rules-count");
    if (!grid) return;

    try {
      const res = await fetch("/api/v1/cart/recommend", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          cart_items: this.cart,
          top_k: 6,
          min_lift: 1.05
        })
      });

      if (!res.ok) return;
      const data = await res.json();

      if (upliftDisplay) upliftDisplay.textContent = `+${data.projected_cross_sell_uplift_pct}%`;
      if (matchedRulesDisplay) matchedRulesDisplay.textContent = `${data.matched_rules_count} rules`;

      grid.innerHTML = "";
      if (data.recommendations.length === 0) {
        grid.innerHTML = `<div style="grid-column:1/-1; padding:30px; text-align:center; color:var(--text-muted)">No matching association rules found for this combination. Try adding items with high catalog frequency.</div>`;
        return;
      }

      data.recommendations.forEach(rec => {
        const card = document.createElement("div");
        card.className = "rec-card";

        card.innerHTML = `
          <div style="display:flex; justify-content:space-between; align-items:flex-start;">
            <div class="rec-title">${rec.item}</div>
            <button class="btn btn-emerald btn-add-rec" style="padding:4px 10px; font-size:0.75rem;">+ Add</button>
          </div>
          <div class="rec-badges">
            <span class="badge badge-lift">Lift: ${rec.lift.toFixed(2)}x</span>
            <span class="badge badge-conf">Conf: ${(rec.confidence * 100).toFixed(1)}%</span>
            <span class="badge badge-kulc">Kulc: ${rec.kulczynski.toFixed(3)}</span>
          </div>
          <p class="rec-explanation">${rec.explanation}</p>
          <div style="font-size:0.7rem; color:var(--text-muted); font-family:var(--font-mono); margin-top:auto; border-top:1px solid rgba(255,255,255,0.05); padding-top:6px;">
            Rule: ${rec.trigger_rule}
          </div>
        `;

        card.querySelector(".btn-add-rec").addEventListener("click", () => {
          if (!this.cart.includes(rec.item)) {
            this.cart.push(rec.item);
            this.updateCartUI();
            this.fetchRecommendations();
            App.toast(`Added ${rec.item} to cart!`);
          }
        });

        grid.appendChild(card);
      });
    } catch (e) {
      console.error("Failed to fetch recommendations", e);
    }
  }
};

window.CartSimulatorView = CartSimulatorView;
