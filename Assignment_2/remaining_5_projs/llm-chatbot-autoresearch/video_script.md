# End-to-End Video Demo Script: SOTA LLM Chatbot & Autoresearch Engine

**Target Duration:** 2 - 3 Minutes
**Pacing:** Concise and visual, explicitly linking backend source code architecture to frontend dashboard capabilities.

---

### [0:00 - 0:15] Introduction
**Visual:** Show the "Playground" tab of the AI Engineer Admin Dashboard.
**Audio (Voiceover):** "Welcome to the SOTA LLM Studio demo. Today, I'll walk you through a PyTorch decoder-only transformer built entirely from primitives, alongside an autonomous autoresearch optimization engine. We'll explore three core engine components in the code and see exactly how they manifest in the UI."

---

### [0:15 - 0:55] Feature 1: Grouped-Query Attention & KV-Cache
**Visual (Code):** Open `src/model/attention.py`, highlighting the `KVCache` class.
**Audio:** "First, let's look at `src/model/attention.py`. Here, we implement Grouped-Query Attention (GQA) and a dynamic `KVCache`. The code pre-allocates tensor buffers for keys and values across layers. This architectural choice radically reduces inference complexity from O(N²) to O(N) by preventing identical token re-evaluations during autoregressive decoding."

**Visual (UI):** Switch to the "KV-Cache & Model" Tab in the Dashboard. Play with the VRAM simulator sliders.
**Audio:** "In the dashboard under the 'KV-Cache & Model' tab, this logic powers the live VRAM Memory Simulator. You can visually see how swapping from standard Multi-Head Attention to our GQA implementation vastly reduces memory bandwidth as context windows increase."

---

### [0:55 - 1:40] Feature 2: Autoregressive Streaming & Sampling
**Visual (Code):** Open `src/engine/generator.py`, highlighting the `TextGenerator` class and `sample_next_token()` method.
**Audio:** "Next, in `src/engine/generator.py`, we manage text generation. The `TextGenerator` class handles temperature scaling, Top-K, and Nucleus Top-p sampling. More importantly, it yields Server-Sent Events (SSE) asynchronously, emitting rich chunk metadata alongside each token, such as 'Time-To-First-Token' and 'Tokens Per Second'."

**Visual (UI):** Switch to the "Playground" Tab. Type a prompt and hit generate. Highlight the streaming text and the live telemetry HUD.
**Audio:** "When we jump to the 'Playground' tab and submit a prompt, you can see this SSE stream in action. The UI renders the response token-by-token in real-time. Notice the live telemetry HUD at the bottom—it perfectly mirrors the generation metrics calculated by the backend generator in real-time."

---

### [1:40 - 2:20] Feature 3: Autoresearch Optimization Engine
**Visual (Code):** Open `src/autoresearch/hill_climber.py`, highlighting the `AutoresearchHillClimber` class and `evaluate_state()`.
**Audio:** "Finally, let's explore hyperparameter optimization in `src/autoresearch/hill_climber.py`. The `AutoresearchHillClimber` autonomously mutates model states—like learning rates or attention heads. The `evaluate_state()` method scores these configurations using a multi-objective function, balancing validation loss against hardware throughput, rejecting or accepting steps based on simulated annealing."

**Visual (UI):** Switch to the "Autoresearch Studio" Tab. Show the Chart.js graph updating and the Ledger Table.
**Audio:** "Over in the 'Autoresearch Studio' tab, this hill-climbing process is mapped out visually. The live Chart.js trajectory graph plots the utility score of each iteration loop. Below it, the Experiment Ledger table logs every accepted or rejected parameter jump, giving AI engineers complete observability into the autonomous exploration process."

---

### [2:20 - 2:30] Conclusion
**Visual:** Briefly flash the "Literature Matrix" and "CRISP-DM Flow" tabs.
**Audio:** "With built-in CRISP-DM documentation and empirical literature benchmarks, this dashboard brings SOTA AI engineering principles directly to the edge. Thanks for watching!"
