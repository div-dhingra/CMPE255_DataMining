# Retroactive Implementation Plan: SOTA LLM Chatbot & Autoresearch Engine

## 1. Goal Description
The objective of this project was to implement a high-performance, edge-optimized decoder-only Transformer Language Model from pure PyTorch primitives, accompanied by an autonomous "Autoresearch" hyperparameter optimization engine and a web-based AI engineering dashboard. 

## 2. Technical Stack
- **Core ML Framework:** PyTorch 2.0+ (Tensors, Autograd, `torch.nn`)
- **Backend API:** FastAPI (Uvicorn, SSE Streaming, Pydantic)
- **Frontend Dashboard:** HTML5, Tailwind CSS, Chart.js, Server-Sent Events (SSE)
- **Testing:** Pytest (Unit, Integration, E2E)
- **Architecture Philosophy:** CRISP-DM lifecycle tracking and pure-primitive architectural alignment with SOTA literature (e.g., LLaMA, Chinchilla).

## 3. Built Architecture Overview

### 3.1 Model Architecture (`src/model/`)
The model is a decoder-only autoregressive transformer built from SOTA primitives:
- **Rotary Position Embedding (RoPE):** Complex rotational operators for relative positioning without learning position embeddings (`primitives.py`).
- **SwiGLU Activations:** Replaces standard GELU/ReLU in the feed-forward network to improve representation, expanding hidden dimension sizes dynamically (`primitives.py`).
- **RMSNorm:** Replaces LayerNorm for pre-normalization without mean-centering, yielding faster inference (`primitives.py`).
- **Grouped-Query Attention (GQA):** Optimization over Multi-Head Attention, reducing KV cache memory footprint by sharing Key/Value heads across multiple Query heads (`attention.py`).
- **Low-Latency KV-Cache:** Pre-allocated tensor buffers supporting O(N) decoding generation (`attention.py`).
- **Custom Tokenizer:** ChatML prompt templating support with BPE/Subword alignment (`tokenizer.py`).

### 3.2 Engine (`src/engine/`)
- **Generator:** Handles streaming token generation, sampling algorithms (Top-K, Nucleus Top-p, Temperature scaling), and repetition penalties (`generator.py`).
- **Chat Session:** Maintains stateful conversational history and prompt templates (`chat_session.py`).

### 3.3 Autoresearch Optimizer (`src/autoresearch/`)
- **Hill Climber:** An autonomous loop exploring hyperparameter spaces (learning rates, context windows, sampling temps, etc.) using multi-objective metrics like validation loss, generation throughput, and hardware footprint (`hill_climber.py`).
- **Ledger:** Tracks and records parameter deltas (jumps) and objective metrics for each generation trial (`ledger.py`).
- **Literature Alignment:** Benchmarks results explicitly against foundational research findings (`literature.py`).

### 3.4 API & Dashboard (`src/api/` & `src/dashboard/`)
- **FastAPI Endpoints:** Streams completion events, handles memory telemetry calculations, and controls autoresearch loops (`app.py`, `routes_*.py`).
- **AI Engineer Admin Dashboard:** Vanilla JS & Tailwind UI interacting with API endpoints to provide live generation HUDs, KV-cache simulations, and Autoresearch Chart.js visualizations (`index.html`).

## 4. Testing & Verification
The system enforces validation through a 4-tier Pytest suite (`run_tests.sh`):
1. **Tier 1 (Primitives):** SOTA components (RoPE, SwiGLU, RMSNorm, GQA, KV-cache tensor allocations).
2. **Tier 2 (Generation):** Tokenizer encoding/decoding, generation loops, and probability sampling mechanisms.
3. **Tier 3 (Autoresearch):** Verification of objective evaluations, ledger JSON/CSV exports, and correct plateau escape mechanisms.
4. **Tier 4 (API & E2E):** FastAPI client routing, SSE streaming responses, and endpoint status codes.
