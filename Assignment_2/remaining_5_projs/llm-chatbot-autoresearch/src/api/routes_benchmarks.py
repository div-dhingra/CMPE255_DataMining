"""FastAPI Router for Research Benchmarks, Ablations, and CRISP-DM Summary."""

from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter

from src.autoresearch.literature import (
    ABLATION_STUDIES,
    BENCHMARK_MATRIX,
    LANDMARK_PAPERS,
    get_literature_summary,
)

router = APIRouter(prefix="/api/benchmarks", tags=["Research Benchmarks & CRISP-DM"])


@router.get("/matrix")
async def get_benchmark_matrix() -> Dict[str, Any]:
    """Returns literature comparison matrix and landmark paper metadata."""
    return get_literature_summary()


@router.get("/ablations")
async def get_ablations() -> List[Dict[str, Any]]:
    """Returns empirical ablation studies across transformer primitives."""
    return ABLATION_STUDIES


@router.get("/crisp-dm/summary")
async def get_crisp_dm_summary() -> Dict[str, Any]:
    """Returns structured overview of the 6 CRISP-DM phases for the LLM project."""
    return {
        "framework": "CRISP-DM (Cross-Industry Standard Process for Data Mining)",
        "project": "SOTA LLM Chatbot & Autoresearch Engine",
        "phases": [
            {
                "phase_id": 1,
                "name": "Business Understanding",
                "focus": "Edge/laptop LLM deployment constraints, latency SLAs (<50ms TTFT, >150 TPS), and memory limits (<500MB VRAM).",
                "deliverables": ["SLA matrix", "Resource constraint model", "Hardware boundary definitions"],
            },
            {
                "phase_id": 2,
                "name": "Data Understanding",
                "focus": "Corpus token entropy (4.82 bits/tok), subword distribution, ChatML conversation turn formats.",
                "deliverables": ["Entropy profiles", "Sequence length histograms", "Special token taxonomy"],
            },
            {
                "phase_id": 3,
                "name": "Data Preparation",
                "focus": "Byte-level subword tokenization (100% OOV-free), dynamic batching, causal attention mask generation.",
                "deliverables": ["Tokenizer encoder/decoder", "Causal mask generator", "Sliding-window chunker"],
            },
            {
                "phase_id": 4,
                "name": "Modeling",
                "focus": "Pure PyTorch SOTA transformer with RoPE, SwiGLU, RMSNorm, GQA, and KV-cache autoregressive decoding.",
                "deliverables": ["SotaDecoderLLM architecture", "KVCache buffer manager", "TextGenerator sampling engine"],
            },
            {
                "phase_id": 5,
                "name": "Evaluation",
                "focus": "Loss, perplexity, generation throughput (tokens/sec), KV-cache VRAM scaling, and landmark paper benchmark matrix.",
                "deliverables": ["Multi-objective utility evaluator", "Literature comparison matrix", "Primitive ablation studies"],
            },
            {
                "phase_id": 6,
                "name": "Deployment",
                "focus": "FastAPI asynchronous streaming service (SSE), AI Engineer Admin Dashboard, and autonomous hill-climbing autoresearch loop.",
                "deliverables": ["REST/SSE endpoints", "Admin Dashboard SPA", "Experiment ledger export (JSON/CSV)"],
            },
        ],
    }
