"""Multi-Objective Evaluation Engine for Autoresearch Hill-Climbing.

Evaluates:
1. Cross-Entropy Loss & Perplexity (LM Quality)
2. Generation Latency & Throughput (Tokens/Sec)
3. Memory Footprint (Model Weights + KV-Cache VRAM/RAM in MB)
4. Multi-Objective Normalized Utility Score
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from src.model.attention import KVCache
from src.model.config import ModelConfig
from src.model.tokenizer import Tokenizer
from src.model.transformer import SotaDecoderLLM


BENCHMARK_CORPUS = [
    "The transformer architecture relies entirely on self-attention mechanisms to compute representations.",
    "Rotary Position Embeddings incorporate explicit relative position information into query-key inner products.",
    "SwiGLU feed-forward layers improve language model perplexity compared to standard ReLU activations.",
    "Root Mean Square Layer Normalization normalizes activations without subtracting the mean, speeding up inference.",
    "Grouped-Query Attention shares key and value heads across multiple query heads to reduce memory bandwidth bottlenecks.",
    "Autoresearch algorithms use hill-climbing optimization to autonomously discover compute-optimal model hyperparameters.",
    "Chinchilla scaling laws state that for compute-optimal training, model size and dataset tokens should scale in equal proportion.",
    "Inference latency is dominated by memory bandwidth during autoregressive decoding because weights must be loaded every step.",
]


@dataclass
class EvaluationMetrics:
    """Metrics produced by the objective function."""

    loss: float
    perplexity: float
    tokens_per_sec: float
    memory_mb: float
    composite_score: float
    details: Dict[str, Union[float, int, str]]

    def to_dict(self) -> Dict[str, Union[float, int, str]]:
        res = {
            "loss": round(self.loss, 4),
            "perplexity": round(self.perplexity, 4),
            "tokens_per_sec": round(self.tokens_per_sec, 2),
            "memory_mb": round(self.memory_mb, 2),
            "composite_score": round(self.composite_score, 4),
        }
        res.update(self.details)
        return res


class EvaluationObjective:
    """Evaluates candidate hyperparameter sets on loss, throughput, and memory."""

    def __init__(
        self,
        tokenizer: Optional[Tokenizer] = None,
        corpus: Optional[List[str]] = None,
        device: str = "cpu",
        weight_ppl: float = 0.40,
        weight_tps: float = 0.40,
        weight_mem: float = 0.20,
    ) -> None:
        self.device = device
        self.tokenizer = tokenizer or Tokenizer(vocab_size=2048)
        self.corpus = corpus or BENCHMARK_CORPUS
        self.weight_ppl = weight_ppl
        self.weight_tps = weight_tps
        self.weight_mem = weight_mem

        # Pre-encode benchmark corpus
        self.encoded_corpus: List[torch.Tensor] = []
        for text in self.corpus:
            tokens = self.tokenizer.encode(text, add_bos=True, add_eos=True)
            if len(tokens) > 2:
                self.encoded_corpus.append(
                    torch.tensor([tokens], dtype=torch.long, device=self.device)
                )

    def evaluate_model(
        self,
        model: SotaDecoderLLM,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_eval_tokens: int = 32,
    ) -> EvaluationMetrics:
        """Evaluates an existing or newly initialized model instance."""
        model.eval()
        total_loss = 0.0
        total_batches = 0

        # 1. Evaluate cross-entropy loss across benchmark sequences
        with torch.no_grad():
            for seq in self.encoded_corpus:
                # Target is input shifted by 1
                input_ids = seq[:, :-1]
                targets = seq[:, 1:]
                logits, loss = model(input_ids, targets=targets)
                if loss is not None and not torch.isnan(loss):
                    total_loss += loss.item()
                    total_batches += 1

        avg_loss = (total_loss / max(1, total_batches)) if total_batches > 0 else 5.0
        # Guard against overflow in exp
        clipped_loss = min(avg_loss, 20.0)
        perplexity = math.exp(clipped_loss)

        # 2. Evaluate generation throughput (tokens/sec) with KV-cache
        test_prompt = self.encoded_corpus[0][:, :8]  # First 8 tokens as prompt
        prompt_len = test_prompt.shape[1]
        kv_cache = model.create_kv_cache(
            max_batch_size=1,
            max_seq_len=prompt_len + max_eval_tokens + 4,
            device=self.device,
        )

        t_start = time.perf_counter()
        with torch.no_grad():
            pref_logits, _ = model(test_prompt, start_pos=0, kv_cache=kv_cache)
            next_tok = torch.argmax(pref_logits[:, -1, :], dim=-1, keepdim=True)

            for step in range(max_eval_tokens):
                step_logits, _ = model(
                    next_tok,
                    start_pos=prompt_len + step,
                    kv_cache=kv_cache,
                )
                if temperature < 1e-4:
                    next_tok = torch.argmax(step_logits[:, -1, :], dim=-1, keepdim=True)
                else:
                    probs = F.softmax(step_logits[:, -1, :] / temperature, dim=-1)
                    next_tok = torch.multinomial(probs, num_samples=1)

        t_elapsed = max(time.perf_counter() - t_start, 1e-5)
        tokens_per_sec = max_eval_tokens / t_elapsed

        # 3. Memory calculation: model parameter memory + KV-cache memory in MB
        param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        kv_bytes = kv_cache.get_memory_bytes()
        total_mem_mb = (param_bytes + kv_bytes) / (1024.0 * 1024.0)

        # 4. Multi-Objective Composite Utility Score
        # Higher is strictly better.
        # Quality term: penalize high log(perplexity)
        quality_score = max(0.0, 10.0 - math.log(perplexity + 1e-5))
        # Speed term: reward high throughput log(1 + TPS)
        speed_score = math.log(1.0 + tokens_per_sec) * 2.0
        # Efficiency term: penalize large memory consumption
        mem_score = max(0.0, 10.0 - (total_mem_mb / 20.0))

        composite_score = (
            (self.weight_ppl * quality_score)
            + (self.weight_tps * speed_score)
            + (self.weight_mem * mem_score)
        )

        return EvaluationMetrics(
            loss=avg_loss,
            perplexity=perplexity,
            tokens_per_sec=tokens_per_sec,
            memory_mb=total_mem_mb,
            composite_score=composite_score,
            details={
                "n_layers": model.config.n_layers,
                "n_heads": model.config.n_heads,
                "n_kv_heads": model.config.n_kv_heads,
                "dim": model.config.dim,
                "temperature": temperature,
                "top_p": top_p,
            },
        )
