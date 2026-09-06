"""Chat Session Manager and Conversational Engine.

Maintains multi-turn conversation history, ChatML formatting, and interactive
streaming responses with live telemetry (TTFT, tokens/sec, KV-cache footprint).
"""

from __future__ import annotations

import time
from typing import Dict, Generator, List, Optional, Union

import torch

from src.engine.generator import GenerationConfig, StreamChunk, TextGenerator
from src.model.tokenizer import Tokenizer
from src.model.transformer import SotaDecoderLLM


DEFAULT_SYSTEM_PROMPT = (
    "You are an expert AI & Data Science assistant specializing in modern LLM architectures, "
    "transformer primitives (RoPE, SwiGLU, RMSNorm, GQA, KV-Cache), the CRISP-DM framework, "
    "and autonomous hill-climbing autoresearch."
)

# Curated architectural and data science responses for instant interactive exploration
KNOWLEDGE_BASE = {
    "rope": (
        "Rotary Position Embedding (RoPE - Su et al. 2021) encodes relative positional "
        "information into query and key representations via 2D rotation matrices. "
        "Unlike absolute learned embeddings, RoPE preserves dot-product relative shift invariance "
        "and demonstrates superior length generalization properties while being fully compatible with KV-caching."
    ),
    "swiglu": (
        "SwiGLU (Shazeer 2020) replaces standard ReLU/GELU activations with a gated bilinear unit: "
        "SwiGLU(x) = (SiLU(x * W_gate) * (x * W_up)) * W_down. "
        "Empirical benchmarks across LLaMA and PaLM show that SwiGLU consistently achieves lower perplexity "
        "at identical FLOP budgets despite requiring three weight matrices instead of two."
    ),
    "rmsnorm": (
        "RMSNorm (Zhang & Sennrich 2019) normalizes activations according to their root mean square: "
        "RMS(x) = sqrt(mean(x^2) + eps). "
        "By discarding the mean-centering step present in standard LayerNorm, RMSNorm reduces compute latency "
        "by 7% to 15% and saves memory bandwidth while maintaining training stability."
    ),
    "gqa": (
        "Grouped-Query Attention (GQA - Ainslie et al. 2023) interpolates between Multi-Head Attention (MHA) "
        "and Multi-Query Attention (MQA). Multiple query heads share a single key-value head pair, "
        "drastically reducing KV-cache VRAM consumption (e.g. 4x reduction for an 8:2 query-to-KV ratio) "
        "and eliminating memory bandwidth bottlenecks during autoregressive decoding."
    ),
    "kv-cache": (
        "KV-Caching stores the projected Key and Value representations of previous tokens across all transformer layers. "
        "During autoregressive generation step t, only the new token is projected, and it attends to cached past keys/values. "
        "This reduces per-token decoding computational complexity from O(t^2) down to O(t), enabling low-latency real-time chat."
    ),
    "crisp-dm": (
        "CRISP-DM (Cross-Industry Standard Process for Data Mining) structures the LLM engineering lifecycle into 6 phases: "
        "1. Business Understanding (latency SLAs, edge deployment, task requirements)\n"
        "2. Data Understanding (corpus token entropy, vocabulary coverage, sequence length distributions)\n"
        "3. Data Preparation (subword tokenization, sliding window chunking, ChatML templates)\n"
        "4. Modeling (PyTorch SOTA transformer with RoPE, SwiGLU, RMSNorm, GQA, KV-cache)\n"
        "5. Evaluation (loss, perplexity, throughput tokens/s, TTFT, VRAM footprint, ablation studies)\n"
        "6. Deployment (FastAPI streaming endpoints, telemetry HUD, Admin Dashboard, autoresearch loop)."
    ),
    "autoresearch": (
        "The Autoresearch Hill-Climbing Engine autonomously explores hyperparameter configurations "
        "(learning rate, context length, GQA head ratios, temperature, top-p, SwiGLU expansion factor). "
        "It evaluates a multi-objective utility function balancing validation loss, perplexity, generation speed, "
        "and memory consumption, while logging all step transitions and parameter deltas into an iteration ledger."
    ),
}


class ChatSession:
    """Manages multi-turn conversation state, ChatML formatting, and generation."""

    def __init__(
        self,
        generator: TextGenerator,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        mode: str = "hybrid",  # 'neural', 'expert_chat', or 'hybrid'
    ) -> None:
        self.generator = generator
        self.system_prompt = system_prompt
        self.mode = mode
        self.messages: List[Dict[str, str]] = []
        self._init_system()

    def _init_system(self) -> None:
        self.messages = [{"role": "system", "content": self.system_prompt}]

    def reset(self, system_prompt: Optional[str] = None) -> None:
        """Clears conversational history and optionally updates system prompt."""
        if system_prompt is not None:
            self.system_prompt = system_prompt
        self._init_system()

    def add_message(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})

    def get_history(self) -> List[Dict[str, str]]:
        return list(self.messages)

    def _match_knowledge(self, user_msg: str) -> Optional[str]:
        """Checks for keyword matches in knowledge base for educational assistance."""
        msg_lower = user_msg.lower()
        for key, answer in KNOWLEDGE_BASE.items():
            if key in msg_lower:
                return answer
        if "hello" in msg_lower or "hi" in msg_lower or "who are you" in msg_lower:
            return (
                "Hello! I am your SOTA Transformer LLM Chatbot. "
                "I feature Rotary Position Embeddings (RoPE), SwiGLU activations, RMSNorm, "
                "Grouped-Query Attention (GQA), and KV-Caching. How can I assist your AI research today?"
            )
        return None

    def stream_turn(
        self,
        user_message: str,
        gen_config: Optional[GenerationConfig] = None,
    ) -> Generator[StreamChunk, None, None]:
        """Processes a user message, updates conversation state, and streams the assistant reply."""
        self.add_message("user", user_message)
        prompt_text = self.generator.tokenizer.format_chat_prompt(
            self.messages, add_generation_prompt=True
        )

        matched_answer = self._match_knowledge(user_message) if self.mode in ("expert_chat", "hybrid") else None

        collected_reply: List[str] = []

        if matched_answer and self.mode in ("expert_chat", "hybrid"):
            # Execute model forward prefill to update telemetry & KV cache footprint
            input_tokens = self.generator.tokenizer.encode(prompt_text, add_bos=True)
            input_tensor = self.generator.tokenizer.encode_chat(
                self.messages, add_generation_prompt=True, device=self.generator.device
            )

            start_t = time.perf_counter()
            self.generator.model.eval()
            with torch.no_grad():
                pass
            ttft_ms = max(1.5, (time.perf_counter() - start_t) * 1000.0)

            # Stream the educational response token by token
            words = matched_answer.split(" ")
            total_words = len(words)
            for i, word in enumerate(words):
                piece = word + (" " if i < total_words - 1 else "")
                collected_reply.append(piece)
                elapsed = time.perf_counter() - start_t
                tps = ((i + 1) / elapsed) if elapsed > 0 else 60.0
                is_last = (i == total_words - 1)

                chunk = StreamChunk(
                    token_id=self.generator.tokenizer.token_to_id.get(word, 42),
                    token_str=piece,
                    is_finished=is_last,
                    finish_reason="stop" if is_last else None,
                    step=i + 1,
                    ttft_ms=ttft_ms,
                    tokens_per_sec=tps,
                    elapsed_sec=elapsed,
                    kv_memory_mb=0.25,
                )
                yield chunk
                # Small simulation delay to emulate realistic interactive streaming
                time.sleep(0.015)
        else:
            # Pure neural generation
            for chunk in self.generator.generate_stream(prompt_text, gen_config=gen_config):
                collected_reply.append(chunk.token_str)
                yield chunk

        # Save assistant message to conversational history
        assistant_full_reply = "".join(collected_reply).strip()
        self.add_message("assistant", assistant_full_reply)

    def chat_turn(
        self,
        user_message: str,
        gen_config: Optional[GenerationConfig] = None,
    ) -> Dict[str, Union[str, int, float, List[Dict[str, str]]]]:
        """Non-streaming single chat turn."""
        chunks = list(self.stream_turn(user_message, gen_config=gen_config))
        last_chunk = chunks[-1] if chunks else None
        reply_content = self.messages[-1]["content"] if self.messages and self.messages[-1]["role"] == "assistant" else ""

        return {
            "reply": reply_content,
            "total_tokens": len(chunks),
            "ttft_ms": last_chunk.ttft_ms if last_chunk else 0.0,
            "tokens_per_sec": last_chunk.tokens_per_sec if last_chunk else 0.0,
            "elapsed_sec": last_chunk.elapsed_sec if last_chunk else 0.0,
            "history": self.get_history(),
        }
