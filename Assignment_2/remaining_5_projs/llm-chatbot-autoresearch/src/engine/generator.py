"""Autoregressive Generation Engine with SOTA Sampling Algorithms.

Supports:
- Temperature scaling
- Top-k filtering
- Top-p (nucleus) sampling (Holtzman et al. 2019)
- Repetition penalty (Keskar et al. 2019)
- KV-Cached autoregressive decoding with O(1) step complexity
- Live streaming token generator yielding TTFT (Time To First Token), tokens/sec, and memory telemetry
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import AsyncGenerator, Dict, Generator, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from src.model.attention import KVCache
from src.model.tokenizer import Tokenizer
from src.model.transformer import SotaDecoderLLM


@dataclass
class GenerationConfig:
    """Hyperparameters for text generation."""

    max_new_tokens: int = 128
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.15
    stop_tokens: Optional[List[int]] = None
    seed: Optional[int] = None


@dataclass
class StreamChunk:
    """Data yielded per generated token during streaming."""

    token_id: int
    token_str: str
    is_finished: bool
    finish_reason: Optional[str] = None
    step: int = 0
    ttft_ms: float = 0.0
    tokens_per_sec: float = 0.0
    elapsed_sec: float = 0.0
    kv_memory_mb: float = 0.0


class TextGenerator:
    """High-performance generation engine utilizing SotaDecoderLLM and KV-Cache."""

    def __init__(
        self,
        model: SotaDecoderLLM,
        tokenizer: Tokenizer,
        device: Optional[str] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or str(next(model.parameters()).device)
        self.stop_token_ids = {
            tokenizer.eos_id,
            tokenizer.im_end_id,
        }

    @staticmethod
    def apply_repetition_penalty(
        logits: torch.Tensor,
        generated_tokens: List[int],
        penalty: float,
    ) -> torch.Tensor:
        """Applies multiplicative penalty to logits of already generated tokens."""
        if penalty == 1.0 or not generated_tokens:
            return logits

        # Work on a clone to avoid mutating input tensor
        logits_out = logits.clone()
        for token_id in set(generated_tokens):
            if logits_out[0, token_id] > 0:
                logits_out[0, token_id] /= penalty
            else:
                logits_out[0, token_id] *= penalty
        return logits_out

    @staticmethod
    def sample_next_token(
        logits: torch.Tensor,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """Samples the next token from logits using Temperature, Top-K, and Top-P filtering."""
        # Greedy decoding for near-zero temperature
        if temperature < 1e-4:
            return torch.argmax(logits, dim=-1, keepdim=True)

        # 1. Temperature scaling
        scaled_logits = logits / temperature

        # 2. Top-K filtering
        if top_k > 0:
            top_k = min(top_k, scaled_logits.size(-1))
            k_th_values, _ = torch.topk(scaled_logits, top_k, dim=-1)
            min_k_val = k_th_values[:, -1].unsqueeze(-1)
            scaled_logits = torch.where(
                scaled_logits < min_k_val,
                torch.full_like(scaled_logits, float("-inf")),
                scaled_logits,
            )

        # 3. Top-P (nucleus) filtering
        if 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Mask tokens where cumulative probability exceeds threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift right by 1 so we keep at least the first token above top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False

            # Scatter back to original indices
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            scaled_logits = scaled_logits.masked_fill(indices_to_remove, float("-inf"))

        # 4. Softmax and multinomial sampling
        probs = F.softmax(scaled_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        return next_token

    def generate_stream(
        self,
        prompt: Union[str, List[int], torch.Tensor],
        gen_config: Optional[GenerationConfig] = None,
    ) -> Generator[StreamChunk, None, None]:
        """Streams generated tokens one by one with live latency and memory telemetry.

        Yields:
            StreamChunk containing the decoded token piece and performance HUD metrics.
        """
        config = gen_config or GenerationConfig()
        if config.seed is not None:
            torch.manual_seed(config.seed)

        # 1. Prepare initial input tokens
        if isinstance(prompt, str):
            input_tokens = self.tokenizer.encode(prompt, add_bos=True)
            input_tensor = torch.tensor([input_tokens], dtype=torch.long, device=self.device)
        elif isinstance(prompt, list):
            input_tokens = list(prompt)
            input_tensor = torch.tensor([input_tokens], dtype=torch.long, device=self.device)
        else:
            input_tensor = prompt.to(self.device)
            input_tokens = prompt[0].tolist()

        stop_ids = set(self.stop_token_ids)
        if config.stop_tokens:
            stop_ids.update(config.stop_tokens)

        # 2. Instantiate dedicated KV-cache for this generation run
        prompt_len = input_tensor.shape[1]
        max_total_len = prompt_len + config.max_new_tokens
        kv_cache = self.model.create_kv_cache(
            max_batch_size=1,
            max_seq_len=max_total_len,
            device=self.device,
        )

        start_time = time.perf_counter()
        ttft_ms = 0.0

        # 3. Prefill Phase: process entire prompt in one forward pass
        self.model.eval()
        with torch.no_grad():
            logits, _ = self.model(
                input_tensor,
                start_pos=0,
                kv_cache=kv_cache,
            )

        # First token latency (Time To First Token)
        ttft_ms = (time.perf_counter() - start_time) * 1000.0

        # Sample first token from last position of prefill logits
        last_logits = logits[:, -1, :]
        last_logits = self.apply_repetition_penalty(
            last_logits, input_tokens, config.repetition_penalty
        )
        next_token = self.sample_next_token(
            last_logits,
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p,
        )

        generated_ids: List[int] = []
        cur_pos = prompt_len

        # 4. Decode Phase: autoregressive generation loop
        for step in range(config.max_new_tokens):
            token_id = next_token.item()
            generated_ids.append(token_id)
            token_str = self.tokenizer.decode([token_id], skip_special_tokens=False)

            is_stop = token_id in stop_ids or step == config.max_new_tokens - 1
            finish_reason = "stop" if token_id in stop_ids else ("length" if is_stop else None)

            elapsed = time.perf_counter() - start_time
            tps = (len(generated_ids) / elapsed) if elapsed > 0 else 0.0

            yield StreamChunk(
                token_id=token_id,
                token_str=token_str,
                is_finished=is_stop,
                finish_reason=finish_reason,
                step=step + 1,
                ttft_ms=ttft_ms,
                tokens_per_sec=tps,
                elapsed_sec=elapsed,
                kv_memory_mb=kv_cache.get_memory_mb(),
            )

            if is_stop:
                break

            # Forward single token with KV cache at cur_pos
            with torch.no_grad():
                step_logits, _ = self.model(
                    next_token,
                    start_pos=cur_pos,
                    kv_cache=kv_cache,
                )

            cur_pos += 1
            step_logits_last = step_logits[:, -1, :]
            step_logits_penalized = self.apply_repetition_penalty(
                step_logits_last, input_tokens + generated_ids, config.repetition_penalty
            )
            next_token = self.sample_next_token(
                step_logits_penalized,
                temperature=config.temperature,
                top_k=config.top_k,
                top_p=config.top_p,
            )

    def generate(
        self,
        prompt: Union[str, List[int], torch.Tensor],
        gen_config: Optional[GenerationConfig] = None,
    ) -> Dict[str, Union[str, int, float, List[int]]]:
        """Non-streaming text generation completing up to max_new_tokens."""
        full_tokens: List[int] = []
        token_strs: List[str] = []
        last_chunk: Optional[StreamChunk] = None

        for chunk in self.generate_stream(prompt, gen_config=gen_config):
            full_tokens.append(chunk.token_id)
            token_strs.append(chunk.token_str)
            last_chunk = chunk

        # Clean text
        text_output = self.tokenizer.decode(full_tokens, skip_special_tokens=True)

        return {
            "text": text_output,
            "token_ids": full_tokens,
            "total_tokens": len(full_tokens),
            "ttft_ms": last_chunk.ttft_ms if last_chunk else 0.0,
            "tokens_per_sec": last_chunk.tokens_per_sec if last_chunk else 0.0,
            "elapsed_sec": last_chunk.elapsed_sec if last_chunk else 0.0,
            "finish_reason": last_chunk.finish_reason if last_chunk else "length",
            "kv_memory_mb": last_chunk.kv_memory_mb if last_chunk else 0.0,
        }
