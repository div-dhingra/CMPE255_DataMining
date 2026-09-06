"""Tier 2: Tokenizer, Sampling Strategies, and Chat Generation Engine Tests.

Verifies:
- Tokenizer byte-fallback, OOV freedom, ChatML formatting
- Greedy, Temperature, Top-K, Top-P sampling logic
- Repetition penalty logit dampening
- Streaming generation and live telemetry (TTFT, TPS)
- Multi-turn ChatSession memory preservation
"""

import pytest
import torch

from src.engine.chat_session import ChatSession
from src.engine.generator import GenerationConfig, StreamChunk, TextGenerator
from src.model.tokenizer import Tokenizer


class TestTokenizer:
    def test_special_tokens(self, tokenizer):
        assert tokenizer.pad_id == 0
        assert tokenizer.unk_id == 1
        assert tokenizer.bos_id == 2
        assert tokenizer.eos_id == 3
        assert tokenizer.im_start_id == 4
        assert tokenizer.im_end_id == 5

    def test_encode_decode_roundtrip(self, tokenizer):
        text = "Transformer architecture with attention mechanisms and RoPE embeddings."
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        assert decoded == text

    def test_byte_fallback_zero_oov(self, tokenizer):
        # Even rare Unicode symbols or emojis decompose cleanly via byte fallback
        unicode_text = "Testing 🚀 unicode emojis and special chars: ©, ®, ∑, π."
        encoded = tokenizer.encode(unicode_text)
        assert len(encoded) > 0
        decoded = tokenizer.decode(encoded)
        assert decoded == unicode_text

    def test_chatml_formatting(self, tokenizer):
        messages = [
            {"role": "system", "content": "You are an AI assistant."},
            {"role": "user", "content": "Hello!"},
        ]
        formatted = tokenizer.format_chat_prompt(messages, add_generation_prompt=True)
        assert "<|im_start|>system\nYou are an AI assistant.<|im_end|>" in formatted
        assert "<|im_start|>user\nHello!<|im_end|>" in formatted
        assert formatted.endswith("<|im_start|>assistant\n")


class TestGenerationEngine:
    def test_greedy_generation_determinism(self, generator):
        prompt = "Hello"
        cfg = GenerationConfig(max_new_tokens=8, temperature=0.0)
        res1 = generator.generate(prompt, cfg)
        res2 = generator.generate(prompt, cfg)
        assert res1["token_ids"] == res2["token_ids"]
        assert res1["text"] == res2["text"]

    def test_repetition_penalty(self):
        logits = torch.tensor([[10.0, 5.0, -2.0, 1.0]])
        # Penalize token 0 (which is positive) and token 2 (which is negative)
        penalized = TextGenerator.apply_repetition_penalty(
            logits, generated_tokens=[0, 2], penalty=2.0
        )
        # Positive logit 10.0 / 2.0 = 5.0
        assert penalized[0, 0].item() == 5.0
        # Negative logit -2.0 * 2.0 = -4.0 (made even more negative)
        assert penalized[0, 2].item() == -4.0
        # Unpenalized tokens remain untouched
        assert penalized[0, 1].item() == 5.0
        assert penalized[0, 3].item() == 1.0

    def test_temperature_and_topk_sampling(self):
        logits = torch.randn(1, 100)
        # Sample with high top-k and temperature
        next_tok = TextGenerator.sample_next_token(
            logits, temperature=0.8, top_k=10, top_p=0.95
        )
        assert next_tok.shape == (1, 1)
        assert 0 <= next_tok.item() < 100

    def test_streaming_generation_telemetry(self, generator):
        chunks = list(
            generator.generate_stream(
                "Transformer",
                GenerationConfig(max_new_tokens=6, temperature=0.7),
            )
        )
        assert len(chunks) > 0
        last_chunk = chunks[-1]
        assert last_chunk.step == len(chunks)
        assert last_chunk.ttft_ms > 0.0
        assert last_chunk.tokens_per_sec > 0.0
        assert last_chunk.kv_memory_mb > 0.0


class TestChatSession:
    def test_multi_turn_history(self, chat_session):
        chat_session.reset()
        res1 = chat_session.chat_turn("What is RoPE?")
        history = chat_session.get_history()
        assert len(history) >= 3  # system, user, assistant
        assert history[1]["role"] == "user"
        assert history[1]["content"] == "What is RoPE?"
        assert history[2]["role"] == "assistant"
        assert len(history[2]["content"]) > 0

    def test_session_reset(self, chat_session):
        chat_session.reset(system_prompt="Custom prompt")
        assert len(chat_session.messages) == 1
        assert chat_session.messages[0]["content"] == "Custom prompt"
