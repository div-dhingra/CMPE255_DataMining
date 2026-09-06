"""Tier 1: SOTA Transformer Primitives & Architecture Unit Tests.

Verifies:
- RMSNorm invariance and scale
- Rotary Position Embedding (RoPE) relative properties
- SwiGLU non-linear gating
- Grouped-Query Attention (GQA) head grouping & KV-Cache equivalence
- Full SotaDecoderLLM forward passes and cross-entropy loss
"""

import math
import pytest
import torch

from src.model.attention import GroupedQueryAttention, KVCache, repeat_kv
from src.model.config import ModelConfig
from src.model.primitives import RMSNorm, RotaryEmbedding, SwiGLU
from src.model.transformer import SotaDecoderLLM, TransformerBlock


class TestRMSNorm:
    def test_rmsnorm_shape_and_dtype(self):
        norm = RMSNorm(dim=64)
        x = torch.randn(2, 10, 64)
        out = norm(x)
        assert out.shape == x.shape
        assert out.dtype == x.dtype

    def test_rmsnorm_scale_invariance(self):
        norm = RMSNorm(dim=32)
        x = torch.randn(1, 5, 32)
        # RMSNorm of scaled input: x * 10 has the same unit-normalized direction as x
        out1 = norm._norm(x)
        out2 = norm._norm(x * 10.0)
        assert torch.allclose(out1, out2, atol=1e-5)


class TestRotaryEmbedding:
    def test_rope_output_shape(self):
        head_dim = 32
        rope = RotaryEmbedding(dim=head_dim, max_seq_len=128)
        q = torch.randn(2, 4, 16, head_dim)  # [batch, heads, seq_len, head_dim]
        k = torch.randn(2, 2, 16, head_dim)  # [batch, kv_heads, seq_len, head_dim]

        q_rot, k_rot = rope(q, k, seq_len=16, start_pos=0)
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_rope_preserves_vector_magnitude(self):
        head_dim = 16
        rope = RotaryEmbedding(dim=head_dim, max_seq_len=64)
        q = torch.randn(1, 1, 8, head_dim)
        k = torch.randn(1, 1, 8, head_dim)

        q_rot, k_rot = rope(q, k, seq_len=8, start_pos=0)
        # Rotational matrices are orthogonal; norm must be preserved
        orig_norm = torch.norm(q, dim=-1)
        rot_norm = torch.norm(q_rot, dim=-1)
        assert torch.allclose(orig_norm, rot_norm, atol=1e-5)

    def test_rope_start_pos_offset(self):
        head_dim = 16
        rope = RotaryEmbedding(dim=head_dim, max_seq_len=64)
        # A single token at start_pos=5 should match position 5 of an 8-token sequence
        q_full = torch.randn(1, 1, 8, head_dim)
        k_full = torch.randn(1, 1, 8, head_dim)
        q_full_rot, _ = rope(q_full, k_full, seq_len=8, start_pos=0)

        q_single = q_full[:, :, 5:6, :]
        k_single = k_full[:, :, 5:6, :]
        q_single_rot, _ = rope(q_single, k_single, seq_len=1, start_pos=5)

        assert torch.allclose(q_full_rot[:, :, 5:6, :], q_single_rot, atol=1e-5)


class TestSwiGLU:
    def test_swiglu_dimensions(self):
        dim = 64
        hidden_dim = 128
        swiglu = SwiGLU(dim=dim, hidden_dim=hidden_dim)
        x = torch.randn(2, 10, dim)
        out = swiglu(x)
        assert out.shape == (2, 10, dim)

    def test_swiglu_non_linearity(self):
        dim = 32
        hidden_dim = 64
        swiglu = SwiGLU(dim=dim, hidden_dim=hidden_dim)
        x = torch.randn(1, 4, dim)
        # Should not be strictly linear: swiglu(2*x) != 2*swiglu(x)
        out1 = swiglu(x * 2.0)
        out2 = swiglu(x) * 2.0
        assert not torch.allclose(out1, out2, atol=1e-3)


class TestAttentionAndKVCache:
    def test_repeat_kv(self):
        x = torch.randn(2, 2, 8, 16)  # [batch, n_kv_heads=2, seq, head_dim]
        repeated = repeat_kv(x, n_rep=3)
        assert repeated.shape == (2, 6, 8, 16)  # 2 * 3 = 6 heads
        # Verify first 3 heads equal original head 0
        assert torch.allclose(repeated[:, 0, :, :], x[:, 0, :, :])
        assert torch.allclose(repeated[:, 1, :, :], x[:, 0, :, :])
        assert torch.allclose(repeated[:, 2, :, :], x[:, 0, :, :])
        assert torch.allclose(repeated[:, 3, :, :], x[:, 1, :, :])

    def test_gqa_forward_shapes(self, nano_config):
        attn = GroupedQueryAttention(config=nano_config, layer_idx=0)
        x = torch.randn(2, 12, nano_config.dim)
        out = attn(x, start_pos=0)
        assert out.shape == (2, 12, nano_config.dim)

    def test_kv_cache_autoregressive_equivalence(self, nano_config):
        """Validates that cached single-token decoding produces identical output to non-cached decoding."""
        torch.manual_seed(1337)
        model = SotaDecoderLLM(nano_config)
        model.eval()

        input_ids = torch.randint(0, nano_config.vocab_size, (1, 10))

        # 1. Forward pass without cache (full causal attention over 10 tokens)
        with torch.no_grad():
            full_logits, _ = model(input_ids)

        # 2. Forward pass with KV cache: Prefill first 6 tokens, then step tokens 6, 7, 8, 9
        kv_cache = model.create_kv_cache(max_batch_size=1, max_seq_len=20)
        prefill_ids = input_ids[:, :6]
        with torch.no_grad():
            pref_logits, _ = model(prefill_ids, start_pos=0, kv_cache=kv_cache)

        # Prefill logits for token 5 should match full forward pass at position 5
        assert torch.allclose(full_logits[:, 5, :], pref_logits[:, 5, :], atol=1e-4)

        # Step remaining tokens one by one
        for step in range(6, 10):
            next_id = input_ids[:, step : step + 1]
            with torch.no_grad():
                step_logits, _ = model(next_id, start_pos=step, kv_cache=kv_cache)
            # Step output must match full forward pass at this exact token position
            assert torch.allclose(full_logits[:, step : step + 1, :], step_logits, atol=1e-4)


class TestSotaDecoderLLM:
    def test_model_loss_computation(self, model, nano_config):
        input_ids = torch.randint(0, nano_config.vocab_size, (2, 16))
        targets = torch.randint(0, nano_config.vocab_size, (2, 16))
        logits, loss = model(input_ids, targets=targets)
        assert logits.shape == (2, 16, nano_config.vocab_size)
        assert loss is not None
        assert loss.item() > 0.0

    def test_parameter_count_consistency(self, nano_config):
        model = SotaDecoderLLM(nano_config)
        theo_params = nano_config.estimate_parameter_count()
        actual_params = model.get_num_params()
        assert theo_params == actual_params
