"""SOTA Transformer Primitives.

Implements:
1. RMSNorm (Root Mean Square Layer Normalization) - Zhang & Sennrich 2019 / LLaMA
2. Rotary Position Embedding (RoPE) - Su et al. 2021
3. SwiGLU Feed-Forward Network - Shazeer 2020
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Normalizes activations by their root mean square without subtracting the mean,
    reducing computational overhead and memory bandwidth compared to standard LayerNorm.
    Formula: y = (x / RMS(x)) * weight, where RMS(x) = sqrt(mean(x^2) + eps)
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

    def extra_repr(self) -> str:
        return f"dim={self.weight.shape[0]}, eps={self.eps}"


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) - Su et al. 2021.

    Encodes absolute position with a rotation matrix that naturally incorporates
    relative position dependencies through inner products in attention.
    Supports arbitrary sequence lengths and KV-cache start position offsets.
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        base: float = 10000.0,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute frequencies: theta_i = base^(-2*(i-1)/dim) for i in [1, dim/2]
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute cos and sin cached tables
        self._build_cache(max_seq_len, device=device)

    def _build_cache(self, seq_len: int, device: torch.device | str = "cpu") -> None:
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device))  # [seq_len, dim/2]
        # Repeat frequencies across paired dimensions: [cos(f0), cos(f1), ..., cos(f0), cos(f1)...]
        emb = torch.cat((freqs, freqs), dim=-1)  # [seq_len, dim]
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotates half the hidden dimensions: [-x2, x1]."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seq_len: int,
        start_pos: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Applies Rotary Position Embedding to Query and Key tensors.

        Args:
            q: Query tensor of shape [batch, n_heads, seq_len, head_dim]
            k: Key tensor of shape [batch, n_kv_heads, seq_len, head_dim]
            seq_len: Number of tokens being transformed in this step
            start_pos: Position offset (used during autoregressive KV-cached decoding)

        Returns:
            Tuple of (q_rot, k_rot) with RoPE applied
        """
        end_pos = start_pos + seq_len
        if end_pos > self.cos_cached.shape[0]:
            self._build_cache(max(end_pos, self.cos_cached.shape[0] * 2), device=q.device)

        cos = self.cos_cached[start_pos:end_pos].to(q.device, dtype=q.dtype)  # [seq_len, head_dim]
        sin = self.sin_cached[start_pos:end_pos].to(q.device, dtype=q.dtype)  # [seq_len, head_dim]

        # Reshape cos and sin to broadcast with [batch, heads, seq_len, head_dim]
        cos = cos.unsqueeze(0).unsqueeze(1)  # [1, 1, seq_len, head_dim]
        sin = sin.unsqueeze(0).unsqueeze(1)  # [1, 1, seq_len, head_dim]

        q_rot = (q * cos) + (self.rotate_half(q) * sin)
        k_rot = (k * cos) + (self.rotate_half(k) * sin)
        return q_rot, k_rot


class SwiGLU(nn.Module):
    """SwiGLU (Swish Gated Linear Unit) - Shazeer 2020.

    Replaces standard ReLU or GELU feed-forward networks with a gated bilinear activation.
    Formula: SwiGLU(x) = (SiLU(x * W_gate) * (x * W_up)) * W_down
    Shown by Shazeer (2020) and Touvron et al. (2023) to substantially improve language modeling perplexity.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim

        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_up = nn.Linear(dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Gate path: SiLU(x * W_gate)
        gate = F.silu(self.w_gate(x))
        # Up path: x * W_up
        up = self.w_up(x)
        # Elementwise product and projection down
        return self.dropout(self.w_down(gate * up))

    def extra_repr(self) -> str:
        return f"in_dim={self.dim}, hidden_dim={self.hidden_dim}"
