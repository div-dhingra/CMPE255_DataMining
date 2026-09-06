"""Grouped-Query Attention (GQA) and KV-Cache for Autoregressive Transformer.

Aligned with:
- Ainslie et al. 2023 ("GQA: Training Generalized Multi-Query Transformer Models")
- Vaswani et al. 2017 ("Attention Is All You Need")
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.config import ModelConfig
from src.model.primitives import RotaryEmbedding


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeats key/value heads across query groups for Grouped-Query Attention.

    Shape: [batch_size, n_kv_heads, seq_len, head_dim] -> [batch_size, n_heads, seq_len, head_dim]
    """
    if n_rep == 1:
        return x
    batch_size, n_kv_heads, seq_len, head_dim = x.shape
    return (
        x[:, :, None, :, :]
        .expand(batch_size, n_kv_heads, n_rep, seq_len, head_dim)
        .reshape(batch_size, n_kv_heads * n_rep, seq_len, head_dim)
    )


class KVCache:
    """Dynamic Key-Value Cache for Autoregressive Decoding.

    Stores past key and value representations across transformer layers,
    reducing decoding complexity per token from O(N^2) to O(N).
    """

    def __init__(
        self,
        n_layers: int,
        max_batch_size: int = 4,
        max_seq_len: int = 2048,
        n_kv_heads: int = 2,
        head_dim: int = 32,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ) -> None:
        self.n_layers = n_layers
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype

        # Preallocated buffers: [n_layers, max_batch, n_kv_heads, max_seq_len, head_dim]
        # Using preallocated buffers guarantees zero allocation overhead in decode loops
        self.k_cache = [
            torch.zeros(
                (max_batch_size, n_kv_heads, max_seq_len, head_dim),
                dtype=dtype,
                device=device,
            )
            for _ in range(n_layers)
        ]
        self.v_cache = [
            torch.zeros(
                (max_batch_size, n_kv_heads, max_seq_len, head_dim),
                dtype=dtype,
                device=device,
            )
            for _ in range(n_layers)
        ]
        self.cur_seq_len = 0

    def update(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
        start_pos: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Updates cache at start_pos and returns key, value up to current sequence length.

        Args:
            layer_idx: Layer index [0..n_layers-1]
            k: New key tensor [batch, n_kv_heads, seq_len, head_dim]
            v: New value tensor [batch, n_kv_heads, seq_len, head_dim]
            start_pos: Position index where new tokens begin

        Returns:
            Tuple (cached_keys, cached_values) up to start_pos + seq_len
        """
        batch_size, _, seq_len, _ = k.shape
        end_pos = start_pos + seq_len

        if end_pos > self.max_seq_len or batch_size > self.max_batch_size:
            # Dynamically resize buffers if sequence or batch exceeds preallocated maximum
            new_seq_len = max(end_pos, self.max_seq_len * 2)
            new_batch_size = max(batch_size, self.max_batch_size)
            self._resize(new_batch_size, new_seq_len)

        # Ensure cache is on same device & dtype
        if self.k_cache[layer_idx].device != k.device:
            self.k_cache[layer_idx] = self.k_cache[layer_idx].to(k.device)
            self.v_cache[layer_idx] = self.v_cache[layer_idx].to(v.device)

        self.k_cache[layer_idx][:batch_size, :, start_pos:end_pos, :] = k
        self.v_cache[layer_idx][:batch_size, :, start_pos:end_pos, :] = v

        self.cur_seq_len = max(self.cur_seq_len, end_pos)
        return (
            self.k_cache[layer_idx][:batch_size, :, :end_pos, :],
            self.v_cache[layer_idx][:batch_size, :, :end_pos, :],
        )

    def _resize(self, new_batch_size: int, new_seq_len: int) -> None:
        """Resizes the preallocated cache buffers."""
        old_seq_len = self.max_seq_len
        old_batch = self.max_batch_size
        self.max_batch_size = new_batch_size
        self.max_seq_len = new_seq_len

        for l in range(self.n_layers):
            new_k = torch.zeros(
                (new_batch_size, self.n_kv_heads, new_seq_len, self.head_dim),
                dtype=self.dtype,
                device=self.k_cache[l].device,
            )
            new_v = torch.zeros(
                (new_batch_size, self.n_kv_heads, new_seq_len, self.head_dim),
                dtype=self.dtype,
                device=self.v_cache[l].device,
            )
            # Copy existing contents
            new_k[:old_batch, :, :old_seq_len, :] = self.k_cache[l]
            new_v[:old_batch, :, :old_seq_len, :] = self.v_cache[l]
            self.k_cache[l] = new_k
            self.v_cache[l] = new_v

    def reset(self) -> None:
        """Resets the current sequence position to 0 for a new generation turn."""
        self.cur_seq_len = 0
        for l in range(self.n_layers):
            self.k_cache[l].zero_()
            self.v_cache[l].zero_()

    def get_memory_bytes(self) -> int:
        """Returns the current allocated memory of the cache in bytes."""
        elem_size = 4 if self.dtype == torch.float32 else 2
        return 2 * self.n_layers * self.max_batch_size * self.n_kv_heads * self.max_seq_len * self.head_dim * elem_size

    def get_memory_mb(self) -> float:
        return self.get_memory_bytes() / (1024.0 * 1024.0)


class GroupedQueryAttention(nn.Module):
    """Grouped-Query Attention (GQA) layer with RoPE and KV-Cache support.

    Generalizes Multi-Head Attention (MHA) and Multi-Query Attention (MQA).
    - If n_kv_heads == n_heads -> Standard MHA
    - If n_kv_heads == 1 -> Multi-Query Attention (MQA)
    - If 1 < n_kv_heads < n_heads -> Grouped-Query Attention (GQA)
    """

    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int = 0,
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.dim = config.dim
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.num_rep = config.num_queries_per_kv

        # Projections
        self.q_proj = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(self.n_heads * self.head_dim, self.dim, bias=False)

        # Rotary Positional Embedding
        self.rope = RotaryEmbedding(
            dim=self.head_dim,
            max_seq_len=config.max_seq_len,
            base=config.rope_theta,
        )

        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0.0 else nn.Identity()

        # Cache for last attention weights (for visualization in dashboard)
        self.last_attention_weights: Optional[torch.Tensor] = None

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int = 0,
        kv_cache: Optional[KVCache] = None,
        mask: Optional[torch.Tensor] = None,
        save_attention_weights: bool = False,
    ) -> torch.Tensor:
        """Performs multi-head / grouped-query attention forward pass.

        Args:
            x: Input tensor of shape [batch_size, seq_len, dim]
            start_pos: Position index for rotary embedding & KV cache
            kv_cache: Optional KVCache instance for autoregressive decoding
            mask: Optional causal attention mask [batch, 1, seq_len, total_seq_len]
            save_attention_weights: Whether to store attention weights for visualization

        Returns:
            Output tensor of shape [batch_size, seq_len, dim]
        """
        batch_size, seq_len, _ = x.shape

        # Linear projections
        # q: [batch, seq_len, n_heads * head_dim]
        # k, v: [batch, seq_len, n_kv_heads * head_dim]
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to [batch, heads, seq_len, head_dim]
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        q, k = self.rope(q, k, seq_len=seq_len, start_pos=start_pos)

        # KV-Cache management
        if kv_cache is not None:
            k, v = kv_cache.update(self.layer_idx, k, v, start_pos)

        total_kv_len = k.shape[2]

        # Expand Key and Value heads to match Query heads if using GQA or MQA
        k_rep = repeat_kv(k, self.num_rep)  # [batch, n_heads, total_kv_len, head_dim]
        v_rep = repeat_kv(v, self.num_rep)  # [batch, n_heads, total_kv_len, head_dim]

        # Scaled Dot-Product Attention: Q * K^T / sqrt(head_dim)
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k_rep.transpose(-2, -1)) * scale  # [batch, n_heads, seq_len, total_kv_len]

        # Causal Masking
        if mask is not None:
            scores = scores + mask
        elif seq_len > 1:
            # Prefill stage: generate lower triangular causal mask
            causal_mask = torch.full(
                (seq_len, total_kv_len),
                fill_value=float("-inf"),
                device=x.device,
                dtype=scores.dtype,
            )
            causal_mask = torch.triu(causal_mask, diagonal=start_pos + 1)
            scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)

        # Softmax normalization over key dimension
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        if save_attention_weights:
            self.last_attention_weights = attn_weights.detach().cpu()

        # Weighted sum with Values: [batch, n_heads, seq_len, head_dim]
        output = torch.matmul(attn_weights, v_rep)

        # Transpose back and project out: [batch, seq_len, dim]
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.out_proj(output)
