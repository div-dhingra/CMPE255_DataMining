"""SOTA Transformer Decoder Model Architecture.

Integrates:
- Pre-RMSNorm residual connections (LLaMA style)
- Grouped-Query Attention with Rotary Position Embeddings (RoPE)
- SwiGLU Gated Feed-Forward Networks
- Low-latency KV-Cache autoregressive generation
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.attention import GroupedQueryAttention, KVCache
from src.model.config import ModelConfig
from src.model.primitives import RMSNorm, SwiGLU


class TransformerBlock(nn.Module):
    """A single Transformer Decoder Block using Pre-RMSNorm, GQA, and SwiGLU."""

    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # Attention block with Pre-RMSNorm
        self.attn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.attn = GroupedQueryAttention(config=config, layer_idx=layer_idx)

        # SwiGLU MLP block with Pre-RMSNorm
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.mlp = SwiGLU(
            dim=config.dim,
            hidden_dim=config.hidden_dim,  # type: ignore[arg-type]
            dropout=config.dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int = 0,
        kv_cache: Optional[KVCache] = None,
        mask: Optional[torch.Tensor] = None,
        save_attention_weights: bool = False,
    ) -> torch.Tensor:
        # Pre-norm Self-Attention + Residual
        normed_x = self.attn_norm(x)
        attn_out = self.attn(
            normed_x,
            start_pos=start_pos,
            kv_cache=kv_cache,
            mask=mask,
            save_attention_weights=save_attention_weights,
        )
        h = x + attn_out

        # Pre-norm SwiGLU MLP + Residual
        normed_h = self.ffn_norm(h)
        mlp_out = self.mlp(normed_h)
        out = h + mlp_out

        return out


class SotaDecoderLLM(nn.Module):
    """State-of-the-Art Autoregressive Transformer Language Model.

    Lightweight, high-performance architecture sized to run seamlessly on laptop
    GPU (MPS / CUDA) and CPU.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        # Token Embedding Table
        self.embed_tokens = nn.Embedding(config.vocab_size, config.dim)

        # Stacked Decoder Blocks
        self.layers = nn.ModuleList(
            [TransformerBlock(config=config, layer_idx=i) for i in range(config.n_layers)]
        )

        # Final RMSNorm
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)

        # Output LM Head (Unembedding)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)

        # Weight tying (Press & Wolf 2017)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        # Weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initializes weights using standard scaled normal distribution."""
        if isinstance(module, nn.Linear):
            # Scale initialization by 1 / sqrt(2 * n_layers) for residual projections (GPT-2 style)
            std = 0.02
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        start_pos: int = 0,
        kv_cache: Optional[KVCache] = None,
        targets: Optional[torch.Tensor] = None,
        save_attention_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass for both prefill training and autoregressive generation.

        Args:
            input_ids: Token IDs of shape [batch_size, seq_len]
            start_pos: Position index for KV-cache and RoPE
            kv_cache: Optional KVCache instance
            targets: Optional ground-truth token targets for cross-entropy loss computation
            save_attention_weights: Whether to record attention maps for inspection

        Returns:
            Tuple (logits, loss). Loss is None if targets is None.
        """
        batch_size, seq_len = input_ids.shape

        # Token embedding lookup
        h = self.embed_tokens(input_ids)

        # Pass through all transformer blocks
        for layer in self.layers:
            h = layer(
                h,
                start_pos=start_pos,
                kv_cache=kv_cache,
                save_attention_weights=save_attention_weights,
            )

        # Final RMS Normalization
        h = self.norm(h)

        # Project to vocabulary logits: [batch_size, seq_len, vocab_size]
        logits = self.lm_head(h)

        loss = None
        if targets is not None:
            # Shift targets for next-token prediction if not already shifted
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
                ignore_index=-100,
            )

        return logits, loss

    def create_kv_cache(
        self,
        max_batch_size: int = 1,
        max_seq_len: Optional[int] = None,
        device: Optional[str] = None,
    ) -> KVCache:
        """Instantiates a compatible KVCache buffer for this model."""
        target_len = max_seq_len or self.config.max_seq_len
        target_dev = device or next(self.parameters()).device
        return KVCache(
            n_layers=self.config.n_layers,
            max_batch_size=max_batch_size,
            max_seq_len=target_len,
            n_kv_heads=self.config.n_kv_heads,
            head_dim=self.config.head_dim,
            dtype=self.embed_tokens.weight.dtype,
            device=str(target_dev),
        )

    def get_num_params(self, non_embedding: bool = False) -> int:
        """Returns total parameter count."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.embed_tokens.weight.numel()
        return n_params

    def get_layer_breakdown(self) -> List[Dict[str, Union[str, int, float]]]:
        """Provides detailed layer-by-layer architectural diagnostics."""
        breakdown = []
        breakdown.append({
            "name": "embed_tokens",
            "type": "Embedding",
            "shape": f"[{self.config.vocab_size}, {self.config.dim}]",
            "params": self.embed_tokens.weight.numel(),
            "details": f"Vocab: {self.config.vocab_size}, Dim: {self.config.dim}",
        })
        for i, layer in enumerate(self.layers):
            q_p = layer.attn.q_proj.weight.numel()
            k_p = layer.attn.k_proj.weight.numel()
            v_p = layer.attn.v_proj.weight.numel()
            o_p = layer.attn.out_proj.weight.numel()
            mlp_p = (
                layer.mlp.w_gate.weight.numel()
                + layer.mlp.w_up.weight.numel()
                + layer.mlp.w_down.weight.numel()
            )
            breakdown.append({
                "name": f"layer_{i}",
                "type": "TransformerBlock (GQA + SwiGLU)",
                "params": q_p + k_p + v_p + o_p + mlp_p + layer.attn_norm.weight.numel() + layer.ffn_norm.weight.numel(),
                "attention_heads": f"{self.config.n_heads} Q / {self.config.n_kv_heads} KV ({self.config.attention_type})",
                "mlp_hidden_dim": self.config.hidden_dim,
                "rope_theta": self.config.rope_theta,
            })
        breakdown.append({
            "name": "final_norm",
            "type": "RMSNorm",
            "params": self.norm.weight.numel(),
            "details": f"Dim: {self.config.dim}",
        })
        breakdown.append({
            "name": "lm_head",
            "type": "Linear (Tied)",
            "params": 0 if self.config.tie_word_embeddings else self.lm_head.weight.numel(),
            "details": f"Out: {self.config.vocab_size}",
        })
        return breakdown
