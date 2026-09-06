"""Self-contained Byte/Subword Tokenizer and ChatML Formatter.

Designed for robust local inference without external dependencies.
Features:
- Special control tokens (<|pad|>, <|unk|>, <|bos|>, <|eos|>, <|im_start|>, <|im_end|>)
- Common English subword vocabulary + full byte-level fallback for 100% token coverage
- ChatML formatting aligned with OpenAI / LLaMA-3 instruction standards
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple, Union

import torch

SPECIAL_TOKENS = [
    "<|pad|>",
    "<|unk|>",
    "<|bos|>",
    "<|eos|>",
    "<|im_start|>",
    "<|im_end|>",
]

# Curated high-frequency English words/subwords to provide natural tokenization
COMMON_VOCAB = [
    " ", "the", "be", "to", "of", "and", "a", "in", "that", "have", "I", "it",
    "for", "not", "on", "with", "he", "as", "you", "do", "at", "this", "but",
    "his", "by", "from", "they", "we", "say", "her", "she", "or", "an", "will",
    "my", "one", "all", "would", "there", "their", "what", "so", "up", "out",
    "if", "about", "who", "get", "which", "go", "me", "when", "make", "can",
    "like", "time", "no", "just", "him", "know", "take", "people", "into",
    "year", "your", "good", "some", "could", "them", "see", "other", "than",
    "then", "now", "look", "only", "come", "its", "over", "think", "also",
    "back", "after", "use", "two", "how", "our", "work", "first", "well", "way",
    "even", "new", "want", "because", "any", "these", "give", "day", "most", "us",
    "system", "user", "assistant", "model", "prompt", "token", "attention",
    "transformer", "layer", "cache", "query", "key", "value", "research",
    "loss", "learning", "data", "science", "deep", "neural", "network", "code",
    "python", "tensor", "matrix", "optimization", "hill", "climbing", "algorithm",
    "hyperparameter", "paper", "metric", "perplexity", "latency", "memory",
    "is", "are", "was", "were", "been", "being", "has", "had", "having",
    "hello", "hi", "help", "please", "thank", "thanks", "explain", "generate",
    "answer", "question", "task", "solve", "analysis", "result", "test",
    "true", "false", "null", "none", "def", "return", "class", "import",
]


class Tokenizer:
    """Byte-level Subword Tokenizer with ChatML template support."""

    def __init__(self, vocab_size: int = 2048) -> None:
        self.vocab_size = vocab_size

        self.pad_token = "<|pad|>"
        self.unk_token = "<|unk|>"
        self.bos_token = "<|bos|>"
        self.eos_token = "<|eos|>"
        self.im_start_token = "<|im_start|>"
        self.im_end_token = "<|im_end|>"

        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}

        # 1. Assign special tokens
        for idx, token in enumerate(SPECIAL_TOKENS):
            self._add_token(token, idx)

        # 2. Add 256 raw byte representations (guarantees zero OOV errors)
        self.byte_start = len(self.token_to_id)
        for b in range(256):
            b_char = chr(b)
            self._add_token(b_char, self.byte_start + b)
        self.byte_end = self.byte_start + 256

        # 3. Add curated common vocabulary
        curr_id = len(self.token_to_id)
        for word in COMMON_VOCAB:
            if word not in self.token_to_id and curr_id < vocab_size:
                self._add_token(word, curr_id)
                curr_id += 1
            if (" " + word) not in self.token_to_id and curr_id < vocab_size:
                self._add_token(" " + word, curr_id)
                curr_id += 1

        # 4. Fill remaining vocabulary with synthetic pairs/tokens up to vocab_size
        fill_idx = 0
        while curr_id < vocab_size:
            dummy = f"<|tok_{fill_idx}|>"
            self._add_token(dummy, curr_id)
            curr_id += 1
            fill_idx += 1

        self.pad_id = self.token_to_id[self.pad_token]
        self.unk_id = self.token_to_id[self.unk_token]
        self.bos_id = self.token_to_id[self.bos_token]
        self.eos_id = self.token_to_id[self.eos_token]
        self.im_start_id = self.token_to_id[self.im_start_token]
        self.im_end_id = self.token_to_id[self.im_end_token]

        # Regex for greedy subword matching
        sorted_subwords = sorted(
            [k for k in self.token_to_id.keys() if len(k) > 1 and not k.startswith("<|")],
            key=len,
            reverse=True,
        )
        self._pattern = (
            re.compile(
                r"(<\|im_start\|>|<\|im_end\|>|<\|bos\|>|<\|eos\|>|<\|pad\|>|<\|unk\|>)"
                + ("|" + "|".join(re.escape(w) for w in sorted_subwords) if sorted_subwords else "")
            )
        )

    def _add_token(self, token: str, idx: int) -> None:
        self.token_to_id[token] = idx
        self.id_to_token[idx] = token

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> List[int]:
        """Encodes text into a sequence of token IDs using maximal matching + byte fallback."""
        tokens: List[int] = []
        if add_bos:
            tokens.append(self.bos_id)

        pos = 0
        n = len(text)
        while pos < n:
            match = self._pattern.search(text, pos)
            if match and match.start() == pos:
                piece = match.group(0)
                tokens.append(self.token_to_id[piece])
                pos = match.end()
            else:
                # Byte fallback for individual characters
                char = text[pos]
                for b in char.encode("utf-8"):
                    tokens.append(self.byte_start + b)
                pos += 1

        if add_eos:
            tokens.append(self.eos_id)

        return tokens

    def decode(self, token_ids: Union[List[int], torch.Tensor], skip_special_tokens: bool = True) -> str:
        """Decodes token IDs back into string representation."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        pieces: List[str] = []
        byte_buffer = bytearray()
        special_ids = {
            self.pad_id,
            self.unk_id,
            self.bos_id,
            self.eos_id,
            self.im_start_id,
            self.im_end_id,
        }

        def flush_bytes() -> None:
            nonlocal byte_buffer
            if byte_buffer:
                pieces.append(byte_buffer.decode("utf-8", errors="replace"))
                byte_buffer = bytearray()

        for tid in token_ids:
            if skip_special_tokens and tid in special_ids:
                continue
            if self.byte_start <= tid < self.byte_end:
                byte_buffer.append(tid - self.byte_start)
            else:
                flush_bytes()
                token_str = self.id_to_token.get(tid, "")
                pieces.append(token_str)

        flush_bytes()
        return "".join(pieces)

    def format_chat_prompt(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> str:
        """Formats multi-turn messages into ChatML standard.

        Example:
        <|im_start|>system
        You are a helpful assistant.<|im_end|>
        <|im_start|>user
        Hello!<|im_end|>
        <|im_start|>assistant
        """
        prompt_parts: List[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "").strip()
            prompt_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")

        if add_generation_prompt:
            prompt_parts.append("<|im_start|>assistant\n")

        return "".join(prompt_parts)

    def encode_chat(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = True,
        device: str = "cpu",
    ) -> torch.Tensor:
        """Formats and encodes chat messages directly into a PyTorch tensor."""
        formatted_prompt = self.format_chat_prompt(
            messages, add_generation_prompt=add_generation_prompt
        )
        token_ids = self.encode(formatted_prompt, add_bos=True)
        return torch.tensor([token_ids], dtype=torch.long, device=device)
