"""FastAPI Router for Chatbot Inference and Streaming.

Supports standard completion requests and Server-Sent Events (SSE) streaming
with live latency and memory telemetry.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.engine.chat_session import ChatSession
from src.engine.generator import GenerationConfig, TextGenerator
from src.model.config import ModelConfig
from src.model.tokenizer import Tokenizer
from src.model.transformer import SotaDecoderLLM

router = APIRouter(prefix="/api/chat", tags=["Chat & Inference"])

# Module-level singletons (lazily initialized)
_model: Optional[SotaDecoderLLM] = None
_tokenizer: Optional[Tokenizer] = None
_generator: Optional[TextGenerator] = None
_session: Optional[ChatSession] = None


def get_chat_session() -> ChatSession:
    global _model, _tokenizer, _generator, _session
    if _session is None:
        cfg = ModelConfig.preset_nano()
        _model = SotaDecoderLLM(cfg)
        _tokenizer = Tokenizer(cfg.vocab_size)
        _generator = TextGenerator(_model, _tokenizer)
        _session = ChatSession(_generator, mode="hybrid")
    return _session


class ChatMessage(BaseModel):
    role: str = Field(..., json_schema_extra={"example": "user"})
    content: str = Field(..., json_schema_extra={"example": "What is Rotary Position Embedding?"})


class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessage]
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=0, le=200)
    repetition_penalty: float = Field(default=1.15, ge=1.0, le=2.0)
    max_tokens: int = Field(default=128, ge=1, le=512)
    stream: bool = Field(default=False)
    system_prompt: Optional[str] = None


@router.post("/completions")
async def chat_completions(req: ChatCompletionRequest) -> Dict[str, Any]:
    """Generates a non-streaming chat completion."""
    session = get_chat_session()
    if req.system_prompt:
        session.system_prompt = req.system_prompt

    # Populate session history from messages
    session.messages = []
    if session.system_prompt:
        session.messages.append({"role": "system", "content": session.system_prompt})

    for m in req.messages[:-1]:
        session.add_message(m.role, m.content)

    last_user_msg = req.messages[-1].content if req.messages else "Hello"
    gen_cfg = GenerationConfig(
        max_new_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
        repetition_penalty=req.repetition_penalty,
    )

    result = session.chat_turn(last_user_msg, gen_config=gen_cfg)

    return {
        "id": "chatcmpl-sota-llm",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": result["reply"],
                },
                "finish_reason": "stop",
            }
        ],
        "telemetry": {
            "ttft_ms": result["ttft_ms"],
            "tokens_per_sec": result["tokens_per_sec"],
            "elapsed_sec": result["elapsed_sec"],
            "total_tokens": result["total_tokens"],
        },
        "history": result["history"],
    }


@router.post("/stream")
async def chat_stream(req: ChatCompletionRequest) -> StreamingResponse:
    """Streams token-by-token generation via Server-Sent Events (SSE)."""
    session = get_chat_session()
    if req.system_prompt:
        session.system_prompt = req.system_prompt

    session.messages = []
    if session.system_prompt:
        session.messages.append({"role": "system", "content": session.system_prompt})

    for m in req.messages[:-1]:
        session.add_message(m.role, m.content)

    last_user_msg = req.messages[-1].content if req.messages else "Hello"
    gen_cfg = GenerationConfig(
        max_new_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
        repetition_penalty=req.repetition_penalty,
    )

    def event_stream():
        for chunk in session.stream_turn(last_user_msg, gen_config=gen_cfg):
            payload = {
                "token": chunk.token_str,
                "step": chunk.step,
                "is_finished": chunk.is_finished,
                "finish_reason": chunk.finish_reason,
                "ttft_ms": round(chunk.ttft_ms, 2),
                "tokens_per_sec": round(chunk.tokens_per_sec, 2),
                "kv_memory_mb": round(chunk.kv_memory_mb, 3),
            }
            yield f"data: {json.dumps(payload)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/reset")
async def reset_session() -> Dict[str, str]:
    """Resets conversational history."""
    session = get_chat_session()
    session.reset()
    return {"status": "ok", "message": "Conversational session reset to initial state"}
