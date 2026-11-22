"""OpenAI-compatible, CPU-only embeddings for the project demo."""

from __future__ import annotations

from hmac import compare_digest
from pathlib import Path
from threading import RLock
from typing import Annotated, Literal

import numpy as np
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
DIMENSIONS = 1024
# A CPU operating limit, not the model's 32K architectural context. Fail on
# overflow instead of silently discarding text. All current V4 chunks fit.
MAX_INPUT_TOKENS = 8192
QUERY_INSTRUCTION = "Given a question about project progress, retrieve relevant workplace records that answer the question."
_THREADS = 2

InputText = Annotated[str, Field(min_length=1)]
InputBatch = Annotated[list[InputText], Field(min_length=1, max_length=128)]


class _EmbeddingRequest(BaseModel):
    model: str
    input: InputText | InputBatch
    encoding_format: Literal["float"] = "float"
    dimensions: int | None = None


class LocalEmbedding:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self._lock = RLock()
        self._model = None
        self._tokenizer = None

    @property
    def model_path(self) -> Path:
        return self.cache_dir / "Qwen3-Embedding-0.6B"

    def load(self) -> None:
        """Load downloaded official weights once; serving never downloads code."""
        with self._lock:
            if self._model is not None:
                return
            try:
                import torch
                from transformers import AutoModel, AutoTokenizer
            except ImportError as exc:
                raise RuntimeError("本地 Qwen 向量模型需要在项目虚拟环境安装 CPU PyTorch 和 transformers>=4.57.6") from exc
            if not (self.model_path / "config.json").is_file():
                raise RuntimeError(f"请先将 {MODEL_NAME} 下载到 {self.model_path}")
            torch.set_num_threads(_THREADS)
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, local_files_only=True, padding_side="left", trust_remote_code=False)
            self._model = AutoModel.from_pretrained(
                self.model_path, local_files_only=True, trust_remote_code=False,
                dtype=torch.float32, attn_implementation="sdpa",
            ).eval()

    def _embed(self, texts: list[str]) -> list[list[float]]:
        return self._embed_with_usage(texts)[0]

    def _embed_with_usage(self, texts: list[str]) -> tuple[list[list[float]], int]:
        import torch
        import torch.nn.functional as F

        # Serialize CPU inference and process one sequence at a time to bound
        # peak memory. Validate the whole request before doing any inference.
        with self._lock:
            self.load()
            encoded = [self._tokenizer(text, truncation=False, return_tensors="pt") for text in texts]
            lengths = [int(item["input_ids"].shape[-1]) for item in encoded]
            if any(length > MAX_INPUT_TOKENS for length in lengths):
                raise ValueError(f"文本超过本机服务的 {MAX_INPUT_TOKENS} token 上限，请先分段；未截断文本")
            vectors = []
            with torch.inference_mode():
                for item in encoded:
                    output = self._model(**item, use_cache=False).last_hidden_state
                    pooled = last_token_pool(output, item["attention_mask"])
                    vectors.append(F.normalize(pooled, p=2, dim=1)[0].float().cpu().numpy().copy())
            vectors = np.asarray(vectors)
        if vectors.shape != (len(texts), DIMENSIONS) or not np.isfinite(vectors).all():
            raise RuntimeError("本地向量服务返回了无效向量")
        return vectors.tolist(), sum(lengths)

    def router(self, token: str) -> APIRouter:
        if not token:
            raise ValueError("本地向量服务需要非空访问令牌")
        expected = token.encode()
        router = APIRouter()

        @router.post("/v1/embeddings")
        def embeddings(request: _EmbeddingRequest, authorization: str = Header(default="")) -> dict:
            scheme, _, credentials = authorization.partition(" ")
            if scheme.lower() != "bearer" or not compare_digest(credentials.encode(), expected):
                raise HTTPException(401, "本地向量服务令牌无效", headers={"WWW-Authenticate": "Bearer"})
            if request.model != MODEL_NAME:
                raise HTTPException(400, f"向量服务仅提供 {MODEL_NAME}")
            if request.dimensions not in (None, DIMENSIONS):
                raise HTTPException(400, f"向量服务仅支持 {DIMENSIONS} 维")

            texts = [request.input] if isinstance(request.input, str) else request.input
            try:
                vectors, tokens = self._embed_with_usage(texts)
            except ValueError as exc:
                raise HTTPException(413, str(exc)) from exc
            return {
                "object": "list",
                "data": [
                    {"object": "embedding", "index": index, "embedding": vector}
                    for index, vector in enumerate(vectors)
                ],
                "model": MODEL_NAME,
                "usage": {"prompt_tokens": tokens, "total_tokens": tokens},
            }

        return router


def last_token_pool(hidden, attention_mask):
    """Official Qwen pooling, including both left- and right-padded batches."""
    import torch
    if attention_mask[:, -1].sum() == attention_mask.shape[0]:
        return hidden[:, -1]
    lengths = attention_mask.sum(dim=1) - 1
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), lengths]
