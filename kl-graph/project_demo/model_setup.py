"""Configure the optional ChatGPT and local embedding combination."""

from __future__ import annotations

import os
from pathlib import Path
from secrets import token_urlsafe

from fastapi import APIRouter

from .codex_bridge import CodexBridge
from .local_embedding import DIMENSIONS, MODEL_NAME, QUERY_INSTRUCTION, LocalEmbedding


def setup_chatgpt(port: int, data_dir: Path, model: str | None) -> tuple[APIRouter, APIRouter]:
    codex = CodexBridge(model)
    codex.verify_login()
    embedding = LocalEmbedding(data_dir / "models")
    embedding.load()

    # Native LiteLLM and the Demo answer client share these private loopback endpoints.
    # Codex itself continues to authenticate via the official CLI.
    base_url = f"http://127.0.0.1:{port}/v1"
    token = token_urlsafe(32)
    no_proxy = ",".join(dict.fromkeys(filter(None, (
        *os.environ.get("NO_PROXY", "").split(","), *os.environ.get("no_proxy", "").split(","),
        "127.0.0.1", "localhost", "::1",
    ))))
    os.environ.update({
        "KL_LLM_FLASH_PROVIDER": "openai",
        "KL_LLM_FLASH_BASE_URL": base_url,
        "KL_LLM_FLASH_MODEL": codex.model_name,
        "OPENAI_API_KEY": token,
        "KL_EMBED_BASE_URL": base_url,
        "KL_EMBED_MODEL": MODEL_NAME,
        "KL_EMBEDDING_DIM": str(DIMENSIONS),
        "KL_EMBED_SEND_DIMENSIONS": "0",
        "KL_EMBED_QUERY_INSTRUCTION": QUERY_INSTRUCTION,
        "KL_EMBED_API_KEY": token,
        "DEMO_ANSWER_BASE_URL": base_url,
        "DEMO_ANSWER_MODEL": codex.model_name,
        "DEMO_ANSWER_API_KEY": token,
        "DEMO_TOKENIZE_URL": "",
        "NO_PROXY": no_proxy,
        "no_proxy": no_proxy,
    })
    os.environ.setdefault("KL_LLM_BATCH_SIZE", "12")
    os.environ.setdefault("KL_LLM_FLASH_TIMEOUT", "230")
    os.environ.setdefault("KL_EMBED_BATCH_SIZE", "2")
    os.environ.setdefault("KL_EMBED_CONCURRENCY", "1")
    os.environ.setdefault("KL_EMBED_TIMEOUT", "120")
    os.environ.setdefault("DEMO_INGEST_TIMEOUT_SECONDS", "7200")
    os.environ.setdefault("DEMO_ANSWER_TIMEOUT_SECONDS", "230")
    os.environ.setdefault("DEMO_TIKTOKEN_ENCODING", "o200k_base")
    os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(Path(__file__).resolve().parent.parent /
                                                   "data/project_demo/tokenizers"))
    # Fail before serving if the explicit local estimator cannot be loaded.
    # This is payload token estimation, not a claim about Codex's hidden framing.
    import tiktoken
    tiktoken.get_encoding(os.environ["DEMO_TIKTOKEN_ENCODING"])

    from kl_graph.config import cfg

    llm = cfg.services.llm_flash
    embeddings = cfg.services.embedding
    if (llm.provider, llm.base_url, llm.model) != ("openai", base_url, codex.model_name) or (
        embeddings.base_url, embeddings.model, embeddings.dim, embeddings.api_key,
    ) != (base_url, MODEL_NAME, DIMENSIONS, token) or embeddings.query_instruction != QUERY_INSTRUCTION:
        raise RuntimeError("KL-Graph 的 config.yaml 覆盖了 --chatgpt 指定的模型地址或配置")
    return codex.router(token), embedding.router(token)
