"""Shared utilities: RRF fusion, bounded concurrency."""

from __future__ import annotations

import asyncio
import re
from collections import defaultdict
from collections.abc import Coroutine
from typing import Any


# Trailing runs of the OpenAI version segment, e.g. the doubled suffix a
# caller produces when it appends /v1 to a base URL that already ends in /v1.
_TRAILING_V1_RUN = re.compile(r"(/v1)+$")


def litellm_base_url(provider: str, base_url: str) -> str:
    """Normalize a base URL to the shape litellm's transport for it expects.

    Both litellm transports treat the configured base URL as a root and
    append the endpoint path, but they disagree about where the ``/v1``
    segment lives, so the same gateway URL pasted by a user works for one
    transport and 404s on the other unless normalized:

    - **OpenAI-compatible** — the SDK hands ``base_url`` to the OpenAI
      client, whose own default root is ``https://api.openai.com/v1``; the
      client appends only ``/chat/completions`` / ``/embeddings``. The base
      must therefore end in **exactly one** ``/v1``: missing → gateway 404
      (e.g. DashScope serves only ``…/compatible-mode/v1/...``); doubled
      (``…/v1/v1``, produced by a launcher that appends /v1 to an
      already-versioned URL) 404s identically.
    - **Anthropic** — litellm appends ``/v1/messages`` itself, so the base
      must be a **bare host**: a pasted versioned URL doubles the segment
      (``POST /v1/v1/messages`` → 404, observed live).

    Empty values pass through so provider/litellm defaults still apply.

    Args:
        provider: LiteLLM provider name from config (e.g. ``openai``).
        base_url: Configured endpoint base URL, possibly empty.

    Returns:
        The base URL safe to pass as ``api_base`` for that provider.
    """
    url = (base_url or "").strip().rstrip("/")
    if not url:
        return url
    if provider.strip().lower() == "anthropic":
        return _TRAILING_V1_RUN.sub("", url)
    url = _TRAILING_V1_RUN.sub("/v1", url)
    return url if url.endswith("/v1") else f"{url}/v1"


def rrf(
    ranked_lists: list[list[tuple[str, float]]], k: int = 60
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion across multiple retrieval result lists.

    Each ranked_list is [(item_id, score), ...] in descending score order.
    Returns fused ranking as [(item_id, rrf_score), ...] in descending order.
    """
    scores: dict[str, float] = defaultdict(float)
    for ranked in ranked_lists:
        for rank, (item_id, _score) in enumerate(ranked):
            scores[item_id] += 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def dedup_ranked(
    ranked: list[tuple[str, float]],
) -> list[tuple[str, float]]:
    """Drop repeat ids within a single ranked list, keeping the first (best rank).

    Intended for use *before* :func:`rrf`: a single retrieval route can surface
    the same id at more than one position (e.g. a chunk that MENTIONS two
    matched entities is emitted once per entity during structural expansion).
    Left unchecked, that one route contributes ``1/(k+rank)`` twice for the same
    item, over-weighting it in the fusion. Deduping *within* the list before it
    reaches ``rrf`` collapses those repeats to the item's best (first) rank.

    This must never be applied *across* lists: an id legitimately present in
    several routes is exactly the multi-route corroboration signal RRF is
    designed to harvest.

    Args:
        ranked: A single route's ``[(item_id, score), ...]`` in rank order
            (best first).

    Returns:
        The same list with later repeats of any id removed; the first
        occurrence (best rank) is kept. Order is otherwise preserved.
    """
    seen: set[str] = set()
    out: list[tuple[str, float]] = []
    for item_id, score in ranked:
        if item_id in seen:
            continue
        seen.add(item_id)
        out.append((item_id, score))
    return out


async def semaphore_gather(
    tasks: list[Coroutine], max_concurrent: int = 10
) -> list[Any]:
    """Run async tasks with bounded concurrency."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def bounded(task):
        async with semaphore:
            return await task

    return await asyncio.gather(*[bounded(t) for t in tasks])
