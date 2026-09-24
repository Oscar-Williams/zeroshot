"""Token usage and cost per node, from the run's durable Zeroshot ledger."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from . import ledger
from .util import read_json

FIELDS = ("inputTokens", "outputTokens", "cacheReadInputTokens", "cacheCreationInputTokens")


def usage_from_events(events: Iterable[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Sum ``token_usage_observed`` per graph node (``build``/``check``) and in total.

    Codex reports ``inputTokens`` as all prompt tokens, with cache reads and cache writes as
    subsets, so uncached input is ``input - cache_read - cache_write``.
    """
    node_of_execution: dict[int, str] = {}
    totals: dict[str, dict[str, int]] = defaultdict(lambda: dict.fromkeys(FIELDS, 0))
    for event in events:
        if event.get("kind") == "node_started":
            reference = event.get("reference") or {}
            node_of_execution[int(reference.get("execution", -1))] = reference.get("node", "?")
        elif event.get("kind") == "token_usage_observed":
            node = node_of_execution.get(int(event.get("execution", -1)), "?")
            for field in FIELDS:
                value = int((event.get("usage") or {}).get(field) or 0)
                totals[node][field] += value
                totals["total"][field] += value
    return {node: dict(values) for node, values in totals.items()}


def usage(attempt_dir: Path) -> dict[str, dict[str, int]]:
    events = ledger.events(attempt_dir / "trajectories.tar.gz")
    if events is not None:
        return usage_from_events(events)
    meta_path = attempt_dir / "attempt.json"
    run_total = read_json(meta_path).get("token_usage") if meta_path.exists() else None
    return {"total": {field: int((run_total or {}).get(field) or 0) for field in FIELDS}} if run_total else {}


def cost(tokens: dict[str, int], pricing: dict[str, Any]) -> float:
    price = pricing["usd_per_million_tokens"]
    cache_read = tokens.get("cacheReadInputTokens", 0)
    cache_write = tokens.get("cacheCreationInputTokens", 0)
    uncached = max(0, tokens.get("inputTokens", 0) - cache_read - cache_write)
    return (
        uncached * price["input"]
        + cache_read * price["cached_input"]
        + cache_write * price["cache_write"]
        + tokens.get("outputTokens", 0) * price["output"]
    ) / 1_000_000
