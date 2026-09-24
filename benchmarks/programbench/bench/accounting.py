"""Token usage and cost, priced from Codex's own session transcripts.

Zeroshot's ledger cannot be summed directly: when a node resumes the same Codex thread (the
builder uses ``sessionScope: node_instance``), Codex reports the thread's running total and the
ledger stores it as that execution's usage, so later builder rounds would be counted again. Each
Codex transcript (``rollout-*.jsonl``) is one thread; its last ``token_count`` event carries the
thread's true cumulative usage, and the totals at each ``task_complete`` split it into turns
(one turn per graph execution of that node).
"""

from __future__ import annotations

import json
import tarfile
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from . import ledger
from .config import prompt

FIELDS = ("inputTokens", "outputTokens", "cacheReadInputTokens", "cacheCreationInputTokens")
_CODEX = {"input_tokens": "inputTokens", "output_tokens": "outputTokens", "cached_input_tokens": "cacheReadInputTokens", "cache_write_input_tokens": "cacheCreationInputTokens"}


def _zero() -> dict[str, int]:
    return dict.fromkeys(FIELDS, 0)


def _from_codex(usage: dict[str, Any]) -> dict[str, int]:
    return {ours: int(usage.get(theirs) or 0) for theirs, ours in _CODEX.items()}


def _minus(a: dict[str, int], b: dict[str, int]) -> dict[str, int]:
    return {k: a[k] - b[k] for k in FIELDS}


def _add(a: dict[str, int], b: dict[str, int]) -> None:
    for k in FIELDS:
        a[k] += b[k]


def node_of_session(first_prompt: str) -> str:
    if prompt("checker") in first_prompt:
        return "check"
    if prompt("builder") in first_prompt:
        return "build"
    return "other"


def sessions(trajectories: Path) -> list[dict[str, Any]]:
    """One record per Codex thread: its node, cumulative usage, and per-turn usage."""
    out = []
    if not trajectories.exists():
        return out
    with tarfile.open(trajectories) as tar:
        for member in tar:
            if not (member.isfile() and "/sessions/" in member.name and member.name.endswith(".jsonl")):
                continue
            data = tar.extractfile(member)
            if data is None:
                continue
            first_prompt, total, turns, last_turn_total = "", _zero(), [], _zero()
            for line in data.read().splitlines():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                payload = record.get("payload") or {}
                kind = payload.get("type")
                if not first_prompt and kind == "message" and payload.get("role") == "user":
                    text = " ".join(c.get("text", "") for c in payload.get("content") or [] if isinstance(c, dict))
                    if "Authored instructions" in text:
                        first_prompt = text
                elif kind == "token_count" and (payload.get("info") or {}).get("total_token_usage"):
                    total = _from_codex(payload["info"]["total_token_usage"])
                elif kind == "task_complete":
                    turns.append(_minus(total, last_turn_total))
                    last_turn_total = dict(total)
            if total != last_turn_total:  # usage after the last completed turn (a killed or timed-out turn)
                turns.append(_minus(total, last_turn_total))
            out.append({"file": member.name, "node": node_of_session(first_prompt), "total": total, "turns": turns})
    return out


def usage(attempt_dir: Path) -> dict[str, Any]:
    """Per-node and total usage from transcripts, the builder's first turn separately, and the
    ledger's figure for comparison."""
    per_node: dict[str, dict[str, int]] = defaultdict(_zero)
    first_build_turn = None
    # Transcript names start with their creation time, so the earliest builder thread holds build
    # 1 (a build error makes Zeroshot start a new thread for the next round).
    records = sorted(sessions(attempt_dir / "trajectories.tar.gz"), key=lambda s: Path(s["file"]).name)
    for session in records:
        _add(per_node[session["node"]], session["total"])
        _add(per_node["total"], session["total"])
        if session["node"] == "build" and session["turns"] and first_build_turn is None:
            first_build_turn = dict(session["turns"][0])
    ledger_events = ledger.events(attempt_dir / "trajectories.tar.gz")
    ledger_nodes = usage_from_events(ledger_events) if ledger_events else {}
    return {
        "nodes": dict(per_node),
        "first_build_turn": first_build_turn or _zero(),
        "sessions": [{"node": s["node"], "turns": len(s["turns"]), "total": s["total"]} for s in records],
        "ledger_total": ledger_nodes.get("total"),
        "ledger_nodes": ledger_nodes,
    }


def usage_from_events(events: Iterable[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Zeroshot ledger view (kept as a cross-check; over-counts resumed builder rounds)."""
    node_of_execution: dict[int, str] = {}
    totals: dict[str, dict[str, int]] = defaultdict(_zero)
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
