"""Summaries: per-attempt table, per-arm aggregates, and the pre-registered decision rule."""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any

from . import accounting, audit
from .config import Experiment
from .evaluate import load_scores
from .util import read_json, write_json


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.1f}%"


def build(exp: Experiment, results: Path, proxy_log: str | None = None) -> dict[str, Any]:
    scores = load_scores(results)
    attempts = []
    for directory in sorted((results / "attempts").glob("*")):
        if not directory.is_dir() or ".incomplete-" in directory.name or not (directory / "attempt.json").exists():
            continue
        meta = read_json(directory / "attempt.json")
        label = meta["label"]
        tokens = accounting.usage(directory)
        rounds = {k.split("__", 1)[1]: v.get("score") for k, v in scores.items() if k.startswith(f"{label}__")}
        final = rounds.get("final")
        first = rounds.get("build-1")
        record = {
            "label": label,
            "arm": meta["arm"],
            "state": meta.get("state"),
            "error": meta.get("error"),
            "terminal": meta.get("terminal"),
            "force_stopped": meta.get("force_stopped"),
            "builds": meta.get("builds"),
            "checks": meta.get("checks"),
            "build_outcomes": [b.get("code") or b.get("status") for b in meta.get("build_outcomes", [])],
            "verdicts": [v.get("verdict") or f"{v.get('status')}:{v.get('code')}" for v in meta.get("verdicts", [])],
            "wall_seconds": meta.get("wall_seconds"),
            "score_final": final,
            "score_first_build": first,
            "gain_over_first_build": None if final is None or first is None else final - first,
            "scores_by_round": rounds,
            "eval_error": (scores.get(f"{label}__final") or {}).get("error_code"),
            "tokens": tokens,
            "cost_usd": {node: round(accounting.cost(t, exp.pricing), 4) for node, t in tokens.items()},
            "commands": audit.command_audit(directory / "trajectories.tar.gz"),
            "checker_edits": audit.checker_edits(directory, (meta.get("snapshots") or {}).get("check", 0)),
            "reference_at_snapshot": meta.get("reference_at_snapshot"),
        }
        attempts.append(record)
    arms = {}
    for arm in sorted({a["arm"] for a in attempts}):
        group = [a for a in attempts if a["arm"] == arm]
        finals = [a["score_final"] for a in group if a["score_final"] is not None]
        costs = [a["cost_usd"].get("total", 0.0) for a in group]
        arms[arm] = {
            "runs": len(group),
            "scored": len(finals),
            "final_scores": finals,
            "median_final": statistics.median(finals) if finals else None,
            "mean_final": statistics.fmean(finals) if finals else None,
            "collapses_below_5pct": sum(1 for s in finals if s < 0.05),
            "mean_cost_usd": statistics.fmean(costs) if costs else None,
            "total_cost_usd": sum(costs),
        }
    gains = [a["gain_over_first_build"] for a in attempts if a["arm"] == "loop" and a["gain_over_first_build"] is not None]
    decision = None
    if gains:
        mean_gain = statistics.fmean(gains)
        verdict = "supported" if all(g > 0 for g in gains) and mean_gain >= 0.05 else "refuted" if mean_gain < 0.02 else "inconclusive"
        decision = {"paired_gains": gains, "mean_gain": mean_gain, "all_positive": all(g > 0 for g in gains), "verdict": verdict, "rule": exp.raw.get("decision_rule")}
    summary = {
        "experiment": exp.id,
        "description": exp.raw.get("description"),
        "model": exp.model,
        "effort": exp.effort,
        "attempts": attempts,
        "arms": arms,
        "h1_decision": decision,
        "secrets": audit.secret_scan(results),
        "egress": audit.proxy_audit(proxy_log) if proxy_log is not None else None,
    }
    write_json(results / "summary.json", summary)
    (results / "summary.md").write_text(markdown(summary))
    return summary


def markdown(summary: dict[str, Any]) -> str:
    lines = [f"# {summary['experiment']}", "", summary.get("description") or "", "", f"Model `{summary['model']}` at effort `{summary['effort']}`.", ""]
    lines += ["| Run | Arm | State | Rounds | Verdicts | First build | Final | Gain | Wall (min) | Cost (USD) | Flags |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for a in summary["attempts"]:
        flags = [k for k, v in (a["commands"].get("rule_counts") or {}).items() if v]
        if a["commands"].get("web_search_calls"):
            flags.append("web_search")
        if any(r["workspace_changes"] for r in a["checker_edits"]):
            flags.append("checker_edited_sources")
        if any(state != "in_place" for state in (a.get("reference_at_snapshot") or {}).values()):
            flags.append("reference_moved")
        gain = a["gain_over_first_build"]
        lines.append(
            f"| {a['label']} | {a['arm']} | {a['state']} | {a['builds'] or 0} | {', '.join(v or '—' for v in a['verdicts']) or '—'} | "
            f"{_pct(a['score_first_build'])} | {_pct(a['score_final'])} | {'—' if gain is None else f'{100 * gain:+.1f} pp'} | "
            f"{(a['wall_seconds'] or 0) / 60:.0f} | {a['cost_usd'].get('total', 0):.2f} | {', '.join(flags) or '—'} |"
        )
    lines += ["", "| Arm | Runs | Median final | Mean final | Collapses | Mean cost |", "|---|---|---|---|---|---|"]
    for arm, s in summary["arms"].items():
        lines.append(f"| {arm} | {s['runs']} | {_pct(s['median_final'])} | {_pct(s['mean_final'])} | {s['collapses_below_5pct']} | ${(s['mean_cost_usd'] or 0):.2f} |")
    d = summary.get("h1_decision")
    if d:
        lines += ["", f"**H1 (pre-registered):** {d['verdict']}. Mean paired gain {100 * d['mean_gain']:+.1f} pp; all positive: {d['all_positive']}."]
    s = summary["secrets"]
    lines += ["", f"Secret scan: literal key checked={s['checked_literal_key']}, hits={len(s['literal_key_hits'])}."]
    if summary.get("egress"):
        lines.append(f"Egress: connected {summary['egress']['established']}; refused {summary['egress']['refused']}.")
    return "\n".join(lines) + "\n"
