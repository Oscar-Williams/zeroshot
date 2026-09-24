"""Official ProgramBench evaluation and leaderboard-identical scoring."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from .config import Experiment, pins
from .util import download, log, read_json, write_json


def leaderboard_ignores(cache: Path) -> dict[str, list[str]]:
    pin = pins()["leaderboard_ignores"]
    return json.loads(download(pin["url"], pin["sha256"], cache / "downloads" / "ignored_tests.json").read_text())


def score_eval(eval_json: Path, instance_id: str, ignores: dict[str, list[str]]) -> dict[str, Any]:
    """Score exactly like the leaderboard: ProgramBench's active-branch/ignored-test filtering, then
    the registry's ignore map (``programbench.submission`` + ``compile_leaderboard.py``)."""
    from programbench.submission import benchmark_instances, score_from_tests, test_results_map

    instance = benchmark_instances()[instance_id]
    tests = test_results_map(eval_json, instance)
    ignore = set(ignores.get(instance_id, []))
    kept = {name: passed for name, passed in tests.items() if name not in ignore}
    raw = json.loads(eval_json.read_text())
    return {
        "score": score_from_tests(tests, ignore),
        "passed": sum(kept.values()),
        "scored_tests": len(kept),
        "error_code": raw.get("error_code"),
        "error_details": (raw.get("error_details") or "")[:2000] if isinstance(raw.get("error_details"), str) else raw.get("error_details"),
        "executable_hash": raw.get("executable_hash"),
        "tests": kept,
    }


def targets(results: Path) -> list[tuple[str, Path]]:
    """Every archive worth scoring: each attempt's final workspace and per-round snapshots."""
    items = []
    for attempt in sorted((results / "attempts").glob("*")):
        if not attempt.is_dir() or ".incomplete-" in attempt.name:
            continue
        if (attempt / "submission.tar.gz").exists():
            items.append((f"{attempt.name}__final", attempt / "submission.tar.gz"))
        for snap in sorted((attempt / "snapshots").glob("*.tar.gz")):
            items.append((f"{attempt.name}__{snap.name.removesuffix('.tar.gz')}", snap))
    return items


def evaluate(exp: Experiment, results: Path, cache: Path, force: bool = False) -> dict[str, Any]:
    evals = results / "evals"
    pending = []
    for label, archive in targets(results):
        run_dir = evals / label
        instance_dir = run_dir / exp.instance_id
        instance_dir.mkdir(parents=True, exist_ok=True)
        link = instance_dir / "submission.tar.gz"
        if not link.exists():
            os.link(archive, link)
        if force or not (instance_dir / f"{exp.instance_id}.eval.json").exists():
            pending.append(run_dir)
    if pending:
        cfg = exp.raw["eval"]
        log(f"evaluating {len(pending)} archive(s) with programbench eval")
        args = ["programbench", "eval", *map(str, pending), "--workers", str(cfg["workers"]), "--docker-cpus", str(cfg["docker_cpus"])]
        if force:
            args.append("--force")
        with (results / "programbench-eval.log").open("ab") as out:
            subprocess.run(args, stdout=out, stderr=subprocess.STDOUT, check=False)
    ignores = leaderboard_ignores(cache)
    scores: dict[str, Any] = {}
    for label, _ in targets(results):
        eval_json = evals / label / exp.instance_id / f"{exp.instance_id}.eval.json"
        if eval_json.exists():
            scores[label] = score_eval(eval_json, exp.instance_id, ignores)
        else:
            scores[label] = {"score": None, "error_code": "not_evaluated"}
    write_json(results / "scores.json", {k: {kk: vv for kk, vv in v.items() if kk != "tests"} for k, v in scores.items()})
    write_json(results / "scores-per-test.json", {k: v.get("tests", {}) for k, v in scores.items()})
    return scores


def load_scores(results: Path) -> dict[str, Any]:
    path = results / "scores.json"
    return read_json(path) if path.exists() else {}
