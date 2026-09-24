"""Official ProgramBench evaluation and leaderboard-identical scoring.

Everything that feeds a score is pinned: the eval image by digest, the hidden tests by Hugging
Face revision, ProgramBench by version, pytest-rerunfailures by version (bench/pbeval.py), and the
leaderboard's ignore list by commit.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from .config import Experiment, pins
from .util import SECRET_ENV, download, log, read_json, write_json


def leaderboard_ignores(cache: Path) -> dict[str, list[str]]:
    pin = pins()["leaderboard_ignores"]
    return json.loads(download(pin["url"], pin["sha256"], cache / "downloads" / "ignored_tests.json").read_text())


def eval_image_tag(exp: Experiment) -> str:
    """``task_cleanroom_v6@sha256:...`` so programbench runs the exact image the agent used."""
    from programbench.constants import image_name_from_instance_id

    repository, _, tag_and_digest = exp.task_image.partition(":")
    if repository != image_name_from_instance_id(exp.instance_id) or "@sha256:" not in tag_and_digest:
        raise ValueError(f"task image {exp.task_image} does not match instance {exp.instance_id}")
    return tag_and_digest


def score_eval(eval_json: Path, instance_id: str, ignores: dict[str, list[str]]) -> dict[str, Any]:
    """Score exactly like the leaderboard: ProgramBench's active-branch/ignored-test filtering, then
    the registry's ignore map (``programbench.submission`` + ``compile_leaderboard.py``)."""
    from programbench.submission import (
        benchmark_instances,
        score_from_tests,
        test_results_map,
    )

    instance = benchmark_instances()[instance_id]
    tests = test_results_map(eval_json, instance)
    ignore = set(ignores.get(instance_id, []))
    kept = {name: passed for name, passed in tests.items() if name not in ignore}
    raw = json.loads(eval_json.read_text())
    entries = Counter(f"{t.get('branch')}/{t.get('name')}" for t in raw.get("test_results") or [])
    return {
        "score": score_from_tests(tests, ignore),
        "passed": sum(kept.values()),
        "scored_tests": len(kept),
        "duplicate_result_entries": sum(n - 1 for n in entries.values() if n > 1),
        "rerun_plugin_pinned": "pytest-rerunfailures==" in eval_json.read_text(),
        "error_code": raw.get("error_code"),
        "error_details": str(raw.get("error_details") or "")[:2000] or None,
        "test_branch_errors": raw.get("test_branch_errors") or {},
        "executable_hash": raw.get("executable_hash"),
        "tests": kept,
    }


def targets(results: Path) -> list[tuple[str, Path]]:
    """Every archive worth scoring: each attempt's final workspace and per-round snapshots."""
    items = []
    for attempt in sorted((results / "attempts").glob("*")):
        if not attempt.is_dir() or "." in attempt.name:  # skip NN-arm.discarded-<ts>
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
        if link.exists() and not os.path.samefile(archive, link):
            # The attempt was re-run: drop the stale link and its old results.
            link.unlink()
            for stale in instance_dir.glob("*.eval.json"):
                stale.unlink()
        if not link.exists():
            os.link(archive, link)
        if force or not (instance_dir / f"{exp.instance_id}.eval.json").exists():
            pending.append(run_dir)
    cfg = exp.raw["eval"]
    run_record: dict[str, Any] = {"pending": len(pending)}
    if pending:
        log(f"evaluating {len(pending)} archive(s) with programbench eval")
        args = [
            sys.executable, "-m", "bench.pbeval", "eval", *map(str, pending),
            "--workers", str(cfg["workers"]), "--docker-cpus", str(cfg["docker_cpus"]),
            "--image-tag", eval_image_tag(exp),
        ]
        if force:
            args.append("--force")
        env = {k: v for k, v in os.environ.items() if k != SECRET_ENV}
        env["PROGRAMBENCH_HF_REVISION"] = pins()["programbench_tests"]["revision"]
        with (results / "programbench-eval.log").open("ab") as out:
            try:
                result = subprocess.run(args, stdout=out, stderr=subprocess.STDOUT, env=env, check=False, timeout=int(cfg.get("timeout_seconds", 14400)))
                run_record["returncode"] = result.returncode
            except subprocess.TimeoutExpired:
                run_record["returncode"] = "timeout"
        log(f"programbench eval finished: {run_record['returncode']}")
    ignores = leaderboard_ignores(cache)
    scores: dict[str, Any] = {}
    for label, _ in targets(results):
        eval_json = evals / label / exp.instance_id / f"{exp.instance_id}.eval.json"
        scores[label] = score_eval(eval_json, exp.instance_id, ignores) if eval_json.exists() else {"score": None, "error_code": "not_evaluated"}
    write_json(results / "scores.json", {k: {kk: vv for kk, vv in v.items() if kk != "tests"} for k, v in scores.items()})
    write_json(results / "scores-per-test.json", {k: v.get("tests", {}) for k, v in scores.items()})
    history = results / "eval-runs.json"
    runs = read_json(history) if history.exists() else []
    write_json(history, [*runs, run_record])
    return scores


def load_scores(results: Path) -> dict[str, Any]:
    path = results / "scores.json"
    return read_json(path) if path.exists() else {}
