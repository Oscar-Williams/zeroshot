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
from .util import SECRET_ENV, download, log, read_json, sha256_file, write_json

# Eval error codes that are outcomes of the submission itself: its tree could not be committed,
# its compile.sh failed or timed out, or it produced no usable ./executable. The leaderboard scores them 0 on every test, and so do we.
# Any other error code, and any test-branch error, is an evaluation infrastructure failure.
SUBMISSION_OUTCOMES = frozenset({"compile_failed", "copy_executable_failed", "hash_executable_failed", "no_executable_hash", "seed_git_failed"})


def infrastructure_error(score: dict[str, Any]) -> str | None:
    """Why an archive's evaluation cannot be trusted, or None."""
    if score.get("test_branch_errors"):
        return f"test branch errors {sorted(score['test_branch_errors'])}"
    code = score.get("error_code")
    if code:
        return None if code in SUBMISSION_OUTCOMES else code
    if score.get("score") is not None and score.get("rerun_plugin_pinned") is not True:
        return "pinned pytest-rerunfailures was not active"
    return None


def archive_id(path: Path) -> str:
    """Content hash of one archive: a re-run attempt writes different bytes under the same label,
    and unlike file metadata the hash survives copying the results elsewhere."""
    return sha256_file(path)


def rerun_plugin_active(raw: dict[str, Any]) -> bool | None:
    """Whether the pinned plugin was installed (exit 0) and loaded by pytest; None if no tests ran."""
    log_entries = [e for e in raw.get("log") or [] if isinstance(e, dict)]
    installs = [e for e in log_entries if e.get("step") == "install_rerunfailures"]
    runs = [e for e in log_entries if e.get("step") == "run_tests"]
    if not runs:
        return None
    installed = any(e.get("returncode") == 0 and "pytest-rerunfailures==16.4" in str(e.get("command")) for e in installs)
    return installed and all("rerunfailures-16.4" in str(e.get("output")) for e in runs)


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
        "rerun_plugin_pinned": rerun_plugin_active(raw),
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
            if archive_id(link) != archive_id(archive):  # the attempt was re-run: drop its old results
                for stale in instance_dir.glob("*.eval.json"):
                    stale.unlink()
            link.unlink()  # re-linked below; a copied results tree loses its hard links
        if not link.exists():
            os.link(archive, link)
        if force or not (instance_dir / f"{exp.instance_id}.eval.json").exists():
            pending.append(run_dir)
    run_record: dict[str, Any] = {"pending": len(pending)}
    if pending:
        run_record["returncode"] = _programbench_eval(exp, results, pending, force)
    ignores = leaderboard_ignores(cache)
    scores = _scores(exp, results, ignores)
    # Evaluation is deterministic for a given archive, so an infrastructure failure is retried
    # once; a failure that persists makes the archive's run ineligible (report._eligibility).
    retry = [evals / label for label, score in scores.items() if score.get("error_code") == "not_evaluated" or (score.get("score") is not None and infrastructure_error(score))]
    if retry:
        run_record["retried"] = {run_dir.name: infrastructure_error(scores[run_dir.name]) or scores[run_dir.name].get("error_code") for run_dir in retry}
        run_record["retry_returncode"] = _programbench_eval(exp, results, retry, force=True)
        scores = _scores(exp, results, ignores)
    write_json(results / "scores.json", {k: {kk: vv for kk, vv in v.items() if kk != "tests"} for k, v in scores.items()})
    write_json(results / "scores-per-test.json", {k: v.get("tests", {}) for k, v in scores.items()})
    history = results / "eval-runs.json"
    runs = read_json(history) if history.exists() else []
    write_json(history, [*runs, run_record])
    return scores


def _programbench_eval(exp: Experiment, results: Path, run_dirs: list[Path], force: bool) -> int | str:
    cfg = exp.raw["eval"]
    log(f"evaluating {len(run_dirs)} archive(s) with programbench eval")
    args = [
        sys.executable, "-m", "bench.pbeval", "eval", *map(str, run_dirs),
        "--workers", str(cfg["workers"]), "--docker-cpus", str(cfg["docker_cpus"]),
        "--image-tag", eval_image_tag(exp),
    ]
    if force:
        args.append("--force")
    env = {k: v for k, v in os.environ.items() if k != SECRET_ENV}
    env["PROGRAMBENCH_HF_REVISION"] = pins()["programbench_tests"]["revision"]
    with (results / "programbench-eval.log").open("ab") as out:
        try:
            returncode: int | str = subprocess.run(args, stdout=out, stderr=subprocess.STDOUT, env=env, check=False, timeout=int(cfg.get("timeout_seconds", 14400))).returncode
        except subprocess.TimeoutExpired:
            returncode = "timeout"
    log(f"programbench eval finished: {returncode}")
    return returncode


def _scores(exp: Experiment, results: Path, ignores: dict[str, list[str]]) -> dict[str, Any]:
    scores: dict[str, Any] = {}
    for label, archive in targets(results):
        eval_json = results / "evals" / label / exp.instance_id / f"{exp.instance_id}.eval.json"
        try:
            scores[label] = score_eval(eval_json, exp.instance_id, ignores) if eval_json.exists() else {"score": None, "error_code": "not_evaluated"}
        except ValueError as error:  # e.g. eval.json cut short when the evaluation was killed
            scores[label] = {"score": None, "error_code": "not_evaluated", "error_details": f"unreadable eval.json: {str(error)[:300]}"}
        scores[label]["archive_id"] = archive_id(archive)
    return scores


def load_scores(results: Path) -> dict[str, Any]:
    path = results / "scores.json"
    return read_json(path) if path.exists() else {}
