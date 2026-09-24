"""Command line: ``python -m bench <command> [experiment.json]``.

Commands:
  plan        render graphs, runtime plans and the attempt order without running anything
  check-key   confirm OPENAI_API_KEY can reach the experiment's model (prints only a status)
  smoke       isolation checks, a diagnostic run, the full pipeline with short limits, and a
              scoring-fidelity check (defaults to experiments/smoke.json)
  run         run every attempt, then evaluate and report
  eval        (re)evaluate and score existing attempts
  report      rebuild summary.json and summary.md
  cleanup     remove this experiment's containers and network
"""

from __future__ import annotations

import argparse
import os
import platform
import signal
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import ROOT, config, evaluate, images, report, smoke
from .attempt import Attempt, run_files
from .util import SECRET_ENV, docker, load_secret_file, log, require_secret, set_log_file, write_json

RESULTS = Path(os.environ.get("ZSBENCH_RESULTS", "/results"))
CACHE = Path(os.environ.get("ZSBENCH_CACHE", "/cache"))
SECRET_FILE = os.environ.get("ZSBENCH_SECRET_FILE", "/run/secrets/openai.env")


def _results(exp: config.Experiment) -> Path:
    path = RESULTS / exp.id
    path.mkdir(parents=True, exist_ok=True)
    set_log_file(path / "run.log")
    return path


def cmd_plan(exp: config.Experiment) -> None:
    results = _results(exp)
    for arm in sorted(set(exp.raw["order"])):
        for name, value in run_files(exp, arm).items():
            write_json(results / "plan" / arm / name, value)
    log(f"experiment {exp.id} digest {exp.digest()[:16]}")
    for spec in exp.attempts():
        log(f"  attempt {spec.label}")
    log(f"rendered graphs, runtime plans and input under {results / 'plan'}")


def cmd_check_key(exp: config.Experiment) -> None:
    require_secret()
    request = urllib.request.Request(f"https://api.openai.com/v1/models/{exp.model}", headers={"Authorization": f"Bearer {os.environ[SECRET_ENV]}"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            print(f"OK: key accepted and {exp.model} is available (HTTP {response.status})")
    except urllib.error.HTTPError as error:
        sys.exit(f"FAILED: HTTP {error.code} for {exp.model} (401 = key rejected, 404 = model unavailable to this key)")


def _manifest(exp: config.Experiment, results: Path, agent_info: dict, proxy_image: str) -> None:
    write_json(results / "manifest.json", {
        "experiment": exp.raw,
        "experiment_digest": exp.digest(),
        "pins": config.pins(),
        "agent_image": agent_info["tag"],
        "task_image": agent_info["task_image"],
        "task_image_id": docker("image", "inspect", exp.task_image, "--format", "{{.Id}}").strip(),
        "codex_config": agent_info["codex_config"],
        "proxy_image": proxy_image,
        "prompts": {name: config.prompt(name) for name in ("builder", "checker", "task")},
        "runner": {"vcs_ref": os.environ.get("ZSBENCH_VCS_REF", "unknown"), "python": platform.python_version()},
        "host": {"docker": docker("version", "--format", "{{.Server.Version}}").strip(), "cpus": docker("info", "--format", "{{.NCPU}}").strip(), "memory_bytes": docker("info", "--format", "{{.MemTotal}}").strip()},
        "created_at": time.time(),
    })


def _prepare(exp: config.Experiment, results: Path) -> tuple[str, images.Network]:
    proxy_image = images.build_proxy(CACHE)
    agent_image, info = images.build_agent(exp, CACHE)
    _manifest(exp, results, info, proxy_image)
    network = images.Network(exp, proxy_image)
    network.up()
    return agent_image, network


def _run_attempts(exp: config.Experiment, results: Path, agent_image: str, network: images.Network, keep: bool) -> None:
    attempts = [Attempt(exp, spec, results, agent_image, network.proxy_url, network.name, keep) for spec in exp.attempts()]
    stop = lambda *_: [a.request_stop() for a in attempts]
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    with ThreadPoolExecutor(max_workers=exp.resources["concurrency"]) as pool:
        futures = {pool.submit(a.run): a for a in attempts}  # submitted in the pre-registered order
        for future in as_completed(futures):
            meta = future.result()
            log(f"[{meta['label']}] {meta['state']} after {meta.get('wall_seconds', 0) / 60:.1f} min")


def cmd_run(exp: config.Experiment, keep: bool, skip_eval: bool) -> dict:
    require_secret()
    results = _results(exp)
    agent_image, network = _prepare(exp, results)
    try:
        _run_attempts(exp, results, agent_image, network, keep)
    finally:
        proxy_log = network.proxy_log()
        (results / "proxy.log").write_text(proxy_log)
        network.down()
    if not skip_eval:
        evaluate.evaluate(exp, results, CACHE)
    summary = report.build(exp, results, proxy_log)
    log(f"summary written to {results / 'summary.md'}")
    return summary


def cmd_eval(exp: config.Experiment, force: bool) -> None:
    results = _results(exp)
    evaluate.evaluate(exp, results, CACHE, force=force)
    cmd_report(exp)


def cmd_report(exp: config.Experiment) -> None:
    results = _results(exp)
    proxy_log = (results / "proxy.log").read_text() if (results / "proxy.log").exists() else None
    summary = report.build(exp, results, proxy_log)
    print((results / "summary.md").read_text())
    if summary["secrets"]["literal_key_hits"]:
        sys.exit("SECRET LEAK: the API key appears in results; see summary.json")


def cmd_smoke(exp: config.Experiment) -> None:
    require_secret()
    results = _results(exp)
    agent_image, network = _prepare(exp, results)
    s = smoke.Smoke(exp, results, CACHE, agent_image, network)
    try:
        s.isolation()
        s.diagnostic_run()
    finally:
        network.down()
    summary = cmd_run(exp, keep=False, skip_eval=False)
    smoke.pipeline_checks(s, summary)
    s.scoring_fidelity()
    outcome = s.summary()
    failed = [name for name, c in outcome["checks"].items() if not c["ok"]]
    log(f"smoke {'PASSED' if not failed else 'FAILED'}: {len(outcome['checks']) - len(failed)}/{len(outcome['checks'])} checks passed" + (f"; failed: {failed}" if failed else ""))
    if failed:
        sys.exit(1)


def cmd_cleanup(exp: config.Experiment) -> None:
    for container in docker("ps", "-aq", "--filter", f"label=zsbench.experiment={exp.id}").split():
        docker("rm", "-f", container, check=False)
    images.Network(exp, "").down()
    log(f"removed containers and network for {exp.id}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="bench", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["plan", "check-key", "smoke", "run", "eval", "report", "cleanup"])
    parser.add_argument("experiment", nargs="?", help="experiment JSON (default: experiments/smoke.json for smoke)")
    parser.add_argument("--keep-containers", action="store_true", help="leave attempt containers for inspection")
    parser.add_argument("--skip-eval", action="store_true", help="run attempts without evaluating")
    parser.add_argument("--force", action="store_true", help="re-evaluate archives that already have results")
    args = parser.parse_args()
    load_secret_file(SECRET_FILE)
    default = ROOT / "experiments" / ("smoke.json" if args.command == "smoke" else "luna-xhigh-svgbob.json")
    exp = config.load(args.experiment or default)
    if args.command == "plan":
        cmd_plan(exp)
    elif args.command == "check-key":
        cmd_check_key(exp)
    elif args.command == "smoke":
        cmd_smoke(exp)
    elif args.command == "run":
        cmd_run(exp, args.keep_containers, args.skip_eval)
    elif args.command == "eval":
        cmd_eval(exp, args.force)
    elif args.command == "report":
        cmd_report(exp)
    elif args.command == "cleanup":
        cmd_cleanup(exp)


if __name__ == "__main__":
    main()
