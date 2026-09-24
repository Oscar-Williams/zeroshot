"""One attempt: a fresh task container, one Zeroshot run, and every artifact needed to score it."""

from __future__ import annotations

import json
import queue
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from . import graphs, ledger
from .config import AttemptSpec, Experiment, prompt
from .util import SECRET_ENV, docker, docker_to_file, iter_json_objects, log, read_json, write_json

EVENT_KINDS = {"run_started", "node_started", "node_completed", "token_usage_observed", "terminal", "safe_log"}
RUN_DIR = "/opt/zeroshot-bench/run"
PLACEHOLDER_ORIGIN = "https://github.com/zeroshot-bench/local-workspace.git"
# The reference binary stays out of every archive; the evaluator deletes ./executable anyway.
WORKSPACE_TAR = ["tar", "--exclude=./executable", "-C", "/workspace", "-czf", "-", "."]
TRAJECTORY_TAR = [
    "tar", "--ignore-failed-read",
    "--exclude=*.bootstrap.json", "--exclude=.codex/auth.json", "--exclude=.codex/config.toml",
    "-C", "/home/agent", "-czf", "-", ".codex", ".local/state/zeroshot",
]
FINISH_GRACE_SECONDS = 900


def run_files(exp: Experiment, arm: str) -> dict[str, Any]:
    builder, checker = prompt("builder"), prompt("checker")
    limits = exp.limits
    if arm == "loop":
        graph = graphs.loop_graph(builder, checker, exp.max_iterations, limits["build_timeout_ms"], limits["check_timeout_ms"])
    else:
        graph = graphs.single_graph(builder, limits["build_timeout_ms"])
    return {
        "graph.json": graph,
        "runtime.json": graphs.runtime_plan(arm, exp.model, exp.effort),
        "input.json": {"task": prompt("task")},
    }


def event_of(obj: Any) -> dict[str, Any] | None:
    """Find the durable event inside one ``zeroshot watch`` NDJSON record."""
    if isinstance(obj, dict) and obj.get("kind") in EVENT_KINDS:
        return obj
    for candidate in iter_json_objects(obj):
        if candidate.get("kind") in EVENT_KINDS:
            return candidate
    return None


def completion_of(event: dict[str, Any]) -> tuple[str, int, dict[str, Any]] | None:
    completion = event.get("completion") or {}
    reference = completion.get("reference") or {}
    if "node" not in reference:
        return None
    return reference["node"], int(reference.get("nodeInstance") or 0), completion.get("outcome") or {}


class Attempt:
    def __init__(self, exp: Experiment, spec: AttemptSpec, results: Path, image: str, proxy_url: str, network: str, keep: bool = False):
        self.exp, self.spec, self.image, self.proxy_url, self.network, self.keep = exp, spec, image, proxy_url, network, keep
        self.dir = results / "attempts" / spec.label
        self.name = f"zsbench-{exp.id}-{spec.label}"
        self.meta: dict[str, Any] = {}
        self._stop_requested = threading.Event()

    # -- lifecycle -------------------------------------------------------------------------------

    def done(self) -> bool:
        path = self.dir / "attempt.json"
        return path.exists() and read_json(path).get("state") == "complete"

    def run(self) -> dict[str, Any]:
        if self.done():
            log(f"[{self.spec.label}] already complete; skipping")
            return read_json(self.dir / "attempt.json")
        if self.dir.exists():
            shutil.move(str(self.dir), str(self.dir.with_name(f"{self.dir.name}.incomplete-{int(time.time())}")))
        self.dir.mkdir(parents=True)
        self.meta = {"label": self.spec.label, "arm": self.spec.arm, "index": self.spec.index, "container": self.name, "image": self.image, "state": "running", "started_at": time.time()}
        write_json(self.dir / "attempt.json", self.meta)
        try:
            self._start_container()
            run_id = self._submit()
            self._follow(run_id)
            self._finish(run_id)
            self.meta["state"] = "complete"
        except Exception as error:  # keep whatever the run produced; the error is part of the record
            self.meta["state"] = "error"
            self.meta["error"] = f"{type(error).__name__}: {error}"
            log(f"[{self.spec.label}] ERROR {self.meta['error']}")
            self._salvage()
        finally:
            self.meta["ended_at"] = time.time()
            self.meta["wall_seconds"] = round(self.meta["ended_at"] - self.meta["started_at"], 1)
            write_json(self.dir / "attempt.json", self.meta)
            if not self.keep:
                docker("rm", "-f", self.name, check=False)
        return self.meta

    def request_stop(self) -> None:
        self._stop_requested.set()

    # -- steps -----------------------------------------------------------------------------------

    def _start_container(self) -> None:
        docker("rm", "-f", self.name, check=False)
        resources = self.exp.resources
        proxy_env = []
        for name in ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy"):
            proxy_env += ["-e", f"{name}={self.proxy_url}"]
        for name in ("NO_PROXY", "no_proxy"):
            proxy_env += ["-e", f"{name}=localhost,127.0.0.1,::1"]
        docker(
            "run", "-d", "--name", self.name, "--hostname", "workspace", "--init",
            "--network", self.network, "--user", "agent", "--workdir", "/workspace",
            "--cpus", str(resources["cpus"]), "--memory", resources["memory"], "--memory-swap", resources["memory"],
            "--cap-drop", "SYS_PTRACE", "--label", "zsbench=1", "--label", f"zsbench.experiment={self.exp.id}",
            *proxy_env, self.image, "sleep", "infinity",
        )
        files = run_files(self.exp, self.spec.arm)
        run_dir = self.dir / "run"
        for name, value in files.items():
            write_json(run_dir / name, value)
        docker("cp", f"{run_dir}/.", f"{self.name}:{RUN_DIR}")
        docker("exec", "-u", "root", self.name, "chmod", "-R", "a+rX", RUN_DIR)
        origin = docker("exec", self.name, "git", "-C", "/workspace", "remote", "get-url", "origin").strip()
        if origin != PLACEHOLDER_ORIGIN:
            raise RuntimeError(f"unexpected workspace origin {origin!r}")

    def _submit(self) -> str:
        key = f"{self.exp.id}-{self.spec.label}"
        receipt_text = docker(
            "exec", "-u", "agent", "-w", "/workspace", "-e", SECRET_ENV, self.name,
            "zeroshot", "run", "--title", f"{self.exp.id} {self.spec.label}",
            "--graph", f"{RUN_DIR}/graph.json", "--input", f"{RUN_DIR}/input.json",
            "--runtime-config", f"{RUN_DIR}/runtime.json", "--submission-key", key, "--detach",
            timeout=600,
        )
        (self.dir / "receipt.json").write_text(receipt_text)
        run_id = None
        try:
            parsed = json.loads(receipt_text)
            for obj in iter_json_objects(parsed):
                run_id = run_id or obj.get("runId") or obj.get("run_id")
        except json.JSONDecodeError:
            pass
        if not run_id:
            runs = json.loads(docker("exec", "-u", "agent", self.name, "zeroshot", "list"))["runs"]
            if len(runs) != 1:
                raise RuntimeError(f"cannot identify the run id from the receipt: {receipt_text[:500]!r}")
            run_id = runs[0]["runId"]
        self.meta["run_id"] = run_id
        self.meta["submitted_at"] = time.time()
        write_json(self.dir / "attempt.json", self.meta)
        log(f"[{self.spec.label}] submitted run {run_id}")
        return run_id

    def _follow(self, run_id: str) -> None:
        """Follow ``zeroshot watch`` status projections until the run finishes.

        Each projection lists the executions that are active right now. When a ``build`` or
        ``check`` execution leaves that set, its node has completed, and the workspace is
        snapshotted before the next node can change it. Verdicts come from the ledger afterwards.
        """
        deadline = self.meta["submitted_at"] + self.exp.limits["attempt_seconds"]
        forced = False
        finished = False
        counts = {"build": 0, "check": 0}
        active: dict[str, str] = {}
        cursor: str | None = None
        projections = self.dir / "watch.ndjson"
        while not finished:
            lines: queue.Queue = queue.Queue()
            args = ["docker", "exec", "-u", "agent", self.name, "zeroshot", "watch", run_id, *(["--after", cursor] if cursor else [])]
            proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

            def pump(proc: subprocess.Popen = proc, lines: queue.Queue = lines) -> None:
                for raw in proc.stdout:
                    lines.put(raw)
                lines.put(None)

            threading.Thread(target=pump, daemon=True).start()
            with projections.open("ab") as out:
                while True:
                    now = time.time()
                    if not forced and (now > deadline or self._stop_requested.is_set()):
                        reason = "stop requested" if self._stop_requested.is_set() else "attempt time limit reached"
                        log(f"[{self.spec.label}] {reason}; force-stopping run")
                        docker("exec", "-u", "agent", self.name, "zeroshot", "force-stop", run_id, check=False, timeout=300)
                        self.meta["force_stopped"] = reason
                        forced = True
                        deadline = now + FINISH_GRACE_SECONDS
                    elif forced and now > deadline:
                        proc.kill()
                        raise RuntimeError("run did not finish after force-stop")
                    try:
                        raw = lines.get(timeout=5)
                    except queue.Empty:
                        continue
                    if raw is None:
                        break
                    out.write(raw)
                    out.flush()
                    try:
                        record = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    cursor = record.get("cursor") or cursor
                    status = record.get("status") or {}
                    now_active = {e.get("execution"): e.get("node") for e in status.get("activeExecutions") or []}
                    for execution, node in active.items():
                        if execution not in now_active and node in counts:
                            counts[node] += 1
                            self._snapshot(f"{node}-{counts[node]}")
                            log(f"[{self.spec.label}] {node} {counts[node]} completed")
                    active = now_active
                    if status.get("phase") == "finished":
                        finished = True
                        self.meta["terminal"] = status.get("terminalResult")
                        self.meta["token_usage"] = (status.get("metadata") or {}).get("tokenUsage")
            proc.wait(timeout=60)
            if not finished:
                time.sleep(5)  # the watch stream ended early; reattach after the last cursor
        self.meta["snapshots"] = dict(counts)
        log(f"[{self.spec.label}] finished: {json.dumps(self.meta.get('terminal'))[:200]}")

    def _snapshot(self, name: str) -> None:
        docker_to_file(["exec", "-u", "root", self.name, *WORKSPACE_TAR], self.dir / "snapshots" / f"{name}.tar.gz", timeout=1800)

    def _finish(self, run_id: str) -> None:
        (self.dir / "status.json").write_text(docker("exec", "-u", "agent", self.name, "zeroshot", "status", run_id, check=False))
        self._collect()
        recorded = ledger.events(self.dir / "trajectories.tar.gz")
        if recorded is None:
            raise RuntimeError("the run ledger is missing from the trajectories archive")
        summary = ledger.rounds(recorded)
        self.meta.update({
            "builds": len(summary["builds"]),
            "checks": len(summary["checks"]),
            "build_outcomes": summary["builds"],
            "verdicts": summary["checks"],
        })
        for i, check in enumerate(summary["checks"], 1):
            log(f"[{self.spec.label}] check {i}: {check.get('verdict') or check.get('status')}")

    def _collect(self) -> None:
        docker_to_file(["exec", "-u", "root", self.name, *WORKSPACE_TAR], self.dir / "submission.tar.gz", timeout=1800)
        docker_to_file(["exec", "-u", "root", self.name, *TRAJECTORY_TAR], self.dir / "trajectories.tar.gz", timeout=1800)
        git_log = docker("exec", "-u", "agent", self.name, "git", "-C", "/workspace", "log", "--oneline", "-50", check=False)
        (self.dir / "workspace-git-log.txt").write_text(git_log)

    def _salvage(self) -> None:
        try:
            if docker("ps", "-q", "--filter", f"name=^{self.name}$").strip():
                if self.meta.get("run_id"):
                    docker("exec", "-u", "agent", self.name, "zeroshot", "force-stop", self.meta["run_id"], check=False, timeout=300)
                self._collect()
                self.meta["salvaged"] = True
        except Exception as error:
            self.meta["salvage_error"] = f"{type(error).__name__}: {error}"
