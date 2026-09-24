"""One attempt: a fresh task container on its own isolated network, one Zeroshot run, and every
artifact needed to score and audit it."""

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
from .images import Network
from .util import (
    SECRET_ENV,
    docker,
    docker_to_file,
    iter_json_objects,
    log,
    read_json,
    write_json,
)

RUN_DIR = "/opt/zeroshot-bench/run"
PLACEHOLDER_ORIGIN = "https://github.com/zeroshot-bench/local-workspace.git"
# Workspace archives are made as the agent user, like the upstream baseline: files the agent
# cannot read (the execute-only reference, wherever it was moved) are left out. Files that change
# while tar reads them are recorded as warnings, not failures.
WORKSPACE_TAR = [
    "tar", "--ignore-failed-read", "--warning=no-file-changed", "--warning=no-file-removed",
    "--exclude=./executable", "-C", "/workspace", "-czf", "-", ".",
]
TRAJECTORY_TAR = [
    "tar", "--ignore-failed-read", "--warning=no-file-changed", "--warning=no-file-removed",
    "--exclude=*.bootstrap.json", "--exclude=.codex/auth.json",
    "-C", "/home/agent", "-czf", "-", ".codex", ".local/state/zeroshot",
]
TAR_OK = (0, 1)  # GNU tar: 1 = some files differ/changed while reading; the archive is valid
# Content fingerprint of the harness files a later node would execute with the API key in its
# environment, and of Codex's system configuration layers (/etc/codex, absent in the image). The
# image's sudo rules allow root through package-manager hooks, so the harness is fingerprinted at
# the start, after every node and at the end, and any change is reported.
HARNESS_FINGERPRINT = (
    "cd / && find usr/local/bin/zeroshot usr/local/bin/codex opt/codex etc/codex \\( -type f -o -type l \\) 2>/dev/null | LC_ALL=C sort | "
    "while read -r f; do if [ -L \"$f\" ]; then echo \"link $f $(readlink \"$f\")\"; "
    "else echo \"file $f $(stat -c %a:%u \"$f\") $(sha256sum < \"$f\" | cut -c1-64)\"; fi; done | sha256sum | cut -c1-64"
)
# Files in the Codex home (shared by every node) that a later Codex session would load or run:
# the config, user AGENTS.md, skills (Codex manages skills/.system itself), prompts, rules, hooks,
# plugins and profile configs. Recorded after every node, since a node could change them for the
# next one and restore them before the end; only config.toml exists at the start.
CODEX_HOME_SURFACES = (
    "cd /home/agent/.codex 2>/dev/null || exit 0; "
    "{ find . -maxdepth 1 -type f \\( -name 'AGENTS*.md' -o -name '*config.toml' -o -name 'hooks*' \\); "
    "find skills prompts rules hooks plugins -path skills/.system -prune -o -type f -print; } 2>/dev/null | LC_ALL=C sort | "
    "while read -r f; do echo \"$f $(sha256sum < \"$f\" | cut -c1-16)\"; done"
)
FINISH_GRACE_SECONDS = 900
NEUTRAL_TITLE = "programbench attempt"


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


def snapshot_label(node: str, builds_completed: int, existing: set[str]) -> str:
    """Name snapshots by round: ``build-k`` after the k-th build and ``check-k`` for the check that
    follows it. A retried check in the same round becomes ``check-k.2``."""
    base = f"{node}-{builds_completed}"
    label, n = base, 1
    while label in existing:
        n += 1
        label = f"{base}.{n}"
    return label


class Attempt:
    def __init__(self, exp: Experiment, spec: AttemptSpec, results: Path, image: str, proxy_image: str, provenance: dict[str, Any], keep: bool = False):
        self.exp, self.spec, self.image, self.keep = exp, spec, image, keep
        self.dir = results / "attempts" / spec.label
        self.name = f"zsbench-{exp.id}-{spec.label}"
        self.network = Network(f"{self.name}-net", proxy_image, (f"zsbench.experiment={exp.id}",))
        self.provenance = provenance
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
        if self._stop_requested.is_set():
            return {"label": self.spec.label, "arm": self.spec.arm, "state": "skipped", "wall_seconds": 0}
        if self.dir.exists():
            shutil.move(str(self.dir), str(self.dir.with_name(f"{self.dir.name}.discarded-{int(time.time())}")))
        self.dir.mkdir(parents=True)
        self.meta = {
            "label": self.spec.label, "arm": self.spec.arm, "index": self.spec.index, "container": self.name,
            "image": self.image, "state": "running", "started_at": time.time(), "provenance": self.provenance,
            "snapshots": {}, "snapshot_warnings": {},
        }
        self._save()
        try:
            self.network.up()
            self._start_container()
            run_id = self._submit()
            self._follow(run_id)
            self._finish(run_id)
            self.meta["state"] = "stopped" if self.meta.get("force_stopped") == "stop requested" else "complete"
        except Exception as error:  # keep whatever the run produced; the error is part of the record
            self.meta["state"] = "error"
            self.meta["error"] = f"{type(error).__name__}: {error}"
            log(f"[{self.spec.label}] ERROR {self.meta['error']}")
            self._salvage()
        finally:
            self.meta["ended_at"] = time.time()
            self.meta["wall_seconds"] = round(self.meta["ended_at"] - self.meta["started_at"], 1)
            try:
                (self.dir / "proxy.log").write_text(self.network.proxy_log())
            except Exception as error:
                self.meta["proxy_log_error"] = str(error)
            self._save()
            if not self.keep:
                docker("rm", "-f", self.name, check=False)
                self.network.down()
        return self.meta

    def request_stop(self) -> None:
        self._stop_requested.set()

    def _save(self) -> None:
        write_json(self.dir / "attempt.json", self.meta)

    # -- steps -----------------------------------------------------------------------------------

    def _start_container(self) -> None:
        docker("rm", "-f", self.name, check=False)
        resources = self.exp.resources
        proxy_env = []
        for name in ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy"):
            proxy_env += ["-e", f"{name}={self.network.proxy_url}"]
        for name in ("NO_PROXY", "no_proxy"):
            proxy_env += ["-e", f"{name}=localhost,127.0.0.1,::1"]
        docker(
            "run", "-d", "--name", self.name, "--hostname", "workspace", "--init",
            "--network", self.network.name, "--user", "agent", "--workdir", "/workspace",
            "--cpus", str(resources["cpus"]), "--memory", resources["memory"], "--memory-swap", resources["memory"],
            "--pids-limit", str(resources.get("pids_limit", 8192)),
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
        size_and_hash = docker("exec", "-u", "root", self.name, "sh", "-c", "stat -c %s /workspace/executable && sha256sum /workspace/executable | cut -d' ' -f1").split()
        if len(size_and_hash) != 2:
            raise RuntimeError("reference executable missing at start")
        self.meta["reference_size"], self.meta["reference_sha256"] = int(size_and_hash[0]), size_and_hash[1]
        self.meta["harness_fingerprint"] = self._harness_fingerprint()
        self.meta["codex_home_surfaces"] = self._codex_home_surfaces()

    def _submit(self) -> str:
        receipt_text = docker(
            "exec", "-u", "agent", "-w", "/workspace", "-e", SECRET_ENV, self.name,
            "zeroshot", "run", "--title", NEUTRAL_TITLE,
            "--graph", f"{RUN_DIR}/graph.json", "--input", f"{RUN_DIR}/input.json",
            "--runtime-config", f"{RUN_DIR}/runtime.json",
            "--submission-key", f"{self.exp.id}-{self.spec.index:02d}", "--detach",
            timeout=600,
        )
        (self.dir / "receipt.json").write_text(receipt_text)
        # Zeroshot has admitted and stored the run; the submitted files would only tell a curious
        # builder which arm it is in.
        docker("exec", "-u", "root", self.name, "sh", "-c", f"rm -rf {RUN_DIR}/*")
        run_id = None
        try:
            for obj in iter_json_objects(json.loads(receipt_text)):
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
        self._save()
        log(f"[{self.spec.label}] submitted run {run_id}")
        return run_id

    def _codex_home_surfaces(self) -> list[str]:
        return docker("exec", "-u", "root", self.name, "sh", "-c", CODEX_HOME_SURFACES, timeout=300).splitlines()

    def _harness_fingerprint(self) -> str:
        return docker("exec", "-u", "root", self.name, "sh", "-c", HARNESS_FINGERPRINT, timeout=600).strip()

    def _check_harness(self) -> None:
        """Compare the harness fingerprint at the end, and after every node, with the start."""
        try:
            after = [self._harness_fingerprint(), *(s["harness_fingerprint"] for s in self.meta["snapshots"].values() if s.get("harness_fingerprint"))]
            self.meta["harness_unchanged"] = bool(self.meta.get("harness_fingerprint")) and all(f == self.meta["harness_fingerprint"] for f in after)
            self.meta["codex_home_surfaces_end"] = self._codex_home_surfaces()
        except Exception as error:
            self.meta["harness_check_error"] = f"{type(error).__name__}: {error}"

    def _container_running(self) -> bool:
        return docker("inspect", "-f", "{{.State.Running}}", self.name, check=False).strip() == "true"

    def _follow(self, run_id: str) -> None:
        """Follow ``zeroshot watch`` status projections until the run finishes.

        Each projection lists the executions active right now. When a ``build`` or ``check``
        execution leaves that set, its node has completed and the workspace is snapshotted. The
        next node may already be running by then; snapshot timing is recorded so any overlap can
        be checked against the session transcripts. Verdicts come from the ledger afterwards.
        """
        deadline = self.meta["submitted_at"] + self.exp.limits["attempt_seconds"]
        forced = False
        finished = False
        builds = 0
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
                        self.meta["force_stopped_at"] = now
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
                        if execution not in now_active and node in ("build", "check"):
                            if node == "build":
                                builds += 1
                            self._snapshot(snapshot_label(node, builds, set(self.meta["snapshots"])))
                    active = now_active
                    if status.get("phase") == "finished":
                        finished = True
                        self.meta["terminal"] = status.get("terminalResult")
                        self.meta["run_token_usage"] = (status.get("metadata") or {}).get("tokenUsage")
            proc.wait(timeout=60)
            if not finished:
                if not self._container_running():
                    raise RuntimeError("the attempt container stopped before the run finished")
                time.sleep(5)  # the watch stream ended early; reattach after the last cursor
        log(f"[{self.spec.label}] finished: {json.dumps(self.meta.get('terminal'))[:200]}")

    def _reference_locations(self) -> list[str]:
        """Paths (read as root) that hold the reference binary, found by size and hash."""
        size, digest = self.meta["reference_size"], self.meta["reference_sha256"]
        script = f"find /workspace /tmp /home/agent -xdev -type f -size {size}c -exec sha256sum {{}} + 2>/dev/null | grep '^{digest} ' | cut -c67-"
        return [p for p in docker("exec", "-u", "root", self.name, "sh", "-c", script, check=False).splitlines() if p]

    def _snapshot(self, label: str) -> None:
        started = time.time()
        record: dict[str, Any] = {"started_at": started}
        try:
            warnings = docker_to_file(["exec", "-u", "agent", self.name, *WORKSPACE_TAR], self.dir / "snapshots" / f"{label}.tar.gz", timeout=1800, ok_codes=TAR_OK)
            if warnings.strip():
                self.meta["snapshot_warnings"][label] = warnings
        except Exception as error:  # a snapshot must never cost the run
            record["error"] = f"{type(error).__name__}: {error}"
        for key, probe in (("reference_at", self._reference_locations), ("codex_home_surfaces", self._codex_home_surfaces), ("harness_fingerprint", self._harness_fingerprint)):
            try:
                record[key] = probe()
            except Exception as error:  # recorded; the archive itself is still valid
                record.setdefault("probe_errors", {})[key] = f"{type(error).__name__}: {error}"
        record["seconds"] = round(time.time() - started, 2)
        self.meta["snapshots"][label] = record
        self._save()
        log(f"[{self.spec.label}] snapshot {label} ({record['seconds']}s){' ERROR ' + record['error'] if 'error' in record else ''}")

    def _finish(self, run_id: str) -> None:
        (self.dir / "status.json").write_text(docker("exec", "-u", "agent", self.name, "zeroshot", "status", run_id, check=False))
        self._check_harness()
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
            "final_reference_at": self._reference_locations(),
        })
        for i, check in enumerate(summary["checks"], 1):
            log(f"[{self.spec.label}] check {i}: {check.get('verdict') or check.get('status')}")

    def _collect(self) -> None:
        warnings = docker_to_file(["exec", "-u", "agent", self.name, *WORKSPACE_TAR], self.dir / "submission.tar.gz", timeout=1800, ok_codes=TAR_OK)
        if warnings.strip():
            self.meta["submission_warnings"] = warnings
        docker_to_file(["exec", "-u", "root", self.name, *TRAJECTORY_TAR], self.dir / "trajectories.tar.gz", timeout=1800, ok_codes=TAR_OK)
        git_log = docker("exec", "-u", "agent", self.name, "git", "-C", "/workspace", "log", "--oneline", "-50", check=False)
        (self.dir / "workspace-git-log.txt").write_text(git_log)

    def _salvage(self) -> None:
        try:
            if self._container_running():
                if self.meta.get("run_id"):
                    docker("exec", "-u", "agent", self.name, "zeroshot", "force-stop", self.meta["run_id"], check=False, timeout=300)
                if self.meta.get("harness_fingerprint"):
                    self._check_harness()
                self._collect()
                self.meta["salvaged"] = True
        except Exception as error:
            self.meta["salvage_error"] = f"{type(error).__name__}: {error}"
