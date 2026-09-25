"""Smoke tests: isolation, environment, graph validation, a diagnostic run, the full pipeline with
short limits, and scoring fidelity against a published leaderboard submission."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import tarfile
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import audit
from .attempt import (
    CODEX_HOME_SURFACES,
    EXEC_NONDUMPABLE,
    HARNESS_FINGERPRINT,
    HARNESS_PATH,
    RUN_DIR,
    SUBMIT_PATH,
    TRAJECTORY_TAR,
    ZEROSHOT,
    archive,
    run_files,
)
from .config import UPSTREAM_REFERENCE, Experiment, pins
from .evaluate import eval_image_tag, leaderboard_ignores, score_eval
from .images import Network
from .util import SECRET_ENV, docker, download, log, run, write_json

DIAGNOSTIC_TASK = """This is an environment diagnostic, not a coding task. Run exactly the following shell script once, then stop. Do not modify it and do not run anything else.

```sh
{
  echo "USER=$(id -un)"
  echo "HOME=$HOME"
  echo "ENV_NAMES=$(env | cut -d= -f1 | sort | tr '\\n' ' ')"
  echo "KEY_VARS_VISIBLE=$(env | grep -c -E '^(OPENAI_API_KEY|CODEX_API_KEY)=')"
  echo "PROC_KEY_VISIBLE=$(grep -l -a -E '(OPENAI_API_KEY|CODEX_API_KEY)=' /proc/[0-9]*/environ 2>/dev/null | wc -l)"
  echo "PROXY_VARS_VISIBLE=$(env | grep -c -i -E '^(https?|all)_proxy=')"
  echo "TMPDIR=${TMPDIR:-unset}"
  echo "CARGO=$(cargo --version 2>&1 | head -1)"
  echo "RUSTC=$(rustc --version 2>&1 | head -1)"
  echo "GO=$(go version 2>&1 | head -1)"
  echo "PYTHON=$(python3 --version 2>&1)"
  echo "RG=$(command -v rg || echo missing)"
  if curl -sS -m 8 -o /dev/null https://example.com 2>/dev/null; then echo "EGRESS=open"; else echo "EGRESS=blocked"; fi
} > /workspace/.zsbench-diagnostic.txt 2>&1
```
"""


CANARY = "zsbench-canary-7f3a"
_TOOL_RECORDS = {"function_call", "function_call_output", "custom_tool_call", "custom_tool_call_output", "item_completed", "item_started"}


def _canary_outside_tools(transcript: bytes) -> bool:
    """The canary anywhere but in tool calls and their output (the model may list the workspace)."""
    for line in transcript.splitlines():
        if CANARY.encode() in line:
            try:
                kind = (json.loads(line).get("payload") or {}).get("type")
            except json.JSONDecodeError:
                return True
            if kind not in _TOOL_RECORDS:
                return True
    return False


class Smoke:
    def __init__(self, exp: Experiment, results: Path, cache: Path, agent_image: str, proxy_image: str):
        self.exp, self.results, self.cache, self.image, self.proxy_image = exp, results, cache, agent_image, proxy_image
        self.checks: dict[str, dict[str, Any]] = {}

    def check(self, name: str, fn: Callable[[], tuple[bool, Any]]) -> bool:
        try:
            ok, detail = fn()
        except Exception as error:
            ok, detail = False, f"{type(error).__name__}: {error}"
        self.checks[name] = {"ok": bool(ok), "detail": detail}
        log(f"smoke {'PASS' if ok else 'FAIL'} {name}: {str(detail)[:300]}")
        return bool(ok)

    def _probe(self, suffix: str) -> tuple[str, Network]:
        name = f"zsbench-{self.exp.id}-{suffix}"
        network = Network(f"{name}-net", self.proxy_image, (f"zsbench.experiment={self.exp.id}",))
        network.up()
        docker("rm", "-f", name, check=False)
        env = []
        for var in ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy"):
            env += ["-e", f"{var}={network.proxy_url}"]
        for var in ("NO_PROXY", "no_proxy"):
            env += ["-e", f"{var}=localhost,127.0.0.1,::1"]
        docker("run", "-d", "--name", name, "--hostname", "workspace", "--init", "--network", network.name, "--user", "agent", "--workdir", "/workspace", "--cap-drop", "SYS_PTRACE", "--label", "zsbench=1", "--label", f"zsbench.experiment={self.exp.id}", *env, self.image, "sleep", "infinity")
        return name, network

    def _sh(self, container: str, script: str, user: str = "agent", timeout: float = 120) -> tuple[int, str]:
        result = run(["docker", "exec", "-u", user, container, "bash", "-c", script], check=False, timeout=timeout)
        return result.returncode, (result.stdout + result.stderr).decode(errors="replace").strip()

    def isolation(self) -> None:
        c, network = self._probe("probe")
        try:
            self.check("runs_as_agent", lambda: (self._sh(c, "id -un")[1] == "agent", self._sh(c, "id")[1]))
            ref = shlex.quote(self.exp.reference_path)
            self.check("reference_binary_unreadable", lambda: (self._sh(c, f"test -r {ref}")[0] != 0, "execute-only for the agent"))
            self.check("reference_binary_runs", lambda: (self._sh(c, f"{ref} --version")[0] == 0, self._sh(c, f"{ref} --version")[1]))
            if self.exp.reference_path != UPSTREAM_REFERENCE:
                def reference_protected() -> tuple[bool, str]:
                    """Moved out of the workspace, the reference survives whatever the agent does."""
                    self._sh(c, f"rm -f {ref}; mv {ref} /tmp/zsbench-moved; printf x > {ref}; cp /bin/true {ref}; true")
                    still = self._sh(c, f"{ref} --version")
                    return still[0] == 0 and self._sh(c, "test -e /tmp/zsbench-moved")[0] != 0, f"after rm/mv/overwrite attempts: {still[1][:80]}"

                self.check("reference_protected", reference_protected)
                self.check("workspace_starts_clean", lambda: (not self._sh(c, "git -C /workspace status --porcelain")[1], self._sh(c, "git -C /workspace status --porcelain; git -C /workspace log --oneline")[1]))
            for fix in self.exp.doc_fixes:
                def doc_fix(fix: dict[str, str] = fix) -> tuple[bool, str]:
                    text = self._sh(c, f"cat {shlex.quote('/workspace/' + fix['file'])}")[1]
                    return fix["new"] in text and fix["old"] not in text, f"{fix['file']}: {fix['new'][:60]!r}"

                self.check(f"doc_fix_applied_{self.exp.doc_fixes.index(fix) + 1}", doc_fix)
            self.check("harness_binaries_unreadable", lambda: (
                self._sh(c, "test -r /usr/local/bin/zeroshot || test -r /opt/codex/bin/codex || test -r /opt/codex/bin/codex-code-mode-host")[0] != 0,
                "zeroshot and Codex executables are execute-only (their /proc entries are protected)"))

            def key_holder_nondumpable() -> tuple[bool, str]:
                """The way the runner starts the key-holding submitter keeps its environment
                unreadable to the agent: an execute-only probe started the same way, with a
                dummy variable, must refuse the agent's read of /proc/<pid>/environ."""
                self._sh(c, "cp /bin/sleep /usr/local/bin/zsbench-probe && chmod 0711 /usr/local/bin/zsbench-probe", user="root")
                docker("exec", "-d", "-u", "agent", "-e", "ZSBENCH_PROBE_VAR=1", c, *EXEC_NONDUMPABLE, "/usr/local/bin/zsbench-probe", "60")
                time.sleep(1)
                pid = self._sh(c, "pgrep -x zsbench-probe", user="root")[1].split()
                if not pid:
                    return False, "probe did not start"
                code, out = self._sh(c, f"grep -c ZSBENCH_PROBE_VAR /proc/{pid[0]}/environ")
                return code != 0 and "Permission denied" in out, out[:200]

            self.check("key_holder_environ_unreadable", key_holder_nondumpable)
            self.check("direct_egress_blocked", lambda: (
                self._sh(c, "curl -sS -m 8 --noproxy '*' -o /dev/null https://example.com")[0] != 0
                and self._sh(c, "curl -sS -m 8 --noproxy '*' -o /dev/null https://api.openai.com")[0] != 0,
                "no route out without the proxy"))
            self.check("dns_blocked", lambda: (self._sh(c, "getent hosts example.com")[0] != 0, self._sh(c, "getent hosts example.com")[1] or "no answer"))
            self.check("ipv6_blocked", lambda: (self._sh(c, "curl -6 -sS -m 8 --noproxy '*' -o /dev/null https://example.com")[0] != 0, "no IPv6 route"))
            self.check("proxy_refuses_other_hosts", lambda: (
                self._sh(c, "curl -sS -m 10 -o /dev/null https://example.com")[0] != 0
                and self._sh(c, "curl -sS -m 10 -o /dev/null https://github.com")[0] != 0
                and self._sh(c, "curl -sS -m 10 -o /dev/null https://pypi.org/simple/")[0] != 0,
                "example.com, github.com, pypi.org refused"))

            def model_api() -> tuple[bool, str]:
                code = self._sh(c, "curl -sS -m 20 -o /dev/null -w '%{http_code}' https://api.openai.com/v1/models")[1]
                return code == "401", f"unauthenticated GET /v1/models -> {code}"

            self.check("proxy_allows_model_api", model_api)
            self.check("tool_versions", lambda: (self._sh(c, f"{ZEROSHOT} --version && /usr/local/bin/codex --version")[0] == 0, self._sh(c, f"{ZEROSHOT} --version; /usr/local/bin/codex --version")[1]))

            def codex_config() -> tuple[bool, str]:
                config = self._sh(c, "cat ~/.codex/config.toml")[1]
                required = ('web_search = "disabled"', "shell_snapshot = false", "generate_memories = false", '"*PROXY*"', 'trust_level = "untrusted"')
                return all(s in config for s in required), "web search, shell snapshot and memories off; proxy vars hidden from tools; workspace untrusted"

            self.check("codex_config_hardened", codex_config)
            self.check("workspace_origin_placeholder", lambda: (self._sh(c, "git -C /workspace remote get-url origin")[1].endswith("zeroshot-bench/local-workspace.git"), "placeholder origin"))
            for arm in ("loop", "single"):
                files = run_files(self.exp, arm)
                staging = self.results / "smoke-validate" / arm
                for name, value in files.items():
                    write_json(staging / name, value)
                docker("cp", f"{staging}/.", f"{c}:{RUN_DIR}")
                docker("exec", "-u", "root", *HARNESS_PATH, c, "chmod", "-R", "a+rX", RUN_DIR)
                out = self._sh(c, f"{ZEROSHOT} run --title validate --graph {RUN_DIR}/graph.json --input {RUN_DIR}/input.json --runtime-config {RUN_DIR}/runtime.json --validate-only")[1]
                self.check(f"graph_valid_{arm}", lambda out=out: ('"valid":true' in out, out))
            loop, single = run_files(self.exp, "loop")["graph.json"], run_files(self.exp, "single")["graph.json"]
            self.check("builder_identical_across_arms", lambda: (loop["root"]["children"][0]["body"]["children"][0] == single["root"]["children"][0], "build node byte-identical"))
        finally:
            docker("rm", "-f", c, check=False)
            network.down()

    def diagnostic_run(self) -> None:
        """A tiny real run (not the benchmark prompts): proves the key and the proxy path work, and
        that tool commands get the toolchain but neither the key nor a route out."""
        c, network = self._probe("diagnostic")
        staging = self.results / "smoke-diagnostic"
        status: dict[str, Any] = {}
        try:
            write_json(staging / "input.json", {"task": DIAGNOSTIC_TASK})
            write_json(staging / "runtime.json", {"harness": "codex", "provider": "openai", "model": self.exp.model, "effort": "low"})
            docker("cp", f"{staging}/.", f"{c}:{RUN_DIR}")
            docker("exec", "-u", "root", *HARNESS_PATH, c, "chmod", "-R", "a+rX", RUN_DIR)
            # A workspace AGENTS.md must not reach the model (the workspace is pinned untrusted).
            docker("exec", "-u", "agent", c, "sh", "-c", f"printf '%s\\n' '{CANARY}: workspace instructions were loaded.' > /workspace/AGENTS.md")
            # Decoys in the world-writable directory that leads the image's PATH: the harness must
            # never run them (Zeroshot and every harness command get root-owned PATH entries only).
            for name in ("codex", "zeroshot", "tar", "sha256sum", "find", "stat"):
                self._sh(c, f"printf '#!/bin/sh\\ntouch /tmp/zsbench-decoy-{name}-ran\\nexit 1\\n' > /usr/local/cargo/bin/{name} && chmod 0755 /usr/local/cargo/bin/{name}")
            out = docker("exec", "-u", "agent", "-w", "/workspace", "-e", SECRET_ENV, "-e", f"PATH={SUBMIT_PATH}", c, *EXEC_NONDUMPABLE, ZEROSHOT, "run", "--title", "diagnostic", "--template", "single-worker", "--input", f"{RUN_DIR}/input.json", "--uniform-runtime-config", f"{RUN_DIR}/runtime.json", "--detach", timeout=600)
            (staging / "receipt.json").write_text(out)
            deadline = time.time() + 900
            while time.time() < deadline:
                runs = json.loads(docker("exec", "-u", "agent", *HARNESS_PATH, c, ZEROSHOT, "list", timeout=300))["runs"]
                status = runs[0]["status"] if runs else {}
                if status.get("phase") == "finished":
                    break
                time.sleep(10)
            write_json(staging / "status.json", status)
            self.check("diagnostic_run_finished", lambda: (status.get("phase") == "finished" and (status.get("terminalResult") or {}).get("status") == "succeeded", status.get("terminalResult")))
            report = self._sh(c, "cat /workspace/.zsbench-diagnostic.txt")[1]
            (staging / "diagnostic.txt").write_text(report)
            values = dict(line.split("=", 1) for line in report.splitlines() if "=" in line)
            self.check("model_call_and_key_work", lambda: (bool(values), "worker executed the script" if values else report[:300]))
            self.check("key_not_in_tool_env", lambda: (values.get("KEY_VARS_VISIBLE") == "0", f"KEY_VARS_VISIBLE={values.get('KEY_VARS_VISIBLE')}"))
            self.check("key_not_readable_from_proc", lambda: (values.get("PROC_KEY_VISIBLE") == "0", f"PROC_KEY_VISIBLE={values.get('PROC_KEY_VISIBLE')}"))
            self.check("proxy_not_in_tool_env", lambda: (values.get("PROXY_VARS_VISIBLE") == "0", f"PROXY_VARS_VISIBLE={values.get('PROXY_VARS_VISIBLE')}"))
            self.check("tool_tmpdir_is_plain_tmp", lambda: (values.get("TMPDIR") == "/tmp", f"TMPDIR={values.get('TMPDIR')}"))
            self.check("toolchain_env_in_tools", lambda: (
                values.get("CARGO", "").startswith("cargo ") and values.get("RUSTC", "").startswith("rustc ") and values.get("GO", "").startswith("go version"),
                {k: values.get(k) for k in ("CARGO", "RUSTC", "GO", "PYTHON", "RG")}))
            self.check("tool_egress_blocked", lambda: (values.get("EGRESS") == "blocked", values.get("EGRESS")))
            usage = (status.get("metadata") or {}).get("tokenUsage")
            self.check("token_usage_recorded", lambda: (bool(usage and usage.get("inputTokens")), usage))
        finally:
            try:
                archive(c, "root", TRAJECTORY_TAR, staging / "trajectories.tar.gz", timeout=600)
                # The per-node probes, run as attempts run them: none may execute a decoy either.
                for script in (HARNESS_FINGERPRINT, CODEX_HOME_SURFACES):
                    docker("exec", "-u", "root", *HARNESS_PATH, c, "sh", "-c", script, timeout=600)
                decoys = self._sh(c, "ls /tmp/zsbench-decoy-*-ran 2>/dev/null || true")[1]
                self.check("path_decoys_never_ran", lambda: (
                    status.get("phase") == "finished" and not decoys.strip(),
                    "decoys in /usr/local/cargo/bin (codex, zeroshot, tar, sha256sum, find, stat) never ran" if not decoys.strip() else decoys))
                tools = audit.command_audit(staging / "trajectories.tar.gz")
                sessions = [data for name, data in audit._walk_archive(staging / "trajectories.tar.gz") if "/sessions/" in name and name.endswith(".jsonl")]
                self.check("workspace_instructions_not_loaded", lambda: (
                    bool(sessions) and tools.get("agents_md_loaded") == 0 and not any(_canary_outside_tools(data) for data in sessions),
                    f"{len(sessions)} session(s); AGENTS.md loaded {tools.get('agents_md_loaded')} time(s)"))
                self.check("diagnostic_tools_worked", lambda: (tools.get("harness_tool_errors") == 0 and tools.get("tool_outputs", 0) >= 1, {k: tools.get(k) for k in ("tool_outputs", "harness_tool_errors", "harness_tool_error_examples")}))
            finally:
                docker("rm", "-f", c, check=False)
                network.down()

    def scoring_fidelity(self) -> None:
        """Evaluate a pinned, published leaderboard submission on this task and compare scores."""
        iid = self.exp.instance_id
        ref = pins()["fidelity_reference"]
        run_dir = self.results / "smoke-fidelity" / ref["submission"]
        target = run_dir / iid / "submission.tar.gz"
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            download(ref["archive_url"], ref["archive_sha256"], target)
        registry = f"https://raw.githubusercontent.com/ProgramBench/submissions/{ref['registry_commit']}/submissions/{ref['submission']}/_stats/score.json"
        with urllib.request.urlopen(registry, timeout=120) as r:
            published_tests = json.loads(r.read())[iid]
        ignore = set(leaderboard_ignores(self.cache).get(iid, []))
        kept = [v for k, v in published_tests.items() if k not in ignore]
        published = sum(kept) / len(kept)
        eval_json = run_dir / iid / f"{iid}.eval.json"
        if not eval_json.exists():
            env = {k: v for k, v in os.environ.items() if k != SECRET_ENV}
            env["PROGRAMBENCH_HF_REVISION"] = pins()["programbench_tests"]["revision"]
            cfg = self.exp.raw["eval"]
            with (self.results / "smoke-fidelity" / "eval.log").open("ab") as out:
                subprocess.run([sys.executable, "-m", "bench.pbeval", "eval", str(run_dir), "--docker-cpus", str(cfg["docker_cpus"]), "--image-tag", eval_image_tag(self.exp)], stdout=out, stderr=subprocess.STDOUT, env=env, check=False)
        scored = score_eval(eval_json, iid, leaderboard_ignores(self.cache))
        self.check("scoring_matches_leaderboard", lambda: (abs(scored["score"] - published) <= 0.02 and scored["rerun_plugin_pinned"], {"published": round(published, 4), "ours": round(scored["score"], 4), "rerun_plugin_pinned": scored["rerun_plugin_pinned"], "duplicate_entries": scored["duplicate_result_entries"]}))

    def summary(self) -> dict[str, Any]:
        ok = all(c["ok"] for c in self.checks.values())
        write_json(self.results / "smoke.json", {"ok": ok, "checks": self.checks})
        return {"ok": ok, "checks": self.checks}


def pipeline_checks(smoke: Smoke, summary: dict[str, Any]) -> None:
    """Assertions over the short end-to-end experiment."""
    attempts = {a["arm"]: a for a in summary["attempts"]}
    loop, single = attempts.get("loop"), attempts.get("single")
    results = smoke.results
    arms = set(smoke.exp.raw["arms"])
    smoke.check("pipeline_arms_complete", lambda: (set(attempts) == arms and all(a["state"] == "complete" for a in attempts.values()), {k: v["state"] for k, v in attempts.items()}))
    schedule = smoke.exp.raw["eval"].get("rounds")
    expected = {"build-1", "final"} | ({"check-1"} if schedule is None else {f"build-{r}" for r in schedule if r <= (loop or {}).get("builds", 0)})
    smoke.check("pipeline_loop_snapshots_by_round", lambda: (bool(loop) and expected <= set(loop["rounds"]), loop and sorted(loop["rounds"])))
    smoke.check("pipeline_loop_ran_its_rounds", lambda: (bool(loop) and loop["builds"] - 1 <= loop["checks"] <= loop["builds"] and loop["builds"] >= min(3, smoke.exp.max_iterations), {"builds": loop and loop["builds"], "checks": loop and loop["checks"], "max": smoke.exp.max_iterations}))
    smoke.check("pipeline_snapshots_without_errors", lambda: (all(not any("error" in s for s in (a.get("snapshots") or {}).values()) for a in attempts.values()), {k: {n: s.get("seconds") for n, s in (a.get("snapshots") or {}).items()} for k, a in attempts.items()}))
    if "single" in arms:
        smoke.check("pipeline_single_snapshot_equals_final", lambda: (
            bool(single) and (single["rounds"].get("final") or {}).get("passed") is not None and (single["rounds"].get("build-1") or {}).get("passed") == single["rounds"]["final"]["passed"],
            single and {k: v.get("passed") for k, v in single["rounds"].items()}))
    smoke.check("pipeline_scored_all_tests", lambda: (
        all((a["rounds"].get("final") or {}).get("scored_tests") == summary["expected_scored_tests"] for a in attempts.values()) and bool(summary["expected_scored_tests"]),
        {a["label"]: {k: (a["rounds"].get("final") or {}).get(k) for k in ("score", "passed", "scored_tests", "error_code", "duplicate_result_entries", "rerun_plugin_pinned")} for a in attempts.values()}))
    smoke.check("pipeline_rerun_plugin_pinned", lambda: (
        all(a["rounds"] for a in attempts.values()) and all((v or {}).get("rerun_plugin_pinned") for a in attempts.values() for v in a["rounds"].values()),
        "every eval installed the pinned pytest-rerunfailures"))
    smoke.check("pipeline_costed_from_transcripts", lambda: (
        all(a["cost_usd"].get("total", 0) > 0 and a["tokens"]["sessions"] for a in attempts.values()),
        {k: {"cost": a["cost_usd"], "sessions": [(s["node"], s["rounds"]) for s in a["tokens"]["sessions"]], "ledger_input": (a["tokens"]["ledger_total"] or {}).get("inputTokens"), "transcript_input": a["tokens"]["nodes"].get("total", {}).get("inputTokens")} for k, a in attempts.items()}))

    def transcripts_match_ledger() -> tuple[bool, Any]:
        """Fresh threads (every check; the single arm's build) must agree exactly with the ledger.
        The ledger over-counts a resumed builder thread, and records nothing for an execution that
        was interrupted (timeout, force-stop), so only nodes whose executions all ended normally
        are compared."""
        detail: dict[str, Any] = {}
        for a in attempts.values():
            ledger_nodes, nodes = a["tokens"].get("ledger_nodes") or {}, a["tokens"]["nodes"]
            ended_normally = {
                "check": all(v in ("accepted", "rejected") for v in a["verdicts"]),
                "build": all(b == "verified" for b in a["build_outcomes"]),
            }
            for node in ["check"] + (["build"] if a["arm"] == "single" else []):
                if node not in ledger_nodes and node not in nodes:
                    continue
                if not ended_normally[node] or a.get("force_stopped"):
                    detail[f"{a['label']}.{node}"] = "skipped: interrupted execution"
                    continue
                same = ledger_nodes.get(node) == nodes.get(node)
                detail[f"{a['label']}.{node}"] = "match" if same else {"ledger": ledger_nodes.get(node), "transcripts": nodes.get(node)}
        compared = [v for v in detail.values() if not str(v).startswith("skipped")]
        return bool(compared) and all(v == "match" for v in compared), detail

    smoke.check("pipeline_transcripts_match_ledger", transcripts_match_ledger)
    smoke.check("pipeline_no_secret_in_artifacts", lambda: (summary["secrets"]["checked_literal_key"] and not summary["secrets"]["literal_key_hits"], summary["secrets"]))
    egress = summary.get("egress") or {}
    smoke.check("pipeline_egress_only_model_api", lambda: (set(egress.get("established") or {}) == {"api.openai.com"}, {k: egress.get(k) for k in ("established", "refused")}))
    smoke.check("pipeline_no_web_search", lambda: (all(not a["commands"].get("web_search_calls") for a in attempts.values()), {k: a["commands"].get("web_search_calls") for k, a in attempts.items()}))
    smoke.check("pipeline_no_disqualifying_audit", lambda: (
        all(not (a["commands"].get("rule_counts") or {}).get(rule) for a in attempts.values() for rule in audit.DISQUALIFYING),
        {k: a["commands"].get("rule_counts") for k, a in attempts.items()}))
    smoke.check("pipeline_tools_worked", lambda: (
        all(a["commands"].get("harness_tool_errors") == 0 and a["commands"].get("tool_outputs", 0) >= 10 for a in attempts.values()),
        {k: {f: a["commands"].get(f) for f in ("tool_outputs", "harness_tool_errors", "harness_tool_error_examples")} for k, a in attempts.items()}))

    def has_compile_sh(label: str) -> bool:
        with tarfile.open(results / "attempts" / label / "submission.tar.gz") as tar:
            return any(m.name in ("./compile.sh", "compile.sh") for m in tar.getmembers())

    smoke.check("pipeline_builder_produced_compile_sh", lambda: (all(has_compile_sh(a["label"]) for a in attempts.values()), "compile.sh present in every final workspace"))

    def reference_absent(a: dict[str, Any]) -> bool:
        built = [v.get("executable_hash") for v in a["rounds"].values() if v]
        return bool(a.get("reference_sha256")) and not a["reference_copies_in_final"] and not a.get("reference_copies_in_first_build") and a["reference_sha256"] not in built

    smoke.check("pipeline_no_reference_in_archives", lambda: (
        all(reference_absent(a) for a in attempts.values()),
        {k: {"reference_sha256": (a.get("reference_sha256") or "")[:12], "in_final": a["reference_copies_in_final"], "in_build_1": a.get("reference_copies_in_first_build")} for k, a in attempts.items()}))
    smoke.check("pipeline_codex_home_clean", lambda: (
        all(not a.get("codex_home_changes") and not a["commands"].get("agents_md_loaded") for a in attempts.values()),
        {k: {"changes": a.get("codex_home_changes"), "agents_md_loaded": a["commands"].get("agents_md_loaded")} for k, a in attempts.items()}))
    smoke.check("pipeline_harness_unchanged", lambda: (all(a.get("harness_unchanged") is True for a in attempts.values()), {k: a.get("harness_unchanged") for k, a in attempts.items()}))
    smoke.check("pipeline_codex_config_unchanged", lambda: (all(a["codex_config_unchanged"] is True for a in attempts.values()), {k: a["codex_config_unchanged"] for k, a in attempts.items()}))
    smoke.check("pipeline_provenance_recorded", lambda: (
        (summary.get("provenance") or {}).get("vcs_ref") not in (None, "unknown") and (summary.get("provenance") or {}).get("vcs_dirty") == "false",
        summary.get("provenance")))
