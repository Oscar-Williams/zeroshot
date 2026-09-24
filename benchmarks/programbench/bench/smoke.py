"""Smoke tests: isolation, environment, graph validation, a diagnostic run, the full pipeline with
short limits, and scoring fidelity against a published leaderboard submission."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tarfile
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import audit
from .attempt import RUN_DIR, TAR_OK, TRAJECTORY_TAR, run_files
from .config import Experiment, pins
from .evaluate import eval_image_tag, leaderboard_ignores, score_eval
from .images import Network
from .util import SECRET_ENV, docker, docker_to_file, download, log, run, write_json

DIAGNOSTIC_TASK = """This is an environment diagnostic, not a coding task. Run exactly the following shell script once, then stop. Do not modify it and do not run anything else.

```sh
{
  echo "USER=$(id -un)"
  echo "HOME=$HOME"
  echo "ENV_NAMES=$(env | cut -d= -f1 | sort | tr '\\n' ' ')"
  echo "KEY_VARS_VISIBLE=$(env | grep -c -E '^(OPENAI_API_KEY|CODEX_API_KEY)=')"
  echo "PROC_KEY_VISIBLE=$(grep -l -a -E '(OPENAI_API_KEY|CODEX_API_KEY)=' /proc/[0-9]*/environ 2>/dev/null | wc -l)"
  echo "PROXY_VARS_VISIBLE=$(env | grep -c -i -E '^(https?|all)_proxy=')"
  echo "CARGO=$(cargo --version 2>&1 | head -1)"
  echo "RUSTC=$(rustc --version 2>&1 | head -1)"
  echo "GO=$(go version 2>&1 | head -1)"
  echo "PYTHON=$(python3 --version 2>&1)"
  echo "RG=$(command -v rg || echo missing)"
  if curl -sS -m 8 -o /dev/null https://example.com 2>/dev/null; then echo "EGRESS=open"; else echo "EGRESS=blocked"; fi
} > /workspace/.zsbench-diagnostic.txt 2>&1
```
"""


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
            self.check("reference_binary_unreadable", lambda: (self._sh(c, "test -r /workspace/executable")[0] != 0, "execute-only for the agent"))
            self.check("reference_binary_runs", lambda: (self._sh(c, "./executable --version")[0] == 0, self._sh(c, "./executable --version")[1]))
            self.check("harness_binaries_unreadable", lambda: (
                self._sh(c, "test -r /usr/local/bin/zeroshot || test -r /opt/codex/bin/codex || test -r /opt/codex/bin/codex-code-mode-host")[0] != 0,
                "zeroshot and Codex executables are execute-only (their /proc entries are protected)"))
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
            self.check("tool_versions", lambda: (self._sh(c, "zeroshot --version && codex --version")[0] == 0, self._sh(c, "zeroshot --version; codex --version")[1]))

            def codex_config() -> tuple[bool, str]:
                config = self._sh(c, "cat ~/.codex/config.toml")[1]
                required = ('web_search = "disabled"', "shell_snapshot = false", "generate_memories = false", '"*PROXY*"')
                return all(s in config for s in required), "web search, shell snapshot and memories off; proxy vars hidden from tools"

            self.check("codex_config_hardened", codex_config)
            self.check("workspace_origin_placeholder", lambda: (self._sh(c, "git -C /workspace remote get-url origin")[1].endswith("zeroshot-bench/local-workspace.git"), "placeholder origin"))
            for arm in ("loop", "single"):
                files = run_files(self.exp, arm)
                staging = self.results / "smoke-validate" / arm
                for name, value in files.items():
                    write_json(staging / name, value)
                docker("cp", f"{staging}/.", f"{c}:{RUN_DIR}")
                docker("exec", "-u", "root", c, "chmod", "-R", "a+rX", RUN_DIR)
                out = self._sh(c, f"zeroshot run --title validate --graph {RUN_DIR}/graph.json --input {RUN_DIR}/input.json --runtime-config {RUN_DIR}/runtime.json --validate-only")[1]
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
        try:
            write_json(staging / "input.json", {"task": DIAGNOSTIC_TASK})
            write_json(staging / "runtime.json", {"harness": "codex", "provider": "openai", "model": self.exp.model, "effort": "low"})
            docker("cp", f"{staging}/.", f"{c}:{RUN_DIR}")
            docker("exec", "-u", "root", c, "chmod", "-R", "a+rX", RUN_DIR)
            out = docker("exec", "-u", "agent", "-w", "/workspace", "-e", SECRET_ENV, c, "zeroshot", "run", "--title", "diagnostic", "--template", "single-worker", "--input", f"{RUN_DIR}/input.json", "--uniform-runtime-config", f"{RUN_DIR}/runtime.json", "--detach", timeout=600)
            (staging / "receipt.json").write_text(out)
            deadline = time.time() + 900
            status: dict[str, Any] = {}
            while time.time() < deadline:
                runs = json.loads(docker("exec", "-u", "agent", c, "zeroshot", "list"))["runs"]
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
            self.check("toolchain_env_in_tools", lambda: (
                values.get("CARGO", "").startswith("cargo ") and values.get("RUSTC", "").startswith("rustc ") and values.get("GO", "").startswith("go version"),
                {k: values.get(k) for k in ("CARGO", "RUSTC", "GO", "PYTHON", "RG")}))
            self.check("tool_egress_blocked", lambda: (values.get("EGRESS") == "blocked", values.get("EGRESS")))
            usage = (status.get("metadata") or {}).get("tokenUsage")
            self.check("token_usage_recorded", lambda: (bool(usage and usage.get("inputTokens")), usage))
        finally:
            try:
                docker_to_file(["exec", "-u", "root", c, *TRAJECTORY_TAR], staging / "trajectories.tar.gz", timeout=600, ok_codes=TAR_OK)
                tools = audit.command_audit(staging / "trajectories.tar.gz")
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
    smoke.check("pipeline_both_arms_complete", lambda: (bool(loop and single and loop["state"] == "complete" and single["state"] == "complete"), {k: v["state"] for k, v in attempts.items()}))
    smoke.check("pipeline_loop_snapshots_by_round", lambda: (bool(loop) and {"build-1", "check-1"} <= set(loop["rounds"]), loop and sorted(loop["rounds"])))
    smoke.check("pipeline_snapshots_without_errors", lambda: (all(not any("error" in s for s in (a.get("snapshots") or {}).values()) for a in attempts.values()), {k: {n: s.get("seconds") for n, s in (a.get("snapshots") or {}).items()} for k, a in attempts.items()}))
    smoke.check("pipeline_single_snapshot_equals_final", lambda: (bool(single) and (single["rounds"].get("build-1") or {}).get("passed") == (single["rounds"].get("final") or {}).get("passed"), single and {k: v.get("passed") for k, v in single["rounds"].items()}))
    smoke.check("pipeline_scored_all_tests", lambda: (
        all((a["rounds"].get("final") or {}).get("scored_tests") == summary["expected_scored_tests"] for a in attempts.values()) and bool(summary["expected_scored_tests"]),
        {a["label"]: {k: (a["rounds"].get("final") or {}).get(k) for k in ("score", "passed", "scored_tests", "error_code", "duplicate_result_entries", "rerun_plugin_pinned")} for a in attempts.values()}))
    smoke.check("pipeline_rerun_plugin_pinned", lambda: (all((v or {}).get("rerun_plugin_pinned") for a in attempts.values() for v in a["rounds"].values()), "every eval installed the pinned pytest-rerunfailures"))
    smoke.check("pipeline_costed_from_transcripts", lambda: (
        all(a["cost_usd"].get("total", 0) > 0 and a["tokens"]["sessions"] for a in attempts.values()),
        {k: {"cost": a["cost_usd"], "sessions": [(s["node"], s["turns"]) for s in a["tokens"]["sessions"]], "ledger_input": (a["tokens"]["ledger_total"] or {}).get("inputTokens"), "transcript_input": a["tokens"]["nodes"].get("total", {}).get("inputTokens")} for k, a in attempts.items()}))
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
    smoke.check("pipeline_no_reference_in_archives", lambda: (all(not a["reference_copies_in_final"] for a in attempts.values()), {k: a["reference_copies_in_final"] for k, a in attempts.items()}))
    smoke.check("pipeline_codex_config_unchanged", lambda: (all(a["codex_config_unchanged"] is True for a in attempts.values()), {k: a["codex_config_unchanged"] for k, a in attempts.items()}))
    smoke.check("pipeline_provenance_recorded", lambda: (
        (summary.get("provenance") or {}).get("vcs_ref") not in (None, "unknown") and (summary.get("provenance") or {}).get("vcs_dirty") == "false",
        summary.get("provenance")))
