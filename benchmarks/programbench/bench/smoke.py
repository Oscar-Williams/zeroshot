"""Smoke tests: isolation, environment, graph validation, a diagnostic run, the full pipeline with
short limits, and scoring fidelity against a published leaderboard submission."""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable

from .attempt import RUN_DIR, run_files
from .config import Experiment
from . import audit
from .evaluate import leaderboard_ignores, score_eval
from .images import Network
from .util import SECRET_ENV, docker, log, write_json

# A published leaderboard run on the same task, used to prove our eval + scoring reproduces the
# leaderboard's number (GPT-5.6 Sol xhigh, mini-SWE-agent).
REFERENCE_RUN = "20260802_mini-v2.4.2_gpt-5-6-sol-xhigh"
REFERENCE_REGISTRY_COMMIT = "794fa30bbfd65059d5c67e192ceb6da659c9ef51"

DIAGNOSTIC_TASK = """This is an environment diagnostic, not a coding task. Run exactly the following shell script once, then stop. Do not modify it and do not run anything else.

```sh
{
  echo "USER=$(id -un)"
  echo "HOME=$HOME"
  echo "ENV_NAMES=$(env | cut -d= -f1 | sort | tr '\\n' ' ')"
  echo "KEY_VARS_VISIBLE=$(env | grep -c -E '^(OPENAI_API_KEY|CODEX_API_KEY)=')"
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
    def __init__(self, exp: Experiment, results: Path, cache: Path, agent_image: str, network: Network):
        self.exp, self.results, self.cache, self.image, self.network = exp, results, cache, agent_image, network
        self.checks: dict[str, dict[str, Any]] = {}

    def check(self, name: str, fn: Callable[[], tuple[bool, Any]]) -> bool:
        try:
            ok, detail = fn()
        except Exception as error:
            ok, detail = False, f"{type(error).__name__}: {error}"
        self.checks[name] = {"ok": bool(ok), "detail": detail}
        log(f"smoke {'PASS' if ok else 'FAIL'} {name}: {str(detail)[:300]}")
        return bool(ok)

    def _probe_container(self, name: str) -> None:
        docker("rm", "-f", name, check=False)
        proxy = self.network.proxy_url
        env = []
        for var in ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy"):
            env += ["-e", f"{var}={proxy}"]
        for var in ("NO_PROXY", "no_proxy"):
            env += ["-e", f"{var}=localhost,127.0.0.1,::1"]
        docker("run", "-d", "--name", name, "--hostname", "workspace", "--init", "--network", self.network.name, "--user", "agent", "--workdir", "/workspace", "--cap-drop", "SYS_PTRACE", "--label", "zsbench=1", *env, self.image, "sleep", "infinity")

    def _sh(self, container: str, script: str, user: str = "agent", timeout: float = 120) -> tuple[int, str]:
        from .util import run

        result = run(["docker", "exec", "-u", user, container, "bash", "-c", script], check=False, timeout=timeout)
        return result.returncode, (result.stdout + result.stderr).decode(errors="replace").strip()

    def isolation(self) -> None:
        c = f"zsbench-{self.exp.id}-probe"
        self._probe_container(c)
        try:
            self.check("runs_as_agent", lambda: (self._sh(c, "id -un")[1] == "agent", self._sh(c, "id")[1]))
            self.check("reference_binary_unreadable", lambda: (self._sh(c, "test -r /workspace/executable")[0] != 0, "execute-only for the agent"))
            self.check("reference_binary_runs", lambda: (self._sh(c, "./executable --version")[0] == 0, self._sh(c, "./executable --version")[1]))
            self.check("direct_egress_blocked", lambda: (
                self._sh(c, "curl -sS -m 8 --noproxy '*' -o /dev/null https://example.com")[0] != 0
                and self._sh(c, "curl -sS -m 8 --noproxy '*' -o /dev/null https://api.openai.com")[0] != 0,
                "no route out without the proxy"))
            self.check("proxy_refuses_other_hosts", lambda: (
                self._sh(c, "curl -sS -m 10 -o /dev/null https://example.com")[0] != 0
                and self._sh(c, "curl -sS -m 10 -o /dev/null https://github.com")[0] != 0
                and self._sh(c, "curl -sS -m 10 -o /dev/null https://pypi.org/simple/")[0] != 0,
                "example.com, github.com, pypi.org refused"))
            code = lambda: self._sh(c, "curl -sS -m 20 -o /dev/null -w '%{http_code}' https://api.openai.com/v1/models")[1]
            self.check("proxy_allows_model_api", lambda: (code() == "401", f"unauthenticated GET /v1/models -> {code()}"))
            self.check("tool_versions", lambda: (self._sh(c, "zeroshot --version && codex --version")[0] == 0, self._sh(c, "zeroshot --version; codex --version")[1]))
            cfg = lambda: self._sh(c, "cat ~/.codex/config.toml")[1]
            self.check("codex_web_search_disabled", lambda: ('web_search = "disabled"' in cfg(), "config.toml"))
            self.check("workspace_origin_placeholder", lambda: (self._sh(c, "git -C /workspace remote get-url origin")[1].endswith("zeroshot-bench/local-workspace.git"), "placeholder origin"))
            for arm in ("loop", "single"):
                files = run_files(self.exp, arm)
                staging = self.results / "smoke-validate" / arm
                for name, value in files.items():
                    write_json(staging / name, value)
                docker("cp", f"{staging}/.", f"{c}:{RUN_DIR}")
                docker("exec", "-u", "root", c, "chmod", "-R", "a+rX", RUN_DIR)
                self.check(f"graph_valid_{arm}", lambda: (
                    '"valid":true' in (out := self._sh(c, f"zeroshot run --title validate --graph {RUN_DIR}/graph.json --input {RUN_DIR}/input.json --runtime-config {RUN_DIR}/runtime.json --validate-only")[1]),
                    out))
            loop, single = run_files(self.exp, "loop")["graph.json"], run_files(self.exp, "single")["graph.json"]
            self.check("builder_identical_across_arms", lambda: (loop["root"]["children"][0]["body"]["children"][0] == single["root"]["children"][0], "build node byte-identical"))
        finally:
            docker("rm", "-f", c, check=False)

    def diagnostic_run(self) -> None:
        """A tiny real run (not the benchmark prompts): proves the key, the proxy path, the tool
        environment Codex gives commands, and that the key is not visible to those commands."""
        c = f"zsbench-{self.exp.id}-diagnostic"
        self._probe_container(c)
        try:
            staging = self.results / "smoke-diagnostic"
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
            self.check("key_not_visible_to_tools", lambda: (values.get("KEY_VARS_VISIBLE") == "0", f"KEY_VARS_VISIBLE={values.get('KEY_VARS_VISIBLE')}"))
            self.check("toolchain_env_in_tools", lambda: (
                values.get("CARGO", "").startswith("cargo ") and values.get("RUSTC", "").startswith("rustc ") and values.get("GO", "").startswith("go version"),
                {k: values.get(k) for k in ("CARGO", "RUSTC", "GO", "PYTHON")}))
            self.check("tool_egress_blocked", lambda: (values.get("EGRESS") == "blocked", values.get("EGRESS")))
            names = values.get("ENV_NAMES", "")
            self.check("tool_env_has_cargo_home", lambda: ("CARGO_HOME" in names and "RUSTUP_HOME" in names, names))
            usage = (status.get("metadata") or {}).get("tokenUsage")
            self.check("token_usage_recorded", lambda: (bool(usage and usage.get("inputTokens")), usage))
        finally:
            try:
                from .attempt import TRAJECTORY_TAR
                from .util import docker_to_file

                docker_to_file(["exec", "-u", "root", c, *TRAJECTORY_TAR], self.results / "smoke-diagnostic" / "trajectories.tar.gz", timeout=600)
                tools = audit.command_audit(self.results / "smoke-diagnostic" / "trajectories.tar.gz")
                self.check("diagnostic_tools_worked", lambda: (tools.get("harness_tool_errors") == 0 and tools.get("tool_outputs", 0) >= 1, {k: tools.get(k) for k in ("tool_outputs", "harness_tool_errors", "harness_tool_error_examples")}))
            finally:
                docker("rm", "-f", c, check=False)

    def scoring_fidelity(self) -> None:
        """Evaluate a published leaderboard submission on this task and compare scores."""
        iid = self.exp.instance_id
        base = f"https://raw.githubusercontent.com/ProgramBench/{REFERENCE_RUN}/main/{iid}"
        with urllib.request.urlopen(f"{base}/submission.tar.gz.url", timeout=60) as r:
            tar_url = r.read().decode().strip()
        run_dir = self.results / "smoke-fidelity" / REFERENCE_RUN
        target = run_dir / iid / "submission.tar.gz"
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            with urllib.request.urlopen(tar_url, timeout=300) as r:
                target.write_bytes(r.read())
        registry = f"https://raw.githubusercontent.com/ProgramBench/submissions/{REFERENCE_REGISTRY_COMMIT}/submissions/{REFERENCE_RUN}/_stats/score.json"
        with urllib.request.urlopen(registry, timeout=120) as r:
            published_tests = json.loads(r.read())[iid]
        ignore = set(leaderboard_ignores(self.cache).get(iid, []))
        kept = [v for k, v in published_tests.items() if k not in ignore]
        published = sum(kept) / len(kept)
        eval_json = run_dir / iid / f"{iid}.eval.json"
        if not eval_json.exists():
            import subprocess

            cfg = self.exp.raw["eval"]
            with (self.results / "smoke-fidelity" / "eval.log").open("ab") as out:
                subprocess.run(["programbench", "eval", str(run_dir), "--docker-cpus", str(cfg["docker_cpus"])], stdout=out, stderr=subprocess.STDOUT, check=False)
        ours = score_eval(eval_json, iid, leaderboard_ignores(self.cache))["score"]
        self.check("scoring_matches_leaderboard", lambda: (abs(ours - published) <= 0.02, {"published": round(published, 4), "ours": round(ours, 4)}))

    def summary(self) -> dict[str, Any]:
        ok = all(c["ok"] for c in self.checks.values())
        write_json(self.results / "smoke.json", {"ok": ok, "checks": self.checks})
        return {"ok": ok, "checks": self.checks}


def pipeline_checks(smoke: Smoke, summary: dict[str, Any]) -> None:
    """Assertions over the short end-to-end experiment."""
    attempts = {a["arm"]: a for a in summary["attempts"]}
    loop, single = attempts.get("loop"), attempts.get("single")
    smoke.check("pipeline_both_arms_complete", lambda: (bool(loop and single and loop["state"] == "complete" and single["state"] == "complete"), {k: v["state"] for k, v in attempts.items()}))
    smoke.check("pipeline_loop_snapshots", lambda: (bool(loop and "build-1" in loop["scores_by_round"]), loop and sorted(loop["scores_by_round"])))
    smoke.check("pipeline_single_snapshot_equals_final", lambda: (bool(single) and single["scores_by_round"].get("build-1") == single["score_final"], single and single["scores_by_round"]))
    smoke.check("pipeline_scored", lambda: (all(a["score_final"] is not None for a in attempts.values()), {k: a["score_final"] for k, a in attempts.items()}))
    smoke.check("pipeline_costed", lambda: (all(a["cost_usd"].get("total", 0) > 0 for a in attempts.values()), {k: a["cost_usd"] for k, a in attempts.items()}))
    smoke.check("pipeline_no_secret_in_artifacts", lambda: (summary["secrets"]["checked_literal_key"] and not summary["secrets"]["literal_key_hits"], summary["secrets"]))
    egress = summary.get("egress") or {}
    smoke.check("pipeline_egress_only_model_api", lambda: (set(egress.get("established") or {}) == {"api.openai.com"}, {k: egress.get(k) for k in ("established", "refused")}))
    smoke.check("pipeline_no_web_search", lambda: (all(not a["commands"].get("web_search_calls") for a in attempts.values()), {k: a["commands"].get("web_search_calls") for k, a in attempts.items()}))
    smoke.check("pipeline_tools_worked", lambda: (
        all(a["commands"].get("harness_tool_errors") == 0 and a["commands"].get("tool_outputs", 0) >= 10 for a in attempts.values()),
        {k: {f: a["commands"].get(f) for f in ("tool_outputs", "harness_tool_errors", "harness_tool_error_examples")} for k, a in attempts.items()}))
    results = smoke.results
    def has_compile_sh(label: str) -> bool:
        import tarfile

        with tarfile.open(results / "attempts" / label / "submission.tar.gz") as tar:
            return any(m.name in ("./compile.sh", "compile.sh") for m in tar.getmembers())
    smoke.check("pipeline_builder_produced_compile_sh", lambda: (all(has_compile_sh(a["label"]) for a in attempts.values()), "compile.sh present in every final workspace"))
    scores = json.loads((results / "scores.json").read_text())
    smoke.check("pipeline_hidden_tests_ran", lambda: (
        all((scores.get(f"{a['label']}__final") or {}).get("scored_tests", 0) > 0 for a in attempts.values()),
        {a["label"]: {k: (scores.get(f"{a['label']}__final") or {}).get(k) for k in ("score", "passed", "scored_tests", "error_code")} for a in attempts.values()}))
