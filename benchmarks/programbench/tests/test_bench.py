"""Unit tests that need neither Docker nor an API key: ``python -m unittest discover -s tests``.

Requires Python 3.12 (the runner image); the ProgramBench-dependent tests also need programbench.
"""

from __future__ import annotations

import copy
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench import accounting, audit, config, graphs, images, report  # noqa: E402
from bench.attempt import run_files, snapshot_label  # noqa: E402

EXPERIMENT = config.load("experiments/luna-xhigh-svgbob.json")


def _tar(path: Path, files: dict[str, bytes]) -> Path:
    with tarfile.open(path, "w:gz") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return path


def _rollout(prompt_text: str, turns: list[tuple[int, int]]) -> bytes:
    """A Codex transcript whose cumulative usage grows by (input, output) per turn."""
    lines = [{"type": "response_item", "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Execute this graph node\nAuthored instructions:\n" + prompt_text}]}}]
    total_in = total_out = 0
    for tokens_in, tokens_out in turns:
        total_in += tokens_in
        total_out += tokens_out
        lines.append({"type": "event_msg", "payload": {"type": "task_started"}})
        lines.append({"type": "event_msg", "payload": {"type": "token_count", "info": {"total_token_usage": {"input_tokens": total_in, "cached_input_tokens": total_in // 2, "cache_write_input_tokens": 0, "output_tokens": total_out}}}})
        lines.append({"type": "event_msg", "payload": {"type": "task_complete"}})
    return "\n".join(json.dumps(line) for line in lines).encode()


class GraphTests(unittest.TestCase):
    def test_builder_node_is_identical_across_arms(self):
        loop = run_files(EXPERIMENT, "loop")["graph.json"]
        single = run_files(EXPERIMENT, "single")["graph.json"]
        self.assertEqual(loop["root"]["children"][0]["body"]["children"][0], single["root"]["children"][0])

    def test_prompts_are_frozen_generic_text(self):
        loop = run_files(EXPERIMENT, "loop")["graph.json"]
        build, check = loop["root"]["children"][0]["body"]["children"]
        self.assertEqual(build["instructions"], config.prompt("builder"))
        self.assertEqual(check["instructions"], config.prompt("checker"))
        for text in (build["instructions"], check["instructions"]):
            for word in ("ProgramBench", "svgbob", "SVG", "reverse", "executable"):
                self.assertNotIn(word, text)

    def test_loop_stops_on_accept_and_feeds_back_diagnostics(self):
        loop = run_files(EXPERIMENT, "loop")["graph.json"]["root"]["children"][0]
        self.assertEqual(loop["until"]["value"], {"name": "check", "source": "signal", "field": "verdict"})
        self.assertEqual(loop["maxIterations"], EXPERIMENT.max_iterations)
        self.assertEqual(loop["body"]["children"][1]["writeBindings"][0]["target"], ["feedback"])
        self.assertEqual(loop["promotedStatePaths"], [["feedback"]])

    def test_runtime_sessions(self):
        plan = graphs.runtime_plan("loop", "m", "xhigh")
        self.assertEqual(plan["nodes"]["build"]["sessionScope"], "node_instance")
        self.assertEqual(plan["nodes"]["check"]["sessionScope"], "execution")
        self.assertEqual(set(graphs.runtime_plan("single", "m", "xhigh")["nodes"]), {"build"})

    def test_task_statement_is_upstream_minus_scaffold(self):
        task = config.prompt("task")
        self.assertIn("Make sure that you have a `./compile.sh` file", task)
        for scaffold in ("COMPLETE_TASK_AND_SUBMIT", "bash tool", "system_information", "helpful assistant"):
            self.assertNotIn(scaffold, task)
        root = Path(__file__).resolve().parent.parent
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run([sys.executable, str(root / "scripts/build_task_prompt.py"), str(root / "prompts/upstream/mini-swe-agent-programbench.yaml"), f"{tmp}/task.md"], check=True)
            self.assertEqual(Path(tmp, "task.md").read_text(), (root / "prompts/task.md").read_text())


class ConfigTests(unittest.TestCase):
    def _load(self, raw):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(raw, f)
        try:
            return config.load(f.name)
        finally:
            os.unlink(f.name)

    def test_order_must_match_repeats(self):
        raw = copy.deepcopy(EXPERIMENT.raw)
        raw["order"] = raw["order"][:-1]
        with self.assertRaises(ValueError):
            self._load(raw)

    def test_task_image_must_be_pinned(self):
        raw = copy.deepcopy(EXPERIMENT.raw)
        raw["task"]["image"] = raw["task"]["image"].split("@")[0]
        with self.assertRaises(ValueError):
            self._load(raw)

    def test_pilot_is_five_loop_three_single_interleaved(self):
        order = EXPERIMENT.raw["order"]
        self.assertEqual((order.count("loop"), order.count("single")), (5, 3))
        self.assertEqual(set(order[:2]), {"loop", "single"})
        self.assertFalse(any(a == b == "single" for a, b in zip(order, order[1:], strict=False)))

    def test_digest_covers_code_but_not_tests_or_results(self):
        names = {str(p.relative_to(config.ROOT)) for p in config.code_files()}
        self.assertTrue({"bench/attempt.py", "bench/evaluate.py", "requirements.lock", "agent/Dockerfile"} <= names)
        self.assertFalse(any(n.startswith(("tests/", "results/")) or "/." in f"/{n}" for n in names))


class CodexConfigTests(unittest.TestCase):
    def test_rendered_config_keeps_credentials_web_and_memory_out(self):
        import tomllib

        rendered = images.codex_config({"PATH": "/usr/bin", "CARGO_HOME": "/usr/local/cargo", "HOME": "/root"})
        parsed = tomllib.loads(rendered)
        self.assertEqual(parsed["web_search"], "disabled")
        self.assertFalse(parsed["features"]["shell_snapshot"])
        self.assertFalse(parsed["memories"]["generate_memories"])
        self.assertFalse(parsed["memories"]["use_memories"])
        policy = parsed["shell_environment_policy"]
        self.assertFalse(policy["ignore_default_excludes"])
        self.assertTrue({"*KEY*", "*TOKEN*", "*PROXY*"} <= set(policy["exclude"]))
        self.assertEqual(policy["set"]["CARGO_HOME"], "/usr/local/cargo")
        self.assertEqual(policy["set"]["TMPDIR"], "/tmp")
        self.assertNotIn("HOME", policy["set"])
        self.assertEqual(parsed["projects"]["/workspace"]["trust_level"], "untrusted")


class AccountingTests(unittest.TestCase):
    PRICING = EXPERIMENT.pricing

    def test_cost_splits_cache_reads_and_writes(self):
        tokens = {"inputTokens": 1_000_000, "cacheReadInputTokens": 900_000, "cacheCreationInputTokens": 50_000, "outputTokens": 10_000}
        expected = (50_000 * 0.2 + 900_000 * 0.02 + 50_000 * 0.25 + 10_000 * 1.2) / 1e6
        self.assertAlmostEqual(accounting.cost(tokens, self.PRICING), expected)

    def test_resumed_builder_is_counted_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            _tar(Path(tmp, "trajectories.tar.gz"), {
                ".codex/sessions/2026/09/24/rollout-a.jsonl": _rollout(config.prompt("builder"), [(1000, 100), (300, 20)]),
                ".codex/sessions/2026/09/24/rollout-b.jsonl": _rollout(config.prompt("checker"), [(400, 40)]),
            })
            usage = accounting.usage(Path(tmp))
        self.assertEqual(usage["nodes"]["build"]["inputTokens"], 1300)
        self.assertEqual(usage["nodes"]["check"]["inputTokens"], 400)
        self.assertEqual(usage["nodes"]["total"]["outputTokens"], 160)
        self.assertEqual(usage["first_build_turn"]["inputTokens"], 1000)
        self.assertEqual(sorted((s["node"], s["turns"]) for s in usage["sessions"]), [("build", 2), ("check", 1)])

    def test_first_build_turn_comes_from_the_earliest_builder_thread(self):
        # After a build error Zeroshot starts a new builder thread; only the earliest holds build 1.
        with tempfile.TemporaryDirectory() as tmp:
            _tar(Path(tmp, "trajectories.tar.gz"), {
                ".codex/sessions/2026/09/24/rollout-2026-09-24T02-00-00-b.jsonl": _rollout(config.prompt("builder"), [(700, 70)]),
                ".codex/sessions/2026/09/24/rollout-2026-09-24T01-00-00-a.jsonl": _rollout(config.prompt("builder"), [(1000, 100)]),
            })
            usage = accounting.usage(Path(tmp))
        self.assertEqual(usage["first_build_turn"]["inputTokens"], 1000)
        self.assertEqual(usage["nodes"]["build"]["inputTokens"], 1700)

    def test_ledger_view_double_counts_resumed_sessions(self):
        events = [
            {"kind": "node_started", "reference": {"node": "build", "execution": 1}},
            {"kind": "token_usage_observed", "execution": 1, "usage": {"inputTokens": 1000, "outputTokens": 100}},
            {"kind": "node_started", "reference": {"node": "build", "execution": 3}},
            {"kind": "token_usage_observed", "execution": 3, "usage": {"inputTokens": 1300, "outputTokens": 120}},
        ]
        self.assertEqual(accounting.usage_from_events(events)["build"]["inputTokens"], 2300)  # truth: 1300


class SnapshotLabelTests(unittest.TestCase):
    def test_labels_follow_rounds_and_retries(self):
        self.assertEqual(snapshot_label("build", 1, set()), "build-1")
        self.assertEqual(snapshot_label("check", 1, {"build-1"}), "check-1")
        self.assertEqual(snapshot_label("check", 1, {"build-1", "check-1"}), "check-1.2")
        self.assertEqual(snapshot_label("build", 2, {"build-1", "check-1", "check-1.2"}), "build-2")


class AuditTests(unittest.TestCase):
    def test_command_rules(self):
        rules = audit.COMMAND_RULES
        self.assertTrue(rules["reference_binary_analysis"].search("objdump -d ./executable"))
        self.assertTrue(rules["binary_instrumentation"].search("LD_PRELOAD=./dump.so ./executable"))
        self.assertFalse(rules["binary_instrumentation"].search("./executable < in.txt > out.svg"))
        self.assertTrue(rules["reference_binary_moved_or_copied"].search("mv /workspace/executable /tmp/ref"))
        self.assertFalse(rules["reference_binary_moved_or_copied"].search("./executable -s 'x' > out.svg"))
        self.assertTrue(rules["network_fetch"].search("cargo install svgbob_cli"))
        self.assertTrue(rules["model_api_calls"].search("curl https://api.openai.com/v1/responses"))
        self.assertTrue(rules["proxy_usage"].search("HTTPS_PROXY=http://zsbench-x-proxy:8888 curl x"))
        self.assertTrue(rules["proxy_usage"].search("curl -x 172.18.0.2:8888 https://example.com"))
        self.assertFalse(rules["proxy_usage"].search("curl -s http://localhost:8888/render"))
        self.assertTrue(rules["process_environment_read"].search("cat /proc/123/environ"))
        self.assertTrue(rules["process_environment_read"].search("open(os.path.join('/proc', p, 'environ'))"))
        self.assertFalse(rules["process_environment_read"].search("python3 -c 'import os; print(os.environ)'"))
        self.assertTrue(rules["process_environment_read"].search("ps eww"))
        for listing in ("ps -ef", "ps aux | grep svgbob", "ps -efww", "ps -eo pid,cmd"):
            self.assertFalse(rules["process_environment_read"].search(listing), listing)
        self.assertTrue(rules["harness_internals"].search("ls /opt/codex/bin"))
        self.assertTrue(rules["harness_internals"].search("zeroshot list"))
        self.assertTrue(rules["sudo"].search("sudo apt-get install foo"))
        self.assertFalse(rules["sudo"].search("echo pseudo"))

    def test_commands_are_the_scripts_codex_ran(self):
        def completed(script, output=""):
            return {"type": "event_msg", "payload": {"type": "item_completed", "item": {"type": "CommandExecution", "command": ["/bin/bash", "-lc", script], "aggregated_output": output}}}

        def js(code):
            return {"type": "response_item", "payload": {"type": "custom_tool_call", "input": code}}

        records = [
            {"type": "response_item", "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Authored instructions:\n" + config.prompt("builder")}]}},
            {"type": "event_msg", "payload": {"type": "task_started"}},
            # Code-mode JavaScript: identifiers such as r2 or strings are not commands.
            js('const r2 = await tools.exec_command({cmd:"./executable --help"}); const strings = [1];'),
            completed("./executable --help", "usage"),
            js('await tools.exec_command({cmd:"strings ./executable | head"})'),
            completed("strings ./executable | head", "strings: ./executable: Permission denied"),
            # Never reported as completed: still audited from the call itself.
            js("await tools.exec_command({cmd:'cat /proc/1/environ'})"),
            {"type": "event_msg", "payload": {"type": "item_completed", "item": {"type": "FileChange", "changes": {"/tmp/dump.c": {"type": "add", "content": "long r = ptrace(PTRACE_PEEKTEXT, pid, 0, 0);"}}}}},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            archive = _tar(Path(tmp, "t.tar.gz"), {".codex/sessions/rollout-a.jsonl": "\n".join(json.dumps(r) for r in records).encode()})
            result = audit.command_audit(archive)
        self.assertEqual(result["commands"], 3)
        self.assertEqual(result["rule_counts"]["reference_binary_analysis"], 1)
        self.assertEqual(result["rule_counts"]["reference_binary_analysis_denied"], 1)
        self.assertEqual(result["rule_counts"]["process_environment_read"], 1)
        self.assertEqual(result["rule_counts"]["binary_instrumentation"], 1)
        self.assertEqual(result["rule_counts_by_turn"]["build.turn1"]["process_environment_read"], 1)

    def test_loaded_agents_md_is_detected(self):
        # The two records Codex 0.155.0 writes when it loads a workspace AGENTS.md.
        world = {"type": "world_state", "payload": {"full": True, "state": {"agents_md": {"directory": "/workspace", "text": "x"}}}}
        message = {"type": "response_item", "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "# AGENTS.md instructions for /workspace\n\n<INSTRUCTIONS>x"}]}}
        user_level = {"type": "response_item", "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "# AGENTS.md instructions\n\n<INSTRUCTIONS>x"}]}}
        self.assertTrue(audit.loaded_agents_md(world))
        self.assertTrue(audit.loaded_agents_md(message))
        self.assertTrue(audit.loaded_agents_md(user_level))
        self.assertFalse(audit.loaded_agents_md({"type": "world_state", "payload": {"full": True, "state": {"agents_md": {}}}}))

    def test_turns_continue_across_builder_threads(self):
        # A failed build makes Zeroshot start a new builder thread; its first turn is round 2.
        def session(script):
            return "\n".join(json.dumps(r) for r in [
                {"type": "response_item", "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Authored instructions:\n" + config.prompt("builder")}]}},
                {"type": "event_msg", "payload": {"type": "task_started"}},
                {"type": "event_msg", "payload": {"type": "item_completed", "item": {"type": "CommandExecution", "command": ["/bin/bash", "-lc", script], "aggregated_output": ""}}},
            ]).encode()

        with tempfile.TemporaryDirectory() as tmp:
            archive = _tar(Path(tmp, "t.tar.gz"), {
                ".codex/sessions/2026/09/24/rollout-2026-09-24T02-00-00-b.jsonl": session("zeroshot list"),
                ".codex/sessions/2026/09/24/rollout-2026-09-24T01-00-00-a.jsonl": session("ls"),
            })
            result = audit.command_audit(archive)
        self.assertEqual(result["rule_counts_by_turn"], {"build.turn2": {"harness_internals": 1}})

    def test_git_objects_scan_tolerates_absolute_links(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp, "ws")
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            (repo / "a.txt").write_text("marker-123")
            subprocess.run(["git", "-C", str(repo), "add", "a.txt"], check=True)
            subprocess.run(["git", "-C", str(repo), "-c", "user.email=a@b", "-c", "user.name=n", "commit", "-qm", "x"], check=True)
            (repo / ".venv" / "bin").mkdir(parents=True)
            os.symlink("/usr/bin/python3", repo / ".venv" / "bin" / "python3")
            archive = Path(tmp, "ws.tar.gz")
            subprocess.run(["tar", "-czf", str(archive), "-C", str(repo), "."], check=True)
            self.assertIn(b"marker-123", audit._git_objects(archive))

    def test_build_artifacts_are_not_source_edits(self):
        for path in ("__pycache__/svgbob.cpython-310.pyc", "target/release/foo", "src/x.o"):
            self.assertTrue(audit.BUILD_ARTIFACT.search(path), path)
        for path in ("svgbob.py", "src/main.rs", "compile.sh", "targets.txt"):
            self.assertFalse(audit.BUILD_ARTIFACT.search(path), path)

    def test_proxy_log_parsing(self):
        log = "\n".join([
            'CONNECT   Sep 24 01:59:05.430 [1]: Request (file descriptor 4): CONNECT example.com:443 HTTP/1.1',
            'NOTICE    Sep 24 01:59:05.430 [1]: Proxying refused on filtered domain "example.com"',
            'CONNECT   Sep 24 01:59:05.606 [1]: Request (file descriptor 4): CONNECT api.openai.com:443 HTTP/1.1',
            'CONNECT   Sep 24 01:59:05.610 [1]: Established connection to host "api.openai.com" using file descriptor 5.',
        ])
        parsed = audit.proxy_audit(log)
        self.assertEqual(parsed["established"], {"api.openai.com": 1})
        self.assertEqual(parsed["refused"], {"example.com": 1})

    def test_secret_scan_finds_key_in_archives_nested_archives_and_git_objects(self):
        fake = "sk-test-" + "x" * 40
        os.environ["OPENAI_API_KEY"] = fake
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp, "results")
                root.mkdir()
                _tar(root / "a.tar.gz", {"inner.txt": f"leaked {fake}".encode()})
                inner = _tar(Path(tmp, "inner.tar.gz"), {"deep.txt": fake.encode()})
                _tar(root / "b.tar.gz", {"nested.tar.gz": inner.read_bytes()})
                repo = Path(tmp, "repo")
                subprocess.run(["git", "init", "-q", str(repo)], check=True)
                (repo / "f.txt").write_text(fake)
                subprocess.run(["git", "-C", str(repo), "add", "f.txt"], check=True)
                subprocess.run(["git", "-C", str(repo), "-c", "user.email=a@b", "-c", "user.name=a", "commit", "-qm", "x"], check=True)
                (repo / "f.txt").unlink()
                with tarfile.open(root / "c.tar.gz", "w:gz") as tar:
                    tar.add(repo, arcname=".")
                (root / "clean.txt").write_text("nothing here")
                hits = audit.secret_scan(root)["literal_key_hits"]
        finally:
            del os.environ["OPENAI_API_KEY"]
        self.assertIn("a.tar.gz!inner.txt", hits)
        self.assertIn("b.tar.gz!nested.tar.gz!deep.txt", hits)
        self.assertIn("c.tar.gz!(git objects)", hits)
        self.assertFalse(any(h.startswith("clean") for h in hits))


class DecisionTests(unittest.TestCase):
    def _loop(self, label, first, final, **extra):
        record = {
            "label": label, "arm": "loop", "state": "complete", "build_outcomes": ["verified"], "snapshots": {},
            "rounds": {"build-1": {"score": first / 472, "passed": first, "scored_tests": 472}, "final": {"score": final / 472, "passed": final, "scored_tests": 472}},
            "commands": {"rule_counts": {}, "rule_counts_by_turn": {}}, "reference_copies_in_final": [], "codex_config_unchanged": True, "harness_unchanged": True,
            "gain_tests": final - first,
        }
        record.update(extra)
        record["ineligible_reasons"] = report._eligibility(record, 472)
        return record

    def test_supported_needs_every_run_to_clear_noise_and_a_large_median(self):
        runs = [self._loop(f"0{i}", 200, 200 + g) for i, g in enumerate([30, 25, 40, 28, 5])]
        self.assertEqual(report._decision(runs, 472, EXPERIMENT)["verdict"], "supported")
        runs[-1] = self._loop("09", 200, 202)  # +2 tests < 1 pp
        self.assertEqual(report._decision(runs, 472, EXPERIMENT)["verdict"], "inconclusive")

    def test_not_supported_and_incomplete(self):
        flat = [self._loop(f"0{i}", 200, 203) for i in range(5)]
        self.assertEqual(report._decision(flat, 472, EXPERIMENT)["verdict"], "not supported")
        flat[0] = self._loop("00", 200, 260, build_outcomes=["timeout"])
        self.assertTrue(report._decision(flat, 472, EXPERIMENT)["verdict"].startswith("inconclusive (only 4 of 5"))

    def test_discarded_attempts_are_reported_not_scored(self):
        with tempfile.TemporaryDirectory() as tmp:
            discarded = Path(tmp, "attempts", "03-loop.discarded-1700000000")
            discarded.mkdir(parents=True)
            (discarded / "attempt.json").write_text(json.dumps({"label": "03-loop", "state": "error", "error": "RuntimeError: boom", "wall_seconds": 12}))
            entries = report._discarded(Path(tmp), EXPERIMENT.pricing)
        self.assertEqual(entries, [{"directory": "03-loop.discarded-1700000000", "label": "03-loop", "state": "error", "error": "RuntimeError: boom", "force_stopped": None, "wall_seconds": 12, "cost_usd": None}])

    def test_files_left_in_codex_home_make_a_run_ineligible(self):
        meta = {"codex_home_surfaces": [], "snapshots": {"build-1": {"codex_home_surfaces": ["./AGENTS.md 0123456789abcdef"]}}, "codex_home_surfaces_end": []}
        self.assertEqual(report.codex_home_changes(meta), ["./AGENTS.md 0123456789abcdef"])
        run = self._loop("01", 200, 260, codex_home_changes=report.codex_home_changes(meta))
        self.assertTrue(any(r.startswith("a node left files for later Codex sessions") for r in run["ineligible_reasons"]))

    def test_building_the_reference_makes_a_run_ineligible(self):
        run = self._loop("01", 200, 260, reference_sha256="ab" * 32)
        run["rounds"]["final"]["executable_hash"] = "ab" * 32
        self.assertIn("final built executable is the reference", report._eligibility(run, 472))

    def test_disqualifying_audit_makes_a_run_ineligible(self):
        run = self._loop("01", 200, 260, commands={"rule_counts": {"process_environment_read": 1}, "rule_counts_by_turn": {}})
        self.assertIn("audit: process_environment_read", run["ineligible_reasons"])


class EvalTests(unittest.TestCase):
    def test_targets_skip_discarded_attempts(self):
        from bench import evaluate

        with tempfile.TemporaryDirectory() as tmp:
            for name in ("01-loop", "01-loop.discarded-123"):
                d = Path(tmp, "attempts", name, "snapshots")
                d.mkdir(parents=True)
                _tar(d.parent / "submission.tar.gz", {"a": b"1"})
                _tar(d / "build-1.tar.gz", {"a": b"1"})
            labels = [label for label, _ in evaluate.targets(Path(tmp))]
        self.assertEqual(labels, ["01-loop__final", "01-loop__build-1"])

    def test_rerun_plugin_is_pinned(self):
        try:
            from bench import pbeval
        except ImportError:
            self.skipTest("programbench not installed")
        seen = []
        original = pbeval._run_step
        pbeval._run_step = lambda self, command, **kw: seen.append(command)
        try:
            pbeval._pinned_run_step(None, "pip3 install -q --disable-pip-version-check pytest-rerunfailures", env=None)
        finally:
            pbeval._run_step = original
        self.assertEqual(seen, ["pip3 install -q --disable-pip-version-check pytest-rerunfailures==16.4"])


if __name__ == "__main__":
    unittest.main()
