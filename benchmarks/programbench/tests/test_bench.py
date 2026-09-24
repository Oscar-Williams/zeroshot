"""Unit tests that need neither Docker nor an API key: ``python -m unittest discover tests``."""

from __future__ import annotations

import copy
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench import accounting, audit, config, graphs  # noqa: E402
from bench.attempt import completion_of, event_of, run_files  # noqa: E402

EXPERIMENT = config.load("experiments/luna-xhigh-svgbob.json")
LUNA = EXPERIMENT.pricing


class GraphTests(unittest.TestCase):
    def test_builder_node_is_identical_across_arms(self):
        loop = run_files(EXPERIMENT, "loop")["graph.json"]
        single = run_files(EXPERIMENT, "single")["graph.json"]
        self.assertEqual(loop["root"]["children"][0]["body"]["children"][0], single["root"]["children"][0])

    def test_prompts_are_frozen_text(self):
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
        check = loop["body"]["children"][1]
        self.assertEqual(check["writeBindings"][0]["target"], ["feedback"])
        self.assertEqual(loop["promotedStatePaths"], [["feedback"]])

    def test_runtime_sessions(self):
        plan = graphs.runtime_plan("loop", "m", "xhigh")
        self.assertEqual(plan["nodes"]["build"]["sessionScope"], "node_instance")
        self.assertEqual(plan["nodes"]["check"]["sessionScope"], "execution")
        self.assertEqual(set(graphs.runtime_plan("single", "m", "xhigh")["nodes"]), {"build"})

    def test_task_statement_matches_upstream_minus_scaffold(self):
        task = config.prompt("task")
        self.assertIn("Make sure that you have a `./compile.sh` file", task)
        for scaffold in ("COMPLETE_TASK_AND_SUBMIT", "bash tool", "system_information", "helpful assistant"):
            self.assertNotIn(scaffold, task)


class CodexConfigTests(unittest.TestCase):
    def test_rendered_config_keeps_credentials_and_web_out(self):
        from bench import images

        rendered = images.codex_config({"PATH": "/usr/bin", "CARGO_HOME": "/usr/local/cargo", "HOME": "/root"})
        self.assertIn('web_search = "disabled"', rendered)
        self.assertIn("shell_snapshot = false", rendered)
        self.assertIn("ignore_default_excludes = false", rendered)
        self.assertIn('"*KEY*"', rendered)
        self.assertIn('CARGO_HOME = "/usr/local/cargo"', rendered)
        self.assertNotIn("HOME = \"/root\"", rendered)
        import tomllib

        parsed = tomllib.loads(rendered)
        self.assertFalse(parsed["features"]["shell_snapshot"])
        self.assertEqual(parsed["shell_environment_policy"]["set"]["CARGO_HOME"], "/usr/local/cargo")


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
        self.assertEqual(set(order[:2]), {"loop", "single"})  # both arms start in the first wave
        self.assertFalse(any(a == b == "single" for a, b in zip(order, order[1:], strict=False)))
        self.assertFalse(any(order[i : i + 3] == ["loop"] * 3 for i in range(len(order) - 2)))


class AccountingTests(unittest.TestCase):
    def test_cost_splits_cache_reads_and_writes(self):
        tokens = {"inputTokens": 1_000_000, "cacheReadInputTokens": 900_000, "cacheCreationInputTokens": 50_000, "outputTokens": 10_000}
        expected = (50_000 * 0.2 + 900_000 * 0.02 + 50_000 * 0.25 + 10_000 * 1.2) / 1e6
        self.assertAlmostEqual(accounting.cost(tokens, LUNA), expected)

    def test_usage_by_node(self):
        events = [
            {"kind": "node_started", "reference": {"node": "build", "execution": 1}},
            {"kind": "token_usage_observed", "execution": 1, "usage": {"inputTokens": 10, "outputTokens": 2, "cacheReadInputTokens": 5, "cacheCreationInputTokens": 1}},
            {"kind": "node_started", "reference": {"node": "check", "execution": 2}},
            {"kind": "token_usage_observed", "execution": 2, "usage": {"inputTokens": 7, "outputTokens": 1}},
            {"kind": "token_usage_observed", "execution": 1, "usage": {"inputTokens": 3, "outputTokens": 1}},
        ]
        usage = accounting.usage_from_events(events)
        self.assertEqual(usage["build"]["inputTokens"], 13)
        self.assertEqual(usage["check"]["inputTokens"], 7)
        self.assertEqual(usage["total"]["outputTokens"], 4)

    def test_ledger_rounds(self):
        from bench import ledger

        events = [
            {"kind": "node_completed", "completion": {"reference": {"node": "build"}, "outcome": {"status": "verified"}}},
            {"kind": "node_completed", "completion": {"reference": {"node": "check"}, "outcome": {"status": "verifier", "signals": {"verdict": "rejected"}, "diagnostic": {"message": "x differs"}}}},
            {"kind": "node_completed", "completion": {"reference": {"node": "build"}, "outcome": {"status": "error", "code": "timeout"}}},
        ]
        summary = ledger.rounds(events)
        self.assertEqual([b["code"] for b in summary["builds"]], [None, "timeout"])
        self.assertEqual(summary["checks"][0]["verdict"], "rejected")
        self.assertEqual(summary["checks"][0]["message"], "x differs")


class EventTests(unittest.TestCase):
    def test_event_and_completion_parsing(self):
        record = {"cursor": "v2:9", "event": {"kind": "node_completed", "completion": {"reference": {"node": "check", "nodeInstance": 2}, "outcome": {"signals": {"verdict": "rejected"}}}}}
        event = event_of(record)
        self.assertEqual(event["kind"], "node_completed")
        self.assertEqual(completion_of(event), ("check", 2, {"signals": {"verdict": "rejected"}}))
        self.assertIsNone(event_of({"cursor": 1, "note": {"x": 1}}))


class AuditTests(unittest.TestCase):
    def test_command_rules(self):
        rules = audit.COMMAND_RULES
        self.assertTrue(rules["reference_binary_analysis"].search("objdump -d ./executable"))
        self.assertFalse(rules["reference_binary_analysis"].search("./executable --help | strings-like-name"))
        self.assertTrue(rules["reference_binary_moved_or_copied"].search("cp ./executable /tmp/x"))
        self.assertFalse(rules["reference_binary_moved_or_copied"].search("./executable -s 'x' > out.svg"))
        self.assertTrue(rules["network_fetch"].search("cargo install svgbob_cli"))
        self.assertTrue(rules["sudo"].search("sudo apt-get install foo"))
        self.assertFalse(rules["sudo"].search("echo pseudo"))

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

    def test_secret_scan_finds_key_inside_archives(self):
        import io
        import tarfile

        fake = "sk-test-" + "x" * 40
        os.environ["OPENAI_API_KEY"] = fake
        try:
            with tempfile.TemporaryDirectory() as tmp:
                data = f"leaked {fake}".encode()
                with tarfile.open(Path(tmp, "a.tar.gz"), "w:gz") as tar:
                    info = tarfile.TarInfo("inner.txt")
                    info.size = len(data)
                    tar.addfile(info, io.BytesIO(data))
                Path(tmp, "clean.txt").write_text("nothing here")
                result = audit.secret_scan(Path(tmp))
        finally:
            del os.environ["OPENAI_API_KEY"]
        self.assertEqual(result["literal_key_hits"], ["a.tar.gz:inner.txt"])


if __name__ == "__main__":
    unittest.main()
