"""Render the two benchmark graphs.

Both arms share one byte-identical ``build`` node (name, worker, instructions, input type,
bindings, attempts, deadline). The single arm runs it once. The loop arm follows it with an
independent ``check`` verifier whose diagnostic becomes the builder's ``feedback`` on the next
round. The builder's first round therefore receives exactly what the single arm receives.
"""

from __future__ import annotations

import copy
from typing import Any

WORKER_ERRORS = ["crash", "malformed", "refusal", "timeout"]
STRING = {"kind": "string"}


def _record(*names: str) -> dict[str, Any]:
    return {"kind": "record", "fields": {name: {"type": STRING, "required": True} for name in names}}


def _from_state(*names: str) -> list[dict[str, Any]]:
    return [{"target": [name], "value": {"source": "state", "path": [name]}} for name in names]


def _error(node: str) -> dict[str, Any]:
    return {"kind": "in", "value": {"name": node, "source": "error", "field": None}, "labels": WORKER_ERRORS}


def build_node(instructions: str, timeout_ms: int) -> dict[str, Any]:
    return {
        "kind": "step",
        "name": "build",
        "worker": "agent.builder@1",
        "instructions": instructions,
        "input": _record("feedback", "task"),
        "output": {"kind": "null"},
        "inputBindings": _from_state("task", "feedback"),
        "writeBindings": [],
        "attempts": 1,
        "timeoutMs": timeout_ms,
    }


def check_node(instructions: str, timeout_ms: int) -> dict[str, Any]:
    return {
        "kind": "verifier",
        "name": "check",
        "worker": "agent.checker@1",
        "instructions": instructions,
        "input": _record("task"),
        "output": {"kind": "null"},
        "inputBindings": _from_state("task"),
        "writeBindings": [
            {"value": {"node": "check", "channel": "diagnostic", "path": ["message"]}, "target": ["feedback"]}
        ],
        "attempts": 2,
        "signals": {"verdict": ["accepted", "rejected"]},
        "diagnostic": _record("message"),
        "timeoutMs": timeout_ms,
    }


def _graph(root: dict[str, Any]) -> dict[str, Any]:
    return {
        "profile": "openengine.graph.full/v1",
        "initialInput": _record("task"),
        "policy": {"policy": "policy.native-v2@1", "default": "deny"},
        "root": root,
    }


def single_graph(builder: str, build_timeout_ms: int) -> dict[str, Any]:
    state = _record("feedback", "task")
    return _graph(
        {
            "kind": "seq",
            "name": "run",
            "state": state,
            "children": [
                build_node(builder, build_timeout_ms),
                {
                    "kind": "choice",
                    "name": "build_result",
                    "state": copy.deepcopy(state),
                    "branches": [
                        {"when": _error("build"), "node": {"kind": "fail", "name": "build_failed", "reason": "build_failed"}}
                    ],
                    "otherwise": {"kind": "succeed", "name": "done", "output": {"kind": "null"}, "bindings": []},
                    "promotedStatePaths": [],
                },
            ],
            "promotedStatePaths": [],
        }
    )


def loop_graph(builder: str, checker: str, max_iterations: int, build_timeout_ms: int, check_timeout_ms: int) -> dict[str, Any]:
    """Do-while ``build -> check`` until the check accepts or the round cap is reached.

    A build error does not stop a round: the check still judges the workspace as it stands. A
    check error leaves the previous feedback in place and the loop continues. The run succeeds only
    when a check accepted; otherwise it ends as ``unverified``. Either way the final workspace is
    what gets scored.
    """
    state = _record("feedback", "task")
    feedback = [["feedback"]]
    return _graph(
        {
            "kind": "seq",
            "name": "run",
            "state": state,
            "children": [
                {
                    "kind": "loop",
                    "name": "solve",
                    "state": copy.deepcopy(state),
                    "body": {
                        "kind": "seq",
                        "name": "round",
                        "state": copy.deepcopy(state),
                        "children": [build_node(builder, build_timeout_ms), check_node(checker, check_timeout_ms)],
                        "promotedStatePaths": copy.deepcopy(feedback),
                    },
                    "until": {"kind": "in", "value": {"name": "check", "source": "signal", "field": "verdict"}, "labels": ["accepted"]},
                    "maxIterations": max_iterations,
                    "promotedStatePaths": copy.deepcopy(feedback),
                },
                {
                    "kind": "choice",
                    "name": "outcome",
                    "state": copy.deepcopy(state),
                    "branches": [
                        {
                            "when": {"kind": "in", "value": {"name": "solve", "source": "group", "field": "terminated"}, "labels": ["converged"]},
                            "node": {"kind": "succeed", "name": "accepted", "output": {"kind": "null"}, "bindings": []},
                        }
                    ],
                    "otherwise": {"kind": "fail", "name": "unverified", "reason": "unverified"},
                    "promotedStatePaths": [],
                },
            ],
            "promotedStatePaths": [],
        }
    )


def runtime_plan(arm: str, model: str, effort: str) -> dict[str, Any]:
    """Exact runtime plan. The builder resumes its own session across rounds; each check is fresh."""
    nodes = {"build": {"kind": "agent", "model": model, "effort": effort, "sessionScope": "node_instance"}}
    if arm == "loop":
        nodes["check"] = {"kind": "agent", "model": model, "effort": effort, "sessionScope": "execution"}
    return {"harness": "codex", "provider": "openai", "size": "medium", "nodes": nodes}
