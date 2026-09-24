"""Post-run audits: secret leaks, forbidden commands, egress, and checker edits.

Findings are reported, not silently acted on: a human decides whether a flagged command was a
rule violation. The literal API key must never appear in any artifact; that check is strict.
"""

from __future__ import annotations

import hashlib
import json
import re
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

from .util import secret_values

KEY_SHAPE = re.compile(rb"(?<![A-Za-z0-9_-])sk-(?:proj-|svcacct-|admin-)?[A-Za-z0-9_-]{20,}")
# Tool results that mean the harness itself failed, not the command the model asked for.
HARNESS_TOOL_ERROR = re.compile(r"failed to spawn|code-mode host")

COMMAND_RULES = {
    "reference_binary_analysis": re.compile(
        r"\b(objdump|readelf|strings|xxd|hexdump|od|gdb|lldb|strace|ltrace|ghidra\w*|radare2|r2|rizin|nm|ldd|valgrind|uftrace|perf)\b[^\n;&|]*\bexecutable\b"
    ),
    "reference_binary_moved_or_copied": re.compile(r"\b(cp|mv|ln|install|dd|base64|rsync)\s[^\n;&|]*(\./|/workspace/)executable\b"),
    "network_fetch": re.compile(
        r"\b(curl|wget|nc|ncat|socat|ssh|scp|rsync|git\s+(clone|fetch|pull|ls-remote|submodule)|pip3?\s+(install|download)|cargo\s+(install|fetch|add|update|search)|go\s+(get|install|mod\s+download)|npm\s+(i|install|view)|apt(-get)?\s+(install|source|download|update))\b"
    ),
    "sudo": re.compile(r"(^|[\s;&|(])sudo\b"),
    "cached_dependency_sources": re.compile(r"(\.cargo/registry/src|/usr/local/cargo/registry/src|mod-cache/|/pkg/mod/)"),
    "harness_internals": re.compile(r"(\.local/state/zeroshot|/opt/zeroshot-bench|\.codex/sessions)"),
    "process_environment_read": re.compile(r"/proc/\S*environ"),
}


def _walk_archive(path: Path) -> Iterator[tuple[str, bytes]]:
    with tarfile.open(path) as tar:
        for member in tar:
            if member.isfile():
                data = tar.extractfile(member)
                if data is not None:
                    yield member.name, data.read()


def secret_scan(root: Path) -> dict[str, Any]:
    """Look for the literal API key everywhere under ``root``, including inside archives."""
    needles = secret_values()
    hits, shaped = [], Counter()
    shaped_where: list[str] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = str(path.relative_to(root))
        blobs: Iterator[tuple[str, bytes]]
        if path.name.endswith((".tar.gz", ".tgz")):
            try:
                blobs = _walk_archive(path)
            except (tarfile.TarError, OSError, EOFError):
                continue
        else:
            blobs = iter([("", path.read_bytes())])
        for member, data in blobs:
            if any(needle in data for needle in needles):
                hits.append(f"{rel}:{member}" if member else rel)
            found = len(KEY_SHAPE.findall(data))
            if found:
                shaped[rel] += found
                if len(shaped_where) < 20:
                    shaped_where.append(f"{rel}:{member}" if member else rel)
    return {
        "checked_literal_key": bool(needles),
        "literal_key_hits": hits,
        "key_shaped_strings": {k: v for k, v in shaped.items() if v},
        "key_shaped_locations": shaped_where,
    }


def _commands(record: dict[str, Any]) -> Iterator[str]:
    payload = record.get("payload") or {}
    kind = payload.get("type")
    if kind == "function_call":
        try:
            args = json.loads(payload.get("arguments") or "{}")
        except json.JSONDecodeError:
            yield str(payload.get("arguments"))
            return
        for key in ("cmd", "command", "script", "input"):
            value = args.get(key)
            if isinstance(value, list):
                yield " ".join(map(str, value))
            elif isinstance(value, str):
                yield value
    elif kind == "custom_tool_call":
        yield str(payload.get("input") or "")
    elif kind == "local_shell_call":
        action = payload.get("action") or {}
        command = action.get("command")
        yield " ".join(command) if isinstance(command, list) else str(command or "")


def command_audit(trajectories: Path) -> dict[str, Any]:
    findings: dict[str, list[str]] = {name: [] for name in COMMAND_RULES}
    counts = Counter()
    web_calls = 0
    sessions = 0
    tool_outputs = 0
    harness_errors: list[str] = []
    if not trajectories.exists():
        return {"error": "no trajectories archive"}
    for name, data in _walk_archive(trajectories):
        if "/sessions/" not in name or not name.endswith(".jsonl"):
            continue
        sessions += 1
        for line in data.splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            payload = record.get("payload") or {}
            if payload.get("type") in ("web_search_call", "web_search"):
                web_calls += 1
            if payload.get("type") in ("custom_tool_call_output", "function_call_output"):
                tool_outputs += 1
                output = payload.get("output")
                text = output if isinstance(output, str) else json.dumps(output)
                if HARNESS_TOOL_ERROR.search(text or ""):
                    harness_errors.append((text or "")[:300])
            for command in _commands(record):
                counts["commands"] += 1
                for rule, pattern in COMMAND_RULES.items():
                    if pattern.search(command):
                        counts[rule] += 1
                        if len(findings[rule]) < 25:
                            findings[rule].append(command[:300])
    return {
        "sessions": sessions,
        "commands": counts.pop("commands", 0),
        "tool_outputs": tool_outputs,
        "harness_tool_errors": len(harness_errors),
        "harness_tool_error_examples": harness_errors[:5],
        "rule_counts": dict(counts),
        "web_search_calls": web_calls,
        "examples": {k: v for k, v in findings.items() if v},
    }


def proxy_audit(log_text: str) -> dict[str, Any]:
    """Summarize tinyproxy's log: hosts it connected to, and requests it refused."""
    requested, established, refused = Counter(), Counter(), Counter()
    for line in log_text.splitlines():
        if match := re.search(r"Request \(file descriptor \d+\): (\S+) (\S+)", line):
            requested[f"{match.group(1)} {match.group(2)}"] += 1
        elif match := re.search(r'Established connection to host "([^"]+)"', line):
            established[match.group(1)] += 1
        elif match := re.search(r'(?:Proxying refused on filtered (?:domain|url)|refused)\s*"?([^"\s]+)"?', line, re.IGNORECASE):
            refused[match.group(1)] += 1
    return {"established": dict(established), "refused": dict(refused), "requests": dict(requested)}


def _file_hashes(path: Path) -> dict[str, str]:
    hashes = {}
    with tarfile.open(path) as tar:
        for member in tar:
            if member.isfile():
                data = tar.extractfile(member)
                hashes[member.name.removeprefix("./")] = hashlib.sha256(data.read()).hexdigest() if data else ""
            elif member.issym():
                hashes[member.name.removeprefix("./")] = "symlink:" + member.linkname
    return hashes


# Paths a checker legitimately regenerates by running or building the candidate.
BUILD_ARTIFACT = re.compile(r"(^|/)(__pycache__|target|build|dist|node_modules|\.pytest_cache|\.mypy_cache)/|\.(pyc|pyo|o|a|so|d|rlib|rmeta)$")


def checker_edits(attempt_dir: Path, checks: int) -> list[dict[str, Any]]:
    """Diff the workspace before and after each check. Verifiers must not modify reviewed files;
    regenerated build artifacts are reported separately from source changes."""
    rounds = []
    for n in range(1, checks + 1):
        before, after = attempt_dir / "snapshots" / f"build-{n}.tar.gz", attempt_dir / "snapshots" / f"check-{n}.tar.gz"
        if not (before.exists() and after.exists()):
            continue
        a, b = _file_hashes(before), _file_hashes(after)
        changed = sorted(p for p in set(a) | set(b) if a.get(p) != b.get(p))
        workspace = [p for p in changed if not p.startswith(".git/")]
        rounds.append({
            "round": n,
            "workspace_changes": [p for p in workspace if not BUILD_ARTIFACT.search(p)],
            "artifact_changes": len([p for p in workspace if BUILD_ARTIFACT.search(p)]),
            "git_metadata_changes": len([p for p in changed if p.startswith(".git/")]),
        })
    return rounds
