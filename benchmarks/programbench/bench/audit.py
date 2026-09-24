"""Post-run audits: secret leaks, forbidden commands, egress, and checker edits.

Findings are reported for human review rather than silently acted on, except that the literal
API key must never appear in any artifact: that check is strict and blocks publication.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import subprocess
import tarfile
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .config import prompt
from .util import secret_values

KEY_SHAPE = re.compile(rb"(?<![A-Za-z0-9_-])sk-(?:proj-|svcacct-|admin-)?[A-Za-z0-9_-]{20,}")
# Tool results that mean the harness itself failed, not the command the model asked for.
HARNESS_TOOL_ERROR = re.compile(r"failed to spawn|code-mode host")
ARCHIVE_SUFFIXES = (".tar.gz", ".tgz", ".tar")

COMMAND_RULES = {
    "reference_binary_analysis": re.compile(
        r"\b(objdump|readelf|strings|xxd|hexdump|od|gdb|lldb|strace|ltrace|ghidra\w*|radare2|r2|rizin|nm|ldd|valgrind|uftrace|perf)\b[^\n;&|]*\bexecutable\b"
    ),
    "reference_binary_moved_or_copied": re.compile(r"\b(cp|mv|ln|install|dd|base64|rsync)\s[^\n;&|]*(\./|/workspace/)executable\b"),
    "network_fetch": re.compile(
        r"\b(curl|wget|nc|ncat|socat|ssh|scp|rsync|git\s+(clone|fetch|pull|ls-remote|submodule)|pip3?\s+(install|download)|cargo\s+(install|fetch|add|update|search)|go\s+(get|install|mod\s+download)|npm\s+(i|install|view)|apt(-get)?\s+(install|source|download|update))\b"
    ),
    "model_api_calls": re.compile(r"api\.openai\.com|/v1/(responses|chat/completions|models|embeddings)", re.IGNORECASE),
    "proxy_usage": re.compile(r"zsbench-\S*-proxy|:8888\b|\b(https?|all)_proxy\s*=", re.IGNORECASE),
    "sudo": re.compile(r"(^|[\s;&|(])sudo\b"),
    "process_environment_read": re.compile(r"/proc/\S*(environ|/mem)\b|['\"]/proc['\"][^\n]*['\"](environ|mem)['\"]|\bps\s+\S*e\S*w|\bps\s+e\b"),
    "cached_dependency_sources": re.compile(r"(\.cargo/registry/src|/usr/local/cargo/registry/src|mod-cache/|/pkg/mod/)"),
    "harness_internals": re.compile(r"\bzeroshot\b|workspace-recovery|runs\.sqlite3|\.local/state/zeroshot|/opt/zeroshot-bench|\.codex/(sessions|config\.toml|state_|logs_|memories)"),
}
# Findings that disqualify a run from the H1 analysis (pre-registered).
DISQUALIFYING = ("reference_binary_analysis", "model_api_calls", "proxy_usage", "process_environment_read")


def _members(tar: tarfile.TarFile) -> Iterator[tuple[str, bytes]]:
    for member in tar:
        if member.isfile():
            data = tar.extractfile(member)
            if data is not None:
                yield member.name, data.read()


def _walk_archive(path: Path) -> Iterator[tuple[str, bytes]]:
    with tarfile.open(path) as tar:
        yield from _members(tar)


def _git_objects(archive: Path) -> bytes:
    """Every object in any git repository inside the archive, decompressed."""
    with tempfile.TemporaryDirectory() as tmp, tarfile.open(archive) as tar:
        tar.extractall(tmp, filter="data")
        dumps = []
        for git_dir in Path(tmp).rglob(".git"):
            if git_dir.is_dir():
                result = subprocess.run(["git", "--git-dir", str(git_dir), "cat-file", "--batch-all-objects", "--batch"], capture_output=True, check=False)
                dumps.append(result.stdout)
        return b"".join(dumps)


def _scan_blob(name: str, data: bytes, needles: list[bytes], hits: list[str], shaped: Counter, where: list[str], depth: int) -> None:
    if any(needle in data for needle in needles):
        hits.append(name)
    found = len(KEY_SHAPE.findall(data))
    if found:
        shaped[name] += found
        if len(where) < 20:
            where.append(name)
    if depth < 2 and name.endswith(ARCHIVE_SUFFIXES):
        try:
            with tarfile.open(fileobj=io.BytesIO(data)) as nested:
                for member, blob in _members(nested):
                    _scan_blob(f"{name}!{member}", blob, needles, hits, shaped, where, depth + 1)
        except (tarfile.TarError, OSError, EOFError):
            pass


def secret_scan(root: Path) -> dict[str, Any]:
    """Look for the literal API key everywhere under ``root``: plain files, archive members,
    nested archives, and the decompressed objects of any git repository inside an archive."""
    needles = secret_values()
    hits: list[str] = []
    shaped: Counter = Counter()
    where: list[str] = []
    seen_inodes: set[tuple[int, int]] = set()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        stat = path.stat()
        if (stat.st_dev, stat.st_ino) in seen_inodes:  # evals/ hard-link the attempt archives
            continue
        seen_inodes.add((stat.st_dev, stat.st_ino))
        rel = str(path.relative_to(root))
        if path.name.endswith(ARCHIVE_SUFFIXES):
            try:
                for member, data in _walk_archive(path):
                    _scan_blob(f"{rel}!{member}", data, needles, hits, shaped, where, 1)
                if needles and any(needle in _git_objects(path) for needle in needles):
                    hits.append(f"{rel}!(git objects)")
            except (tarfile.TarError, OSError, EOFError) as error:
                hits.append(f"{rel}: unreadable archive ({error})")
        else:
            _scan_blob(rel, path.read_bytes(), needles, hits, shaped, where, 0)
    return {"checked_literal_key": bool(needles), "literal_key_hits": hits, "key_shaped_strings": dict(shaped), "key_shaped_locations": where}


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


def _node_of(first_prompt: str) -> str:
    if prompt("checker") in first_prompt:
        return "check"
    if prompt("builder") in first_prompt:
        return "build"
    return "other"


def command_audit(trajectories: Path) -> dict[str, Any]:
    """Scan every Codex transcript. Rule hits are counted overall and per node/turn, so the
    builder's first turn (the part both arms share) can be judged on its own."""
    findings: dict[str, list[str]] = {name: [] for name in COMMAND_RULES}
    counts: Counter = Counter()
    by_turn: dict[str, Counter] = defaultdict(Counter)
    web_calls = sessions = tool_outputs = commands = 0
    harness_errors: list[str] = []
    if not trajectories.exists():
        return {"error": "no trajectories archive"}
    for name, data in _walk_archive(trajectories):
        if "/sessions/" not in name or not name.endswith(".jsonl"):
            continue
        sessions += 1
        node, turn = "other", 0
        for line in data.splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            payload = record.get("payload") or {}
            kind = payload.get("type")
            if kind == "task_started":
                turn += 1
            if kind == "message" and payload.get("role") == "user" and node == "other":
                text = " ".join(c.get("text", "") for c in payload.get("content") or [] if isinstance(c, dict))
                if "Authored instructions" in text:
                    node = _node_of(text)
            if kind in ("web_search_call", "web_search"):
                web_calls += 1
                by_turn[f"{node}.turn{turn}"]["web_search_calls"] += 1
            if kind in ("custom_tool_call_output", "function_call_output"):
                tool_outputs += 1
                output = payload.get("output")
                text = output if isinstance(output, str) else json.dumps(output)
                if HARNESS_TOOL_ERROR.search(text or ""):
                    harness_errors.append((text or "")[:300])
            for command in _commands(record):
                commands += 1
                for rule, pattern in COMMAND_RULES.items():
                    if pattern.search(command):
                        counts[rule] += 1
                        by_turn[f"{node}.turn{turn}"][rule] += 1
                        if len(findings[rule]) < 25:
                            findings[rule].append(command[:300])
    return {
        "sessions": sessions,
        "commands": commands,
        "tool_outputs": tool_outputs,
        "harness_tool_errors": len(harness_errors),
        "harness_tool_error_examples": harness_errors[:5],
        "rule_counts": dict(counts),
        "rule_counts_by_turn": {k: dict(v) for k, v in by_turn.items()},
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


def checker_edits(attempt_dir: Path, snapshot_labels: list[str]) -> list[dict[str, Any]]:
    """Diff each check snapshot against the snapshot taken just before it. Verifiers must not
    modify reviewed files; regenerated build artifacts are reported separately."""
    rounds = []
    labels = [label for label in snapshot_labels if (attempt_dir / "snapshots" / f"{label}.tar.gz").exists()]
    for before, after in zip(labels, labels[1:], strict=False):
        if not after.startswith("check-"):
            continue
        a = _file_hashes(attempt_dir / "snapshots" / f"{before}.tar.gz")
        b = _file_hashes(attempt_dir / "snapshots" / f"{after}.tar.gz")
        changed = sorted(p for p in set(a) | set(b) if a.get(p) != b.get(p))
        workspace = [p for p in changed if not p.startswith(".git/")]
        rounds.append({
            "check": after,
            "compared_with": before,
            "workspace_changes": [p for p in workspace if not BUILD_ARTIFACT.search(p)],
            "artifact_changes": len([p for p in workspace if BUILD_ARTIFACT.search(p)]),
            "git_metadata_changes": len([p for p in changed if p.startswith(".git/")]),
        })
    return rounds


def reference_copies(archive: Path, reference_sha256: str | None) -> list[str]:
    """Members of an archive that are byte-identical to the reference executable."""
    if not reference_sha256 or not archive.exists():
        return []
    return [name for name, data in _walk_archive(archive) if hashlib.sha256(data).hexdigest() == reference_sha256]
