#!/usr/bin/env python3
"""Derive the benchmark task statement from the upstream mini-SWE-agent ProgramBench prompt.

The upstream prompt mixes the benchmark's task and rules with mini-SWE-agent scaffold mechanics
(its one-bash-command-per-turn protocol, its submit sentinel, command examples, and a system
information block). Zeroshot drives Codex, which brings its own tool protocol, so only the
scaffold mechanics are removed. Every remaining line is copied verbatim.

Usage: build_task_prompt.py UPSTREAM_YAML OUTPUT_MD
"""

import re
import sys
from pathlib import Path

PERSONA = "You are a helpful assistant that can interact with a computer.\n"
SCAFFOLD_SECTIONS = "## Command Execution Rules"


def block(yaml_text: str, key: str) -> str:
    """Return the literal block scalar ``key: |`` under ``agent:`` (two-space indent)."""
    match = re.search(rf"^  {key}: \|\n((?:    .*\n|\n)+)", yaml_text, re.MULTILINE)
    if not match:
        raise SystemExit(f"missing block {key!r}")
    lines = match.group(1).splitlines(keepends=True)
    return "".join(line[4:] if line.startswith("    ") else line for line in lines)


def main() -> None:
    upstream, output = Path(sys.argv[1]), Path(sys.argv[2])
    text = upstream.read_text()
    system = block(text, "system_template")
    instance = block(text, "instance_template")
    if not system.startswith(PERSONA):
        raise SystemExit("unexpected upstream system_template preamble")
    system = system[len(PERSONA) :].lstrip("\n")
    cut = instance.index(SCAFFOLD_SECTIONS)
    instance = instance[:cut].rstrip() + "\n"
    if "{{" in system + instance:
        raise SystemExit("template placeholder survived the extraction")
    output.write_text(system.rstrip() + "\n\n" + instance)


if __name__ == "__main__":
    main()
