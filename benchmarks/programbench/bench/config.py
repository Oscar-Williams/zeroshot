"""Experiment configuration."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import ROOT

ARMS = ("loop", "single")
EFFORTS = ("low", "medium", "high", "xhigh", "max")


@dataclass(frozen=True)
class AttemptSpec:
    index: int
    arm: str

    @property
    def label(self) -> str:
        return f"{self.index:02d}-{self.arm}"


@dataclass(frozen=True)
class Experiment:
    path: Path
    raw: dict[str, Any]

    @property
    def id(self) -> str:
        return self.raw["id"]

    @property
    def instance_id(self) -> str:
        return self.raw["task"]["instance_id"]

    @property
    def task_image(self) -> str:
        return self.raw["task"]["image"]

    @property
    def model(self) -> str:
        return self.raw["model"]["id"]

    @property
    def effort(self) -> str:
        return self.raw["model"]["effort"]

    @property
    def limits(self) -> dict[str, int]:
        return self.raw["limits"]

    @property
    def resources(self) -> dict[str, Any]:
        return self.raw["resources"]

    @property
    def max_iterations(self) -> int:
        return self.raw["arms"]["loop"]["max_iterations"]

    @property
    def pricing(self) -> dict[str, Any]:
        return self.raw["pricing"]

    def attempts(self) -> list[AttemptSpec]:
        return [AttemptSpec(index + 1, arm) for index, arm in enumerate(self.raw["order"])]

    def digest(self) -> str:
        """Hash of everything that defines the experiment's behavior."""
        h = hashlib.sha256()
        h.update(json.dumps(self.raw, sort_keys=True).encode())
        for name in ("prompts/builder.md", "prompts/checker.md", "prompts/task.md", "pins.json", "bench/graphs.py", "agent/Dockerfile", "agent/codex-config.toml.in", "proxy/Dockerfile", "proxy/tinyproxy.conf", "proxy/filter"):
            h.update(name.encode())
            h.update((ROOT / name).read_bytes())
        return h.hexdigest()


def load(path: str | Path) -> Experiment:
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = ROOT / path
    raw = json.loads(path.read_text())
    _validate(raw)
    return Experiment(path.resolve(), raw)


def _validate(raw: dict[str, Any]) -> None:
    if not re.fullmatch(r"[a-z0-9][a-z0-9.-]{0,62}", raw["id"]):
        raise ValueError("experiment id must be lowercase letters, digits, dots and dashes")
    if "@sha256:" not in raw["task"]["image"]:
        raise ValueError("task image must be pinned by digest")
    if raw["model"]["effort"] not in EFFORTS:
        raise ValueError(f"effort must be one of {EFFORTS}")
    arms = raw["arms"]
    if set(arms) - set(ARMS):
        raise ValueError(f"unknown arms: {set(arms) - set(ARMS)}")
    order = raw["order"]
    for arm in ARMS:
        if arm in arms and order.count(arm) != arms[arm]["repeats"]:
            raise ValueError(f"order lists {order.count(arm)} {arm} runs but repeats is {arms[arm]['repeats']}")
    if set(order) - set(arms):
        raise ValueError("order names an arm that is not configured")
    limits = raw["limits"]
    for key in ("attempt_seconds", "build_timeout_ms", "check_timeout_ms"):
        if int(limits[key]) <= 0:
            raise ValueError(f"{key} must be positive")
    if "loop" in arms and not 1 <= arms["loop"]["max_iterations"] <= 100:
        raise ValueError("max_iterations must be between 1 and 100")
    prices = raw["pricing"]["usd_per_million_tokens"]
    for key in ("input", "cached_input", "cache_write", "output"):
        if float(prices[key]) < 0:
            raise ValueError("prices must be non-negative")


def pins() -> dict[str, Any]:
    return json.loads((ROOT / "pins.json").read_text())


def prompt(name: str) -> str:
    return (ROOT / "prompts" / f"{name}.md").read_text().strip()
