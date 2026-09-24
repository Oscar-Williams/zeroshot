"""Read a run's durable Zeroshot ledger (``runs.sqlite3``) from an attempt's trajectories archive."""

from __future__ import annotations

import json
import sqlite3
import tarfile
import tempfile
from pathlib import Path
from typing import Any


def events(trajectories: Path) -> list[dict[str, Any]] | None:
    """Every durable event in sequence order, or ``None`` if the archive has no single ledger."""
    if not trajectories.exists():
        return None
    with tarfile.open(trajectories) as tar:
        members = {m.name: m for m in tar.getmembers()}
        ledgers = [name for name in members if name.endswith("runs.sqlite3")]
        if len(ledgers) != 1:
            return None
        with tempfile.TemporaryDirectory() as tmp:
            for name in (ledgers[0], ledgers[0] + "-wal", ledgers[0] + "-shm", ledgers[0] + "-journal"):
                if name in members:
                    tar.extract(members[name], tmp, filter="data")
            db = sqlite3.connect(Path(tmp) / ledgers[0])
            try:
                rows = db.execute("select event_json from v2_run_events order by sequence").fetchall()
            except sqlite3.DatabaseError:  # a copy torn mid-write: treat as missing
                return None
            finally:
                db.close()
    return [json.loads(row[0]) for row in rows]


def rounds(ledger_events: list[dict[str, Any]]) -> dict[str, Any]:
    """Build and check executions in order, with outcomes and the checker's verdicts."""
    builds, checks = [], []
    for event in ledger_events:
        if event.get("kind") != "node_completed":
            continue
        completion = event.get("completion") or {}
        node = (completion.get("reference") or {}).get("node")
        outcome = completion.get("outcome") or {}
        entry = {"status": outcome.get("status"), "code": outcome.get("code"), "reason": outcome.get("reason")}
        if node == "build":
            builds.append(entry)
        elif node == "check":
            diagnostic = outcome.get("diagnostic") or {}
            entry["verdict"] = (outcome.get("signals") or {}).get("verdict")
            entry["message"] = str(diagnostic.get("message", ""))[:8000] if isinstance(diagnostic, dict) else None
            checks.append(entry)
    return {"builds": builds, "checks": checks}
