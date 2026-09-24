"""Logging, subprocess, download, and secret helpers.

Secret values never appear in argv: containers receive the API key through ``docker exec -e NAME``
(value taken from this process's environment), and logs only ever show variable names.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from collections.abc import Iterable
from pathlib import Path
from typing import Any

SECRET_ENV = "OPENAI_API_KEY"
_log_lock = threading.Lock()
_log_file: Path | None = None


def set_log_file(path: Path) -> None:
    global _log_file
    path.parent.mkdir(parents=True, exist_ok=True)
    _log_file = path


def log(message: str) -> None:
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {message}"
    with _log_lock:
        print(line, flush=True)
        if _log_file:
            with _log_file.open("a") as f:
                f.write(line + "\n")


def run(args: list[str], *, check: bool = True, timeout: float | None = None, input: bytes | None = None, stdout: Any = subprocess.PIPE) -> subprocess.CompletedProcess:
    """Run a command with captured output. Never pass secret values in ``args``. Children get their
    own session, so a Ctrl-C in the runner's terminal stops the runner gracefully instead of
    killing the docker commands of attempts that are winding down."""
    return subprocess.run(args, input=input, stdout=stdout, stderr=subprocess.PIPE, timeout=timeout, check=check, start_new_session=True)


def docker(*args: str, check: bool = True, timeout: float | None = None, input: bytes | None = None) -> str:
    try:
        result = run(["docker", *args], check=check, timeout=timeout, input=input)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"docker {' '.join(args[:3])} failed: {error.stderr.decode(errors='replace')[-2000:]}") from None
    return result.stdout.decode(errors="replace")


def docker_to_file(args: list[str], path: Path, timeout: float | None = None, ok_codes: tuple[int, ...] = (0,)) -> str:
    """Stream a docker command's stdout into ``path``; return its stderr (warnings) on success."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    with tmp.open("wb") as out:
        result = subprocess.run(["docker", *args], stdout=out, stderr=subprocess.PIPE, timeout=timeout, start_new_session=True)
    stderr = result.stderr.decode(errors="replace")
    if result.returncode not in ok_codes:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"docker {' '.join(args[:4])} exited {result.returncode}: {stderr[-2000:]}")
    tmp.replace(path)
    return stderr[-2000:]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, sha256: str, dest: Path) -> Path:
    if dest.exists() and sha256_file(dest) == sha256:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".partial")
    with urllib.request.urlopen(url, timeout=300) as response, tmp.open("wb") as out:
        while chunk := response.read(1 << 20):
            out.write(chunk)
    actual = sha256_file(tmp)
    if actual != sha256:
        tmp.unlink()
        raise RuntimeError(f"checksum mismatch for {url}: expected {sha256}, got {actual}")
    tmp.replace(dest)
    return dest


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def require_secret() -> None:
    value = os.environ.get(SECRET_ENV, "")
    if not value.strip():
        sys.exit(f"{SECRET_ENV} is not set. Export it before scripts/zsbench, or store it with scripts/push-openai-key.sh (see README).")


def load_secret_file(path: str | None) -> None:
    """Load ``OPENAI_API_KEY=...`` from a mounted file into this process only."""
    if not path or os.environ.get(SECRET_ENV):
        return
    file = Path(path)
    if not file.exists():
        return
    for line in file.read_text().splitlines():
        name, sep, value = line.partition("=")
        if sep and name.strip() == SECRET_ENV:
            os.environ[SECRET_ENV] = value.strip().strip('"').strip("'")


def secret_values() -> list[bytes]:
    value = os.environ.get(SECRET_ENV, "").strip()
    return [value.encode()] if len(value) >= 16 else []


def iter_json_objects(value: Any) -> Iterable[dict[str, Any]]:
    """Yield every dict nested anywhere inside ``value``."""
    if isinstance(value, dict):
        yield value
        for item in value.values():
            yield from iter_json_objects(item)
    elif isinstance(value, list):
        for item in value:
            yield from iter_json_objects(item)
