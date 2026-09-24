"""Build the proxy and agent images and manage the isolated network."""

from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
from pathlib import Path

from . import ROOT
from .config import Experiment, pins
from .util import docker, download, log, run

# Variables that Zeroshot or Codex set per process; everything else from the image ENV is mirrored.
_NOT_MIRRORED = {"HOME", "HOSTNAME"}
# Non-interactive settings the upstream mini-SWE-agent ProgramBench baseline exported.
_BASELINE_ENV = {"PAGER": "cat", "MANPAGER": "cat", "LESS": "-R", "PIP_PROGRESS_BAR": "off", "TQDM_DISABLE": "1"}


def _extract(archive: Path, member: str, dest: Path) -> None:
    with tarfile.open(archive) as tar:
        source = tar.extractfile(member)
        if source is None:
            raise RuntimeError(f"{member} missing from {archive}")
        dest.write_bytes(source.read())
    dest.chmod(0o755)


def ensure_image(reference: str) -> None:
    if docker("image", "inspect", reference, check=False).strip() in ("", "[]"):
        log(f"pulling {reference}")
        docker("pull", "--quiet", reference, timeout=3600)


def image_env(reference: str) -> dict[str, str]:
    env = json.loads(docker("image", "inspect", reference, "--format", "{{json .Config.Env}}"))
    return dict(item.split("=", 1) for item in env or [])


def codex_config(task_env: dict[str, str]) -> str:
    mirrored = {k: v for k, v in task_env.items() if k not in _NOT_MIRRORED}
    for key, value in _BASELINE_ENV.items():
        mirrored.setdefault(key, value)
    lines = [f"{key} = {json.dumps(value)}" for key, value in sorted(mirrored.items())]
    template = (ROOT / "agent" / "codex-config.toml.in").read_text()
    return template.replace("@SET@", "\n".join(lines))


def build_proxy(cache: Path) -> str:
    base = pins()["proxy_base_image"]
    digest = hashlib.sha256()
    for name in ("Dockerfile", "tinyproxy.conf", "filter"):
        digest.update((ROOT / "proxy" / name).read_bytes())
    digest.update(base.encode())
    tag = f"zsbench-proxy:{digest.hexdigest()[:16]}"
    if docker("image", "inspect", tag, check=False).strip() not in ("", "[]"):
        return tag
    log(f"building {tag}")
    docker("build", "--quiet", "--build-arg", f"BASE_IMAGE={base}", "-t", tag, str(ROOT / "proxy"), timeout=1800)
    return tag


def build_agent(exp: Experiment, cache: Path) -> tuple[str, dict[str, str]]:
    """Return the agent image tag and the rendered Codex config (for the manifest)."""
    pin = pins()
    ensure_image(exp.task_image)
    config = codex_config(image_env(exp.task_image))
    digest = hashlib.sha256()
    digest.update(exp.task_image.encode())
    digest.update(config.encode())
    digest.update((ROOT / "agent" / "Dockerfile").read_bytes())
    for tool in ("zeroshot", "codex"):
        digest.update(pin[tool]["sha256"].encode())
    tag = f"zsbench-agent:{digest.hexdigest()[:16]}"
    info = {"tag": tag, "task_image": exp.task_image, "codex_config": config}
    if docker("image", "inspect", tag, check=False).strip() not in ("", "[]"):
        return tag, info
    context = cache / "agent-context" / tag.split(":", 1)[1]
    shutil.rmtree(context, ignore_errors=True)
    context.mkdir(parents=True)
    zeroshot = download(pin["zeroshot"]["url"], pin["zeroshot"]["sha256"], cache / "downloads" / Path(pin["zeroshot"]["url"]).name)
    _extract(zeroshot, pin["zeroshot"]["member"], context / "zeroshot")
    codex = download(pin["codex"]["url"], pin["codex"]["sha256"], cache / "downloads" / Path(pin["codex"]["url"]).name)
    with tarfile.open(codex) as tar:
        tar.extractall(context / "codex-package", filter="data")
    shutil.copy(ROOT / "agent" / "Dockerfile", context / "Dockerfile")
    (context / "codex-config.toml").write_text(config)
    log(f"building {tag} from {exp.task_image}")
    docker("build", "--quiet", "--build-arg", f"TASK_IMAGE={exp.task_image}", "-t", tag, str(context), timeout=3600)
    return tag, info


class Network:
    """An internal Docker network whose only exit is the allowlist proxy."""

    def __init__(self, exp: Experiment, proxy_image: str):
        self.name = f"zsbench-{exp.id}"
        self.proxy = f"zsbench-{exp.id}-proxy"
        self.proxy_image = proxy_image

    @property
    def proxy_url(self) -> str:
        return f"http://{self.proxy}:8888"

    def up(self) -> None:
        self.down()
        docker("network", "create", "--internal", "--label", "zsbench=1", self.name)
        docker("run", "-d", "--name", self.proxy, "--label", "zsbench=1", "--restart", "unless-stopped", "--network", self.name, self.proxy_image)
        docker("network", "connect", "bridge", self.proxy)
        log(f"network {self.name} up; egress only through {self.proxy} (allowlist: api.openai.com:443)")

    def proxy_log(self) -> str:
        result = run(["docker", "logs", self.proxy], check=False)
        return result.stdout.decode(errors="replace") + result.stderr.decode(errors="replace")

    def down(self) -> None:
        docker("rm", "-f", self.proxy, check=False)
        docker("network", "rm", self.name, check=False)
