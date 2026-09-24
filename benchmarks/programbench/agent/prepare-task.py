"""Apply an experiment's declared task adjustments while the agent image is built (runs as root).

Both adjustments are declared in the experiment file and recorded in the manifest:
- ``reference_path``: move the reference executable out of the workspace into a root-owned
  directory, where the agent can run it for the whole attempt but cannot overwrite, move or delete
  it (the task asks agents to build their own ``./executable`` at the reference's original path);
- ``doc_fixes``: replace exact text in the task's bundled documentation, each exactly once, then
  fold the edits into the workspace's initial commit so every attempt starts from a clean tree.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

UPSTREAM_REFERENCE = Path("/workspace/executable")
# runuser is resolved with root's PATH; the git it starts gets the agent's minimal environment.
AGENT_GIT = [shutil.which("runuser") or "/usr/sbin/runuser", "-u", "agent", "--", "git", "-C", "/workspace"]
AGENT_ENV = {"HOME": "/home/agent", "PATH": "/usr/bin:/bin"}


def git(*args: str) -> str:
    return subprocess.run([*AGENT_GIT, *args], env=AGENT_ENV, check=True, capture_output=True, text=True).stdout.strip()


def main() -> None:
    spec = json.loads(Path(sys.argv[1]).read_text())
    reference = Path(spec["reference_path"])
    if reference != UPSTREAM_REFERENCE:
        reference.parent.mkdir(parents=True, exist_ok=True)
        os.replace(UPSTREAM_REFERENCE, reference)
        os.chown(reference.parent, 0, 0)
        os.chmod(reference.parent, 0o755)
        os.chown(reference, 0, 0)
        os.chmod(reference, 0o111)
    edited = []
    for fix in spec["doc_fixes"]:
        path = Path("/workspace") / fix["file"]
        text = path.read_text()
        if text.count(fix["old"]) != 1:
            sys.exit(f"{fix['file']}: expected exactly one {fix['old']!r}")
        path.write_text(text.replace(fix["old"], fix["new"]))
        edited.append(fix["file"])
    if edited:
        # Keep the commit's own identity so the history looks as the task shipped it.
        name, email = git("log", "-1", "--format=%an"), git("log", "-1", "--format=%ae")
        git("add", "--", *sorted(set(edited)))
        git("-c", f"user.name={name}", "-c", f"user.email={email}", "commit", "--amend", "--no-edit", "--quiet")
    status = git("status", "--porcelain")
    if reference != UPSTREAM_REFERENCE and status:
        sys.exit(f"workspace not clean after the adjustments:\n{status}")


if __name__ == "__main__":
    main()
