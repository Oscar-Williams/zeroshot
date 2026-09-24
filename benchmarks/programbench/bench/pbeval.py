"""Run the ProgramBench CLI with pytest-rerunfailures pinned.

ProgramBench 1.2.4 (the leaderboard's version) installs the newest pytest-rerunfailures inside
each eval container. Releases from 16.6.1 on write a bare ``<testcase/>`` per rerun, which the
JUnit parser reads as a pass. Upstream pinned 16.4 in commit b08d862; this applies the same pin
to 1.2.4 without changing anything else. Evaluations run in threads, so the patch covers them.
"""

import programbench.eval.eval as evaluation
from programbench.cli.main import app

RERUN_PIN = "pytest-rerunfailures==16.4"
_run_step = evaluation.Evaluator._run_step


def _pinned_run_step(self, command, **kwargs):
    if "pytest-rerunfailures" in command and "pytest-rerunfailures==" not in command:
        command = command.replace("pytest-rerunfailures", RERUN_PIN)
    return _run_step(self, command, **kwargs)


evaluation.Evaluator._run_step = _pinned_run_step

if __name__ == "__main__":
    app()
