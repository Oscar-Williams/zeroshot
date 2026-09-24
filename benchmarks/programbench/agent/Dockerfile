# The ProgramBench task image, unchanged except for the agent tooling below.
ARG TASK_IMAGE
FROM ${TASK_IMAGE}
USER root
COPY --chmod=0755 zeroshot /usr/local/bin/zeroshot
# Codex's full platform package, as the npm distribution installs it: the CLI finds its helpers
# (codex-code-mode-host for tool calls, rg, bwrap) relative to its own executable.
COPY codex-package /opt/codex
COPY codex-config.toml /home/agent/.codex/config.toml
# The experiment's declared task adjustments (reference location, documentation fixes); no-ops for
# an unmodified ProgramBench task.
COPY task-adjustments.json prepare-task.py /tmp/zsbench-prepare/
RUN python3 /tmp/zsbench-prepare/prepare-task.py /tmp/zsbench-prepare/task-adjustments.json \
    && rm -rf /tmp/zsbench-prepare
# Zeroshot and Codex hold the API key in their process environment and memory. Execute-only,
# root-owned executables make the kernel mark those processes non-dumpable, so the agent user's
# tool commands cannot read /proc/<pid>/environ or /proc/<pid>/mem (the same protection that keeps
# the reference ./executable unreadable). Shared libraries stay readable for the loader.
RUN ln -s /opt/codex/bin/codex /usr/local/bin/codex \
    && chown -R root:root /opt/codex /usr/local/bin/zeroshot \
    && chmod 0711 /usr/local/bin/zeroshot /opt/codex/bin/* /opt/codex/codex-path/* /opt/codex/codex-resources/bwrap /opt/codex/codex-resources/voice/bin/* \
    && chown -R agent:agent /home/agent/.codex \
    && chmod 0700 /home/agent/.codex \
    && mkdir -p /opt/zeroshot-bench/run \
    && chmod 0755 /opt/zeroshot-bench /opt/zeroshot-bench/run \
    && zeroshot --version \
    && codex --version
# Zeroshot's local mode needs a GitHub-shaped origin to identify the checkout. Nothing is ever
# pushed: runs have no delivery step and the container cannot reach GitHub.
USER agent
RUN git -C /workspace remote add origin https://github.com/zeroshot-bench/local-workspace.git
USER root
WORKDIR /workspace
CMD ["sleep", "infinity"]
