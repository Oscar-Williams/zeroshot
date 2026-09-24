# syntax=docker/dockerfile:1
# The ProgramBench task image, unchanged except for the agent tooling below.
ARG TASK_IMAGE
FROM ${TASK_IMAGE}
USER root
COPY --chmod=0755 zeroshot /usr/local/bin/zeroshot
# Codex's full platform package, as the npm distribution installs it: the CLI finds its helpers
# (codex-code-mode-host for tool calls, rg, bwrap) relative to its own executable.
COPY codex-package /opt/codex
COPY codex-config.toml /home/agent/.codex/config.toml
RUN ln -s /opt/codex/bin/codex /usr/local/bin/codex \
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
