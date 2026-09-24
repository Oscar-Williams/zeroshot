# syntax=docker/dockerfile:1
# The ProgramBench task image, unchanged except for the agent tooling below.
ARG TASK_IMAGE
FROM ${TASK_IMAGE}
USER root
COPY --chmod=0755 zeroshot codex /usr/local/bin/
COPY codex-config.toml /home/agent/.codex/config.toml
RUN chown -R agent:agent /home/agent/.codex \
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
