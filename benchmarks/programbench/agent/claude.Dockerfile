# The ProgramBench task image, unchanged except for the agent tooling below (Claude Code harness).
ARG TASK_IMAGE
FROM ${TASK_IMAGE}
USER root
COPY --chmod=0755 zeroshot /usr/local/bin/zeroshot
COPY claude /opt/claude-code/claude
# Zeroshot resolves `claude` through its root-owned PATH and finds this launcher, which starts the
# real binary with the benchmark's isolation flags and environment (claude-launcher.sh.in).
COPY claude-launcher.sh /usr/local/bin/claude
# The experiment's declared task adjustments (reference location, documentation fixes); no-ops for
# an unmodified ProgramBench task.
COPY task-adjustments.json prepare-task.py /tmp/zsbench-prepare/
RUN python3 /tmp/zsbench-prepare/prepare-task.py /tmp/zsbench-prepare/task-adjustments.json \
    && rm -rf /tmp/zsbench-prepare
# Execute-only, root-owned executables make the kernel mark the Zeroshot and Claude Code processes
# non-dumpable, so tool commands cannot read their /proc/<pid>/environ or memory. The API key itself
# never enters this container: Claude Code talks to the attempt's model gateway with a placeholder.
RUN chown -R root:root /opt/claude-code /usr/local/bin/zeroshot /usr/local/bin/claude \
    && chmod 0755 /opt/claude-code /usr/local/bin/claude \
    && chmod 0711 /usr/local/bin/zeroshot /opt/claude-code/claude \
    && mkdir -p /opt/zeroshot-bench/run \
    && chmod 0755 /opt/zeroshot-bench /opt/zeroshot-bench/run \
    && zeroshot --version \
    && /opt/claude-code/claude --version
# Zeroshot's local mode needs a GitHub-shaped origin to identify the checkout. Nothing is ever
# pushed: runs have no delivery step and the container cannot reach GitHub.
USER agent
RUN git -C /workspace remote add origin https://github.com/zeroshot-bench/local-workspace.git
USER root
WORKDIR /workspace
CMD ["sleep", "infinity"]
