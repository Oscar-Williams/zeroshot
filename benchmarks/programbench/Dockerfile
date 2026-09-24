# The benchmark runner. It drives the host's Docker daemon (mount /var/run/docker.sock) to build
# the agent and proxy images, run isolated attempts, and evaluate them with ProgramBench.
FROM python:3.12-slim-bookworm@sha256:392307d22300de8b5986851a12d9176dfc0fc073e65bf6523ebd7dcbeb23564e

ARG DOCKER_VERSION=29.8.1
ARG DOCKER_SHA256=d8db66739d2e28d4933786d73e918d9be643a67fbd835db1bf740d650a259e70
ARG BUILDX_VERSION=v0.37.1
ARG BUILDX_SHA256=9447199cdb435f25880548343c128a4b6650e8891ee598905d8d29d39a8e359b
ARG VCS_REF=unknown

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl git \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL -o /tmp/docker.tgz "https://download.docker.com/linux/static/stable/x86_64/docker-${DOCKER_VERSION}.tgz" \
    && echo "${DOCKER_SHA256}  /tmp/docker.tgz" | sha256sum -c - \
    && tar -xzf /tmp/docker.tgz -C /tmp docker/docker \
    && install -m 0755 /tmp/docker/docker /usr/local/bin/docker \
    && mkdir -p /usr/local/lib/docker/cli-plugins \
    && curl -fsSL -o /usr/local/lib/docker/cli-plugins/docker-buildx "https://github.com/docker/buildx/releases/download/${BUILDX_VERSION}/buildx-${BUILDX_VERSION}.linux-amd64" \
    && echo "${BUILDX_SHA256}  /usr/local/lib/docker/cli-plugins/docker-buildx" | sha256sum -c - \
    && chmod 0755 /usr/local/lib/docker/cli-plugins/docker-buildx \
    && rm -rf /tmp/docker /tmp/docker.tgz

COPY requirements.lock /tmp/requirements.lock
RUN pip install --no-cache-dir --require-hashes -r /tmp/requirements.lock

WORKDIR /opt/zeroshot-bench
COPY . .
ENV PYTHONPATH=/opt/zeroshot-bench \
    PYTHONUNBUFFERED=1 \
    ZSBENCH_VCS_REF=${VCS_REF}
ENTRYPOINT ["python", "-m", "bench"]
