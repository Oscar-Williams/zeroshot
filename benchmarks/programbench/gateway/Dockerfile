# Model gateway for Claude attempts: holds the API key outside the attempt container (gateway.py).
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
COPY gateway.py /opt/zsbench/gateway.py
RUN adduser -D -H gateway && mkdir -p /var/log/zsbench && chown gateway /var/log/zsbench
USER gateway
EXPOSE 8889
# The runner starts the gateway process with `docker exec` once it has the key; the container
# itself only idles.
CMD ["sleep", "infinity"]
