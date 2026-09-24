# Egress allowlist for agent containers: HTTPS CONNECT to api.openai.com only.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
RUN apk add --no-cache "tinyproxy~1.11"
COPY tinyproxy.conf /etc/tinyproxy/tinyproxy.conf
COPY filter /etc/tinyproxy/filter
USER tinyproxy
EXPOSE 8888
CMD ["tinyproxy", "-d", "-c", "/etc/tinyproxy/tinyproxy.conf"]
