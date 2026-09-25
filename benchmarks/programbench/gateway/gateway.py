"""Model gateway: the only way out of a Claude attempt's network.

Claude Code in the attempt container sends plain HTTP to this gateway with a placeholder key. The
gateway holds the real key (it never enters the attempt container), forwards to api.anthropic.com
over TLS, refuses server-side tools (web search, web fetch, code execution, remote MCP), refuses
requests once the attempt's spending cap is reached, and appends one JSON line per request to its
log: path, model, tool names, request id and token usage. Keys, headers and bodies are never
logged. Python standard library only.
"""

from __future__ import annotations

import http.client
import http.server
import json
import os
import ssl
import sys
import threading
import time
from typing import Any

UPSTREAM = "api.anthropic.com"
PORT = 8889
LOG_PATH = "/var/log/zsbench/gateway.jsonl"
# Hop-by-hop and connection-specific headers are not forwarded in either direction.
HOP_BY_HOP = frozenset({"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailer", "transfer-encoding", "upgrade", "host", "content-length"})
CLIENT_CREDENTIALS = frozenset({"x-api-key", "authorization"})
USAGE_FIELDS = ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens")


def blocked_request_features(payload: Any) -> list[str]:
    """Server-side capabilities a Messages request asks for. Claude Code's own tools are custom
    tools (no ``type``, or ``custom``); anything else runs on Anthropic's side, outside the
    attempt's egress controls."""
    if not isinstance(payload, dict):
        return []
    blocked = [str(t.get("type") or t.get("name")) for t in payload.get("tools") or [] if isinstance(t, dict) and t.get("type") not in (None, "custom")]
    for key in ("mcp_servers", "container"):
        if payload.get(key):
            blocked.append(key)
    return blocked


def tool_names(payload: Any) -> list[str]:
    if not isinstance(payload, dict):
        return []
    return sorted(str(t.get("name")) for t in payload.get("tools") or [] if isinstance(t, dict) and t.get("name"))


def forward_headers(incoming: list[tuple[str, str]], key: str) -> dict[str, str]:
    """The client's headers minus hop-by-hop headers and its placeholder credentials, plus the key.
    Responses are requested uncompressed so the gateway can read their usage."""
    headers = {name: value for name, value in incoming if name.lower() not in HOP_BY_HOP | CLIENT_CREDENTIALS | {"accept-encoding"}}
    headers["x-api-key"] = key
    headers["host"] = UPSTREAM
    headers["accept-encoding"] = "identity"
    return headers


def merge_usage(total: dict[str, Any], update: dict[str, Any] | None) -> None:
    """Fold a usage object (from message_start or a cumulative message_delta) into ``total``."""
    for name, value in (update or {}).items():
        if name == "cache_creation" and isinstance(value, dict):
            total.setdefault("cache_creation", {}).update({k: v for k, v in value.items() if isinstance(v, int)})
        elif name == "server_tool_use" and isinstance(value, dict):
            total["server_tool_use"] = {k: v for k, v in value.items() if isinstance(v, int)}
        elif name in USAGE_FIELDS and isinstance(value, int):
            total[name] = value


class SSEUsage:
    """Reads token usage out of a streamed Messages response as the bytes pass through."""

    def __init__(self) -> None:
        self.usage: dict[str, Any] = {}
        self.model: str | None = None
        self.stop_reason: str | None = None
        self._pending = b""

    def feed(self, chunk: bytes) -> None:
        self._pending += chunk
        *lines, self._pending = self._pending.split(b"\n")
        for line in lines:
            self._line(line.rstrip(b"\r"))

    def _line(self, line: bytes) -> None:
        if not line.startswith(b"data:"):
            return
        try:
            event = json.loads(line[5:].strip())
        except ValueError:
            return
        if not isinstance(event, dict):
            return
        if event.get("type") == "message_start":
            message = event.get("message") or {}
            self.model = message.get("model")
            merge_usage(self.usage, message.get("usage"))
        elif event.get("type") == "message_delta":
            merge_usage(self.usage, event.get("usage"))
            self.stop_reason = (event.get("delta") or {}).get("stop_reason") or self.stop_reason


def usage_from_body(body: bytes) -> tuple[dict[str, Any], str | None]:
    try:
        data = json.loads(body)
    except ValueError:
        return {}, None
    usage: dict[str, Any] = {}
    if isinstance(data, dict):
        merge_usage(usage, data.get("usage"))
        return usage, data.get("model")
    return usage, None


def cost_usd(usage: dict[str, Any], prices: dict[str, float]) -> float:
    """Price one response's usage (USD per million tokens); 1-hour cache writes when priced."""
    one_hour = (usage.get("cache_creation") or {}).get("ephemeral_1h_input_tokens", 0)
    writes = usage.get("cache_creation_input_tokens", 0)
    return (
        usage.get("input_tokens", 0) * prices.get("input", 0)
        + usage.get("cache_read_input_tokens", 0) * prices.get("cached_input", 0)
        + (writes - one_hour) * prices.get("cache_write", 0)
        + one_hour * prices.get("cache_write_1h", prices.get("cache_write", 0))
        + usage.get("output_tokens", 0) * prices.get("output", 0)
    ) / 1_000_000


class Gateway:
    def __init__(self, key: str, prices: dict[str, float], cap_usd: float | None, log_path: str) -> None:
        self.key, self.prices, self.cap_usd = key, prices, cap_usd
        self.spent = 0.0
        self.lock = threading.Lock()
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.log_file = open(log_path, "a", buffering=1)
        self.context = ssl.create_default_context()

    def log(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, sort_keys=True)
        with self.lock:
            self.log_file.write(line + "\n")
        print(line, flush=True)

    def charge(self, usage: dict[str, Any]) -> float:
        cost = cost_usd(usage, self.prices)
        with self.lock:
            self.spent += cost
        return cost

    def over_cap(self) -> bool:
        with self.lock:
            return self.cap_usd is not None and self.spent >= self.cap_usd


def make_handler(gateway: Gateway) -> type[http.server.BaseHTTPRequestHandler]:
    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
            pass  # the JSON log is the record; never echo request lines with query strings

        def do_GET(self) -> None:
            self._forward()

        def do_POST(self) -> None:
            self._forward()

        def do_PUT(self) -> None:
            self._forward()

        def do_PATCH(self) -> None:
            self._forward()

        def do_DELETE(self) -> None:
            self._forward()

        def _refuse(self, status: int, message: str, record: dict[str, Any]) -> None:
            body = json.dumps({"type": "error", "error": {"type": "permission_error", "message": f"zsbench gateway: {message}"}}).encode()
            self.send_response_only(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            gateway.log({**record, "status": status, "refused": message})

        def _forward(self) -> None:
            started = time.time()
            length = int(self.headers.get("content-length") or 0)
            body = self.rfile.read(length) if length else b""
            record: dict[str, Any] = {"t": round(started, 3), "method": self.command, "path": self.path.split("?", 1)[0], "bytes_in": len(body)}
            payload: Any = None
            if body and "json" in (self.headers.get("content-type") or ""):
                try:
                    payload = json.loads(body)
                except ValueError:
                    payload = None
            if isinstance(payload, dict):
                record.update(model=payload.get("model"), stream=bool(payload.get("stream")), tools=tool_names(payload))
            blocked = blocked_request_features(payload)
            if blocked:
                self._refuse(403, f"server-side tools are not allowed: {blocked}", {**record, "blocked": blocked})
                return
            if gateway.over_cap():
                self._refuse(403, "the attempt's spending cap is reached", {**record, "cap_reached": True})
                return
            upstream = http.client.HTTPSConnection(UPSTREAM, 443, timeout=1800, context=gateway.context)
            try:
                upstream.request(self.command, self.path, body=body, headers=forward_headers(self.headers.items(), gateway.key))
                response = upstream.getresponse()
            except OSError as error:
                upstream.close()
                self._refuse(502, "upstream connection failed", {**record, "upstream_error": type(error).__name__})
                return
            record.update(status=response.status, request_id=response.getheader("request-id"))
            usage: dict[str, Any] = {}
            model = None
            try:
                self.send_response_only(response.status)
                for name, value in response.getheaders():
                    if name.lower() not in HOP_BY_HOP:
                        self.send_header(name, value)
                if "text/event-stream" in (response.getheader("content-type") or ""):
                    self.send_header("transfer-encoding", "chunked")
                    self.end_headers()
                    parser = SSEUsage()
                    while chunk := response.read1(65536):
                        parser.feed(chunk)
                        self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                        self.wfile.flush()
                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()
                    usage, model = parser.usage, parser.model
                    record["stop_reason"] = parser.stop_reason
                else:
                    data = response.read()
                    self.send_header("content-length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    if record["path"].startswith("/v1/messages") and response.status == 200:
                        usage, model = usage_from_body(data)
            except OSError as error:
                record["client_error"] = type(error).__name__
                self.close_connection = True
            finally:
                upstream.close()
            if usage:
                record.update(usage=usage, response_model=model, cost_usd=round(gateway.charge(usage), 6))
            record["seconds"] = round(time.time() - started, 3)
            gateway.log(record)

    return Handler


def main() -> None:
    key = os.environ.pop("ANTHROPIC_API_KEY", "")
    if len(key) < 16:
        sys.exit("ANTHROPIC_API_KEY is not set")
    prices = json.loads(os.environ.get("ZSBENCH_PRICES") or "{}")
    cap = float(os.environ.get("ZSBENCH_USD_CAP") or 0) or None
    gateway = Gateway(key, prices, cap, os.environ.get("ZSBENCH_GATEWAY_LOG", LOG_PATH))
    server = http.server.ThreadingHTTPServer(("0.0.0.0", PORT), make_handler(gateway))
    server.daemon_threads = True
    gateway.log({"t": round(time.time(), 3), "event": "listening", "port": PORT, "cap_usd": cap})
    server.serve_forever()


if __name__ == "__main__":
    main()
