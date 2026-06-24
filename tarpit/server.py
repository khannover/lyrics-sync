import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


DELAY_SECONDS = float(os.environ.get("TARPIT_DELAY_SECONDS", "20"))
STATUS_CODE = int(os.environ.get("TARPIT_STATUS_CODE", "404"))
HOST = os.environ.get("TARPIT_HOST", "0.0.0.0")
PORT = int(os.environ.get("TARPIT_PORT", "8080"))


class TarpitHandler(BaseHTTPRequestHandler):
    server_version = "tarpit"
    sys_version = ""

    def _client_ip(self) -> str:
        forwarded_for = self.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",", 1)[0].strip()
        return self.headers.get("X-Real-IP") or self.client_address[0]

    def _log_event(self, event: str, **fields: str) -> None:
        parts = [f"[tarpit] {event}"]
        for key, value in fields.items():
            parts.append(f"{key}={value}")
        print(" ".join(parts), flush=True)

    def _delay_and_release(
        self,
        client_ip: str,
        user_agent: str,
        reason: str,
        method: str,
        path: str,
        *,
        body: bytes | None = b"Not found\n",
    ) -> None:
        self._log_event(
            "caught",
            ip=client_ip,
            method=method,
            path=path,
            reason=reason,
            ua=user_agent,
        )

        started = time.monotonic()
        time.sleep(DELAY_SECONDS)
        elapsed = time.monotonic() - started

        status = str(STATUS_CODE)
        try:
            self.send_response(STATUS_CODE)
            self.send_header("Cache-Control", "no-store")
            if body is not None:
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if body is not None:
                self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            status = "client_gone"

        self._log_event(
            "released",
            ip=client_ip,
            method=method,
            path=path,
            reason=reason,
            delay=f"{elapsed:.1f}s",
            status=status,
        )

    def _respond(self) -> None:
        self._delay_and_release(
            self._client_ip(),
            self.headers.get("User-Agent", "-"),
            self.headers.get("X-Tarpit-Reason", "unknown"),
            self.command,
            self.path,
        )

    def do_GET(self) -> None:
        self._respond()

    def do_POST(self) -> None:
        self._respond()

    def do_PUT(self) -> None:
        self._respond()

    def do_DELETE(self) -> None:
        self._respond()

    def do_HEAD(self) -> None:
        self._delay_and_release(
            self._client_ip(),
            self.headers.get("User-Agent", "-"),
            self.headers.get("X-Tarpit-Reason", "unknown"),
            self.command,
            self.path,
            body=None,
        )

    def do_OPTIONS(self) -> None:
        self._respond()

    def do_PATCH(self) -> None:
        self._respond()

    def log_message(self, format: str, *args) -> None:
        return


if __name__ == "__main__":
    httpd = ThreadingHTTPServer((HOST, PORT), TarpitHandler)
    print(
        f"[tarpit] listening on {HOST}:{PORT} delay={DELAY_SECONDS}s status={STATUS_CODE}",
        flush=True,
    )
    httpd.serve_forever()