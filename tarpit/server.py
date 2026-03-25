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

    def _respond(self) -> None:
        time.sleep(DELAY_SECONDS)
        body = b"Not found\n"
        self.send_response(STATUS_CODE)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        self._respond()

    def do_POST(self) -> None:
        self._respond()

    def do_PUT(self) -> None:
        self._respond()

    def do_DELETE(self) -> None:
        self._respond()

    def do_HEAD(self) -> None:
        time.sleep(DELAY_SECONDS)
        self.send_response(STATUS_CODE)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

    def do_OPTIONS(self) -> None:
        self._respond()

    def do_PATCH(self) -> None:
        self._respond()

    def log_message(self, format: str, *args) -> None:
        print("%s - - [%s] %s" % (self.address_string(), self.log_date_time_string(), format % args), flush=True)


if __name__ == "__main__":
    httpd = ThreadingHTTPServer((HOST, PORT), TarpitHandler)
    print(f"Starting tarpit on {HOST}:{PORT} with {DELAY_SECONDS}s delay", flush=True)
    httpd.serve_forever()