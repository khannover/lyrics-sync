from __future__ import annotations

import hmac
import json
import os
import threading
import time
from collections import Counter, deque
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

from classifier import Classification, classify_request, load_extra_rules


DELAY_SECONDS = float(os.environ.get("TARPIT_DELAY_SECONDS", "20"))
STATUS_CODE = int(os.environ.get("TARPIT_STATUS_CODE", "404"))
HOST = os.environ.get("TARPIT_HOST", "0.0.0.0")
PORT = int(os.environ.get("TARPIT_PORT", "8080"))
LOG_PATH = Path(os.environ.get("TARPIT_LOG_PATH", "/data/tarpit-events.jsonl"))
MAX_LOG_BYTES = int(os.environ.get("TARPIT_MAX_LOG_BYTES", str(5 * 1024 * 1024)))
MAX_MEMORY_EVENTS = int(os.environ.get("TARPIT_MAX_MEMORY_EVENTS", "5000"))

EXTRA_RULES = load_extra_rules()


def _expected_stats_token() -> str:
    return (
        os.environ.get("TARPIT_STATS_TOKEN")
        or os.environ.get("LYRIC_SYNC_CALLBACK_SECRET")
        or ""
    ).strip()


class TarpitStore:
    """Thread-safe in-memory aggregator + JSONL persistence for tarpit catches."""

    def __init__(self, log_path: Path, max_memory_events: int = MAX_MEMORY_EVENTS) -> None:
        self.log_path = log_path
        self._lock = threading.Lock()
        self._events: Deque[Dict[str, Any]] = deque(maxlen=max_memory_events)
        self._total_events = 0
        self._client_gone = 0
        self._total_wasted_sec = 0.0
        self._by_category: Counter[str] = Counter()
        self._by_cve: Counter[str] = Counter()
        self._by_severity: Counter[str] = Counter()
        self._by_reason: Counter[str] = Counter()
        self._by_ip: Counter[str] = Counter()
        self._by_path: Counter[str] = Counter()
        self._load_existing()

    def _index_event(self, ev: Dict[str, Any]) -> None:
        self._events.append(ev)
        self._total_events += 1
        if ev.get("status") == "client_gone":
            self._client_gone += 1
        self._total_wasted_sec += float(ev.get("delay_sec") or 0.0)
        self._by_category[str(ev.get("category") or "unknown")] += 1
        if ev.get("cve"):
            self._by_cve[str(ev["cve"])] += 1
        self._by_severity[str(ev.get("severity") or "low")] += 1
        self._by_reason[str(ev.get("reason") or "unknown")] += 1
        self._by_ip[str(ev.get("ip") or "-")] += 1
        self._by_path[str(ev.get("path") or "/")] += 1

    def _load_existing(self) -> None:
        if not self.log_path.is_file():
            return
        try:
            with self.log_path.open("r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ev = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(ev, dict):
                        self._index_event(ev)
        except OSError:
            pass

    def record(self, ev: Dict[str, Any]) -> None:
        with self._lock:
            self._index_event(ev)
            self._append_jsonl(ev)

    def _append_jsonl(self, ev: Dict[str, Any]) -> None:
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            if self.log_path.exists() and self.log_path.stat().st_size >= MAX_LOG_BYTES:
                rotated = self.log_path.with_suffix(".jsonl.1")
                self.log_path.replace(rotated)
            with self.log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(ev, ensure_ascii=False) + "\n")
        except OSError:
            pass

    def stats(self, recent_limit: int = 20) -> Dict[str, Any]:
        with self._lock:
            recent = list(self._events)[-max(1, min(recent_limit, 200)) :]
            recent.reverse()
            return {
                "total_events": self._total_events,
                "client_gone_count": self._client_gone,
                "total_wasted_seconds": round(self._total_wasted_sec, 1),
                "total_wasted_minutes": round(self._total_wasted_sec / 60.0, 2),
                "by_category": dict(self._by_category.most_common(25)),
                "by_cve": dict(self._by_cve.most_common(25)),
                "by_severity": dict(self._by_severity.most_common()),
                "by_reason": dict(self._by_reason.most_common()),
                "top_ips": [
                    {"ip": ip, "count": count} for ip, count in self._by_ip.most_common(20)
                ],
                "top_paths": [
                    {"path": path, "count": count}
                    for path, count in self._by_path.most_common(20)
                ],
                "recent": recent,
            }

    def events(
        self,
        *,
        limit: int = 50,
        category: Optional[str] = None,
        cve: Optional[str] = None,
        ip: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        limit = max(1, min(limit, 500))
        cat_f = (category or "").strip().lower()
        cve_f = (cve or "").strip().upper()
        ip_f = (ip or "").strip()
        with self._lock:
            out: List[Dict[str, Any]] = []
            for ev in reversed(self._events):
                if cat_f and str(ev.get("category") or "").lower() != cat_f:
                    continue
                if cve_f and str(ev.get("cve") or "").upper() != cve_f:
                    continue
                if ip_f and str(ev.get("ip") or "") != ip_f:
                    continue
                out.append(ev)
                if len(out) >= limit:
                    break
            return out


STORE = TarpitStore(LOG_PATH)


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
            if value:
                parts.append(f"{key}={value}")
        print(" ".join(parts), flush=True)

    def _send_json(self, status_code: int, payload: Any) -> None:
        raw = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _is_stats_authorized(self) -> bool:
        expected = _expected_stats_token()
        if not expected:
            return True
        provided = (self.headers.get("X-Lyrics-Sync-Token") or "").strip()
        return bool(provided) and hmac.compare_digest(provided, expected)

    def _handle_stats_route(self) -> bool:
        parsed = urlparse(self.path)
        if not parsed.path.startswith("/_tarpit/"):
            return False

        if not self._is_stats_authorized():
            self._send_json(401, {"detail": "Invalid or missing X-Lyrics-Sync-Token"})
            return True

        qs = parse_qs(parsed.query)
        if parsed.path == "/_tarpit/stats":
            try:
                recent = int((qs.get("recent") or ["20"])[0])
            except ValueError:
                recent = 20
            self._send_json(200, STORE.stats(recent_limit=recent))
            return True

        if parsed.path == "/_tarpit/events":
            try:
                limit = int((qs.get("limit") or ["50"])[0])
            except ValueError:
                limit = 50
            items = STORE.events(
                limit=limit,
                category=(qs.get("category") or [None])[0],
                cve=(qs.get("cve") or [None])[0],
                ip=(qs.get("ip") or [None])[0],
            )
            self._send_json(200, {"count": len(items), "events": items})
            return True

        self._send_json(404, {"detail": "Not found"})
        return True

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
        classification: Classification = classify_request(
            path,
            reason=reason,
            user_agent=user_agent,
            extra_rules=EXTRA_RULES,
        )
        self._log_event(
            "caught",
            ip=client_ip,
            method=method,
            path=path,
            reason=reason,
            category=classification.category,
            severity=classification.severity,
            cve=classification.cve or "",
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

        STORE.record(
            {
                "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "ip": client_ip,
                "method": method,
                "path": path[:512],
                "reason": reason,
                "category": classification.category,
                "signature": classification.signature,
                "severity": classification.severity,
                "cve": classification.cve,
                "ua": user_agent[:256],
                "delay_sec": round(elapsed, 2),
                "status": status,
            }
        )

        self._log_event(
            "released",
            ip=client_ip,
            method=method,
            path=path,
            reason=reason,
            category=classification.category,
            cve=classification.cve or "",
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
        if self._handle_stats_route():
            return
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
        f"[tarpit] listening on {HOST}:{PORT} delay={DELAY_SECONDS}s status={STATUS_CODE} log={LOG_PATH}",
        flush=True,
    )
    httpd.serve_forever()
