"""Unit tests for the rule-based tarpit classifier, persistence store, and stats API."""

from __future__ import annotations

import json
import sys
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

TARPIT_DIR = str(Path(__file__).resolve().parent.parent / "tarpit")
if TARPIT_DIR not in sys.path:
    sys.path.insert(0, TARPIT_DIR)

import classifier  # noqa: E402
import server as tarpit_server  # noqa: E402


def test_classifier_matches_observed_cves_and_exploit_classes():
    c_gravity = classifier.classify_request(
        "/?rest_route=/gravitysmtp/v1/tests/mock-data&page=gravitysmtp-settings",
        reason="query",
    )
    assert c_gravity.cve == "CVE-2026-4020"
    assert c_gravity.category == "wordpress-plugin"
    assert c_gravity.severity == "high"

    c_sqli = classifier.classify_request(
        "/?band=1)%20AND%201=CAST((SELECT%20version())%20AS%20INT)--",
        reason="query",
    )
    assert c_sqli.category == "sqli"
    assert c_sqli.severity == "high"

    c_cisco = classifier.classify_request("/%2BCSCOE%2B/logon.html", reason="path")
    assert c_cisco.cve == "CVE-2020-3452"
    assert c_cisco.category == "vpn-appliance"

    c_llm = classifier.classify_request("/chat/completions", reason="path")
    assert c_llm.category == "llm-api-scan"

    c_docs = classifier.classify_request("/openapi.json", reason="docs-scanner")
    assert c_docs.category == "api-recon"


def test_classifier_loads_extra_json_rules(tmp_path):
    sig_file = tmp_path / "sigs.json"
    sig_file.write_text(
        json.dumps(
            [
                {
                    "pattern": "/custom-zero-day",
                    "category": "custom-rce",
                    "signature": "Custom Probe",
                    "severity": "critical",
                    "cve": "CVE-2026-9999",
                }
            ]
        ),
        encoding="utf-8",
    )
    extra = classifier.load_extra_rules(str(sig_file))
    hit = classifier.classify_request("/custom-zero-day?x=1", extra_rules=extra)
    assert hit.cve == "CVE-2026-9999"
    assert hit.category == "custom-rce"
    assert hit.severity == "critical"


def test_tarpit_store_persists_and_filters_across_restart(tmp_path):
    log_file = tmp_path / "events.jsonl"
    store1 = tarpit_server.TarpitStore(log_file)
    store1.record(
        {
            "ts": "2026-11-27T12:00:00+00:00",
            "ip": "1.2.3.4",
            "method": "GET",
            "path": "/?rest_route=/gravitysmtp/v1/tests/mock-data",
            "reason": "query",
            "category": "wordpress-plugin",
            "signature": "Gravity SMTP Unauth Info Disclosure",
            "severity": "high",
            "cve": "CVE-2026-4020",
            "ua": "Go-http-client/1.1",
            "delay_sec": 20.0,
            "status": "client_gone",
        }
    )

    store2 = tarpit_server.TarpitStore(log_file)
    stats = store2.stats()
    assert stats["total_events"] == 1
    assert stats["client_gone_count"] == 1
    assert stats["total_wasted_seconds"] == 20.0
    assert stats["by_cve"] == {"CVE-2026-4020": 1}
    assert stats["top_ips"] == [{"ip": "1.2.3.4", "count": 1}]
    assert len(store2.events(cve="CVE-2026-4020")) == 1
    assert store2.events(category="sqli") == []


def test_tarpit_http_handler_token_gates_stats_and_records_catches(tmp_path, monkeypatch):
    monkeypatch.setenv("LYRIC_SYNC_CALLBACK_SECRET", "top-secret-tarpit")
    monkeypatch.setattr(tarpit_server, "DELAY_SECONDS", 0.01)
    monkeypatch.setattr(tarpit_server, "STORE", tarpit_server.TarpitStore(tmp_path / "live.jsonl"))

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), tarpit_server.TarpitHandler)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{port}"

        # 1. Trigger a tarpit catch
        req = urllib.request.Request(
            f"{base}/?rest_route=/gravitysmtp/v1/tests/mock-data&page=gravitysmtp-settings",
            headers={"X-Forwarded-For": "9.8.7.6", "X-Tarpit-Reason": "query", "User-Agent": "Nuclei/3.0"},
        )
        try:
            urllib.request.urlopen(req, timeout=5)
            assert False, "Expected 404 from tarpit"
        except urllib.error.HTTPError as exc:
            assert exc.code == 404

        # 2. /_tarpit/stats without token -> 401
        try:
            urllib.request.urlopen(f"{base}/_tarpit/stats", timeout=5)
            assert False, "Expected 401 without token"
        except urllib.error.HTTPError as exc:
            assert exc.code == 401

        # 3. /_tarpit/stats with valid token -> 200 with CVE-2026-4020
        stats_req = urllib.request.Request(
            f"{base}/_tarpit/stats",
            headers={"X-Lyrics-Sync-Token": "top-secret-tarpit"},
        )
        body = json.loads(urllib.request.urlopen(stats_req, timeout=5).read().decode("utf-8"))
        assert body["total_events"] == 1
        assert body["by_cve"] == {"CVE-2026-4020": 1}
        assert body["recent"][0]["ip"] == "9.8.7.6"

        # 4. /_tarpit/events filtered by CVE
        ev_req = urllib.request.Request(
            f"{base}/_tarpit/events?cve=CVE-2026-4020",
            headers={"X-Lyrics-Sync-Token": "top-secret-tarpit"},
        )
        ev_body = json.loads(urllib.request.urlopen(ev_req, timeout=5).read().decode("utf-8"))
        assert ev_body["count"] == 1
        assert ev_body["events"][0]["signature"] == "Gravity SMTP Unauth Info Disclosure"
    finally:
        httpd.shutdown()
        httpd.server_close()
