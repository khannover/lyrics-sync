"""
Rule-based, zero-dependency threat classifier for the lyrics-sync tarpit.

Maps incoming scanner paths, query strings, and nginx tarpit reasons to
structured categories, known CVE IDs, and severity levels in < 1 ms.
Optionally loads additional path/query signatures from a JSON file
(e.g. generated from ProjectDiscovery's nuclei-templates).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Pattern, Tuple
from urllib.parse import unquote


@dataclass(frozen=True)
class Classification:
    category: str
    signature: str
    severity: str = "low"
    cve: Optional[str] = None


@dataclass(frozen=True)
class _Rule:
    pattern: Pattern[str]
    category: str
    signature: str
    severity: str
    cve: Optional[str] = None


_BUILTIN_RULES: Tuple[_Rule, ...] = (
    # Specific CVEs / named exploits observed in the wild
    _Rule(
        re.compile(r"gravitysmtp/v1/tests/mock-data", re.IGNORECASE),
        category="wordpress-plugin",
        signature="Gravity SMTP Unauth Info Disclosure",
        severity="high",
        cve="CVE-2026-4020",
    ),
    _Rule(
        re.compile(r"wpadverts/v1/classifieds-types", re.IGNORECASE),
        category="wordpress-plugin",
        signature="WPAdverts Broken Access Control",
        severity="medium",
        cve=None,
    ),
    _Rule(
        re.compile(r"/\+?CSCOE\+?/logon\.html", re.IGNORECASE),
        category="vpn-appliance",
        signature="Cisco ASA/FTD WebVPN Probe",
        severity="high",
        cve="CVE-2020-3452",
    ),
    _Rule(
        re.compile(r"^/geoserver(?:/|$)", re.IGNORECASE),
        category="java-rce",
        signature="GeoServer OGC Filter RCE Probe",
        severity="critical",
        cve="CVE-2024-36401",
    ),
    _Rule(
        re.compile(r"^/SDK/webLanguage(?:/|$)", re.IGNORECASE),
        category="iot-rce",
        signature="Hikvision Command Injection Probe",
        severity="critical",
        cve="CVE-2021-36260",
    ),
    _Rule(
        re.compile(r"(?:^|[?&])XDEBUG_SESSION_START=", re.IGNORECASE),
        category="php-rce",
        signature="PHP Xdebug Remote Code Execution Probe",
        severity="high",
        cve=None,
    ),
    # Broad exploit classes
    _Rule(
        re.compile(r"(?:SELECT[\s+]+|UNION[\s+]+|CAST\(|SLEEP\(|BENCHMARK\()", re.IGNORECASE),
        category="sqli",
        signature="SQL Injection Probe",
        severity="high",
    ),
    _Rule(
        re.compile(r"(?:\.\./|/etc/passwd|/proc/self)", re.IGNORECASE),
        category="path-traversal",
        signature="Path Traversal / LFI Probe",
        severity="high",
    ),
    _Rule(
        re.compile(r"(?:^/\.env(?:/|$)|(?:^|[?&])phpinfo=|(?:^|[?&])r=/getconfig|^/actuator(?:/|$)|^/metrics(?:/|$))", re.IGNORECASE),
        category="secret-disclosure",
        signature="Config / Secret Disclosure Probe",
        severity="high",
    ),
    _Rule(
        re.compile(r"(?:^/chat/completions(?:/|$)|^/mcp(?:-sse)?(?:/|$)|^/models/(?:edit|apply)(?:/|$))", re.IGNORECASE),
        category="llm-api-scan",
        signature="Exposed LLM / MCP Endpoint Scan",
        severity="medium",
    ),
    _Rule(
        re.compile(r"(?:rest_route=/wp/v2/users|post_type=revision|wp-login\.php|xmlrpc\.php|wp-admin)", re.IGNORECASE),
        category="wordpress-recon",
        signature="WordPress User / Revision Enumeration",
        severity="low",
    ),
    _Rule(
        re.compile(r"(?:rest_route=|wp-json/)", re.IGNORECASE),
        category="wordpress-plugin",
        signature="WordPress REST API Probe",
        severity="medium",
    ),
    _Rule(
        re.compile(r"(?:^/cgi-bin/|^/remote/login|^/webpages/login\.html|^/doc/index\.html)", re.IGNORECASE),
        category="appliance-exploit",
        signature="Router / Appliance Login & CGI Probe",
        severity="medium",
    ),
    _Rule(
        re.compile(r"^/(?:docs|redoc|openapi\.json)(?:/|$)", re.IGNORECASE),
        category="api-recon",
        signature="OpenAPI / Swagger Scanner",
        severity="low",
    ),
    _Rule(
        re.compile(r"^/(?:admin|manage|login|api|app|_next)(?:/|$)", re.IGNORECASE),
        category="web-recon",
        signature="Common Admin / Framework Path Scan",
        severity="low",
    ),
    _Rule(
        re.compile(r"(?:^|[?&])static=1&order=", re.IGNORECASE),
        category="web-recon",
        signature="Directory Listing / Sort Probe",
        severity="low",
    ),
)

_REASON_FALLBACKS = {
    "docs-scanner": Classification("api-recon", "Non-Browser Docs Scanner", "low"),
    "method": Classification("protocol-anomaly", "Unsupported HTTP Method", "low"),
    "http10": Classification("protocol-anomaly", "Legacy HTTP/1.0 Scanner", "low"),
    "empty-ua": Classification("bot-generic", "Empty User-Agent Probe", "low"),
    "user-agent": Classification("bot-generic", "Known Scanner User-Agent", "low"),
    "query": Classification("query-probe", "Suspicious Query String", "medium"),
    "path": Classification("web-recon", "Blocked Scanner Path", "low"),
}


def load_extra_rules(path: Optional[str] = None) -> Tuple[_Rule, ...]:
    """
    Optionally load extra JSON rules from *path* (or TARPIT_SIGNATURES_FILE).
    Expected JSON format: [{"pattern": "...", "category": "...", "signature": "...", "severity": "high", "cve": "CVE-..."}]
    """
    file_path = (path or os.environ.get("TARPIT_SIGNATURES_FILE", "")).strip()
    if not file_path:
        return ()
    p = Path(file_path)
    if not p.is_file():
        return ()
    try:
        raw: List[dict[str, Any]] = json.loads(p.read_text(encoding="utf-8", errors="replace"))
        loaded: List[_Rule] = []
        for item in raw:
            pat = str(item.get("pattern") or "").strip()
            if not pat:
                continue
            loaded.append(
                _Rule(
                    pattern=re.compile(pat, re.IGNORECASE),
                    category=str(item.get("category") or "custom"),
                    signature=str(item.get("signature") or "Custom Signature"),
                    severity=str(item.get("severity") or "medium"),
                    cve=(str(item["cve"]).strip() if item.get("cve") else None),
                )
            )
        return tuple(loaded)
    except Exception:
        return ()


def classify_request(
    path: str,
    *,
    reason: str = "unknown",
    user_agent: str = "-",
    extra_rules: Tuple[_Rule, ...] = (),
) -> Classification:
    """Classify a tarpitted request deterministically using extra + built-in regex rules."""
    decoded_path = unquote(path or "")
    for rule in (*extra_rules, *_BUILTIN_RULES):
        if rule.pattern.search(decoded_path) or rule.pattern.search(path or ""):
            return Classification(
                category=rule.category,
                signature=rule.signature,
                severity=rule.severity,
                cve=rule.cve,
            )

    fallback = _REASON_FALLBACKS.get((reason or "").strip().lower())
    if fallback:
        return fallback
    return Classification(category="unknown", signature="Unclassified Probe", severity="low")
