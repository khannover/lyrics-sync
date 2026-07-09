"""Shared validators for real /sync output (LRC, report JSON, embedded SYLT)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Tuple

_LRC_TS = re.compile(r"\[(\d{2}):(\d{2})\.(\d{2,3})\]")


def parse_lrc_timestamps_ms(lrc_text: str) -> List[int]:
    """Return timestamps in ms for each non-empty LRC line (first tag per line)."""
    out: List[int] = []
    for raw in lrc_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = _LRC_TS.search(line)
        if not m:
            continue
        mm, ss, frac = m.groups()
        frac_ms = int(frac.ljust(3, "0")[:3])
        out.append((int(mm) * 60 + int(ss)) * 1000 + frac_ms)
    return out


def check_monotonic_timestamps_ms(timestamps_ms: Iterable[int]) -> Tuple[bool, Optional[str]]:
    prev = -1
    for i, ts in enumerate(timestamps_ms):
        if ts < prev:
            return False, f"timestamp regression at index {i}: {ts}ms < {prev}ms"
        prev = ts
    return True, None


@dataclass
class SyncQualityVerdict:
    ok: bool
    quality: str
    warnings: List[str]
    report: dict
    issues: List[str]

    def summary(self) -> str:
        lines = [
            f"quality={self.quality}",
            f"ok={self.ok}",
        ]
        if self.report:
            lines.append(
                "report: "
                + ", ".join(
                    f"{k}={self.report.get(k)}"
                    for k in ("line_count", "duration_ms", "whisper_word_count")
                    if k in self.report
                )
            )
        if self.warnings:
            lines.append("warnings: " + "; ".join(self.warnings))
        if self.issues:
            lines.append("issues: " + "; ".join(self.issues))
        return "\n".join(lines)


def evaluate_sync_report(
    report: dict,
    *,
    lrc_text: str,
    allow_fallback: bool = False,
    min_coverage_ratio: float = 0.35,
    min_whisper_words: int = 5,
    min_lrc_lines: int = 3,
) -> SyncQualityVerdict:
    """Apply product-quality gates aligned with bancamp lyrics_sync_quality."""
    issues: List[str] = []
    quality = str(report.get("quality") or "unknown")
    warnings = list(report.get("warnings") or [])

    ts = parse_lrc_timestamps_ms(lrc_text)
    if len(ts) < min_lrc_lines:
        issues.append(f"LRC has only {len(ts)} timed lines (need >={min_lrc_lines})")

    mono_ok, mono_err = check_monotonic_timestamps_ms(ts)
    if not mono_ok and mono_err:
        issues.append(mono_err)

    duration_ms = int(report.get("duration_ms") or 0)
    if duration_ms > 0 and ts:
        coverage = max(ts) / duration_ms
        if coverage < min_coverage_ratio:
            issues.append(
                f"low timestamp coverage: max_ts/duration={coverage:.2f} < {min_coverage_ratio}"
            )
    elif report.get("legacy_zip") and ts:
        # Without duration_ms, require timestamps to span at least 60s of song.
        if max(ts) < 60_000:
            issues.append(f"legacy ZIP: max LRC timestamp {max(ts)}ms < 60000ms")

    whisper_words = int(report.get("whisper_word_count") or 0)
    if whisper_words < min_whisper_words:
        if report.get("legacy_zip") and whisper_words < 0:
            pass
        else:
            issues.append(f"whisper_word_count={whisper_words} < {min_whisper_words}")

    line_count = int(report.get("line_count") or 0)
    if line_count < min_lrc_lines:
        issues.append(f"report line_count={line_count} < {min_lrc_lines}")

    if quality == "fallback" and not allow_fallback:
        issues.append("quality is fallback (not allowed for this gate)")
    elif quality not in ("good", "degraded", "fallback"):
        issues.append(f"unknown quality value: {quality}")

    ok = len(issues) == 0
    return SyncQualityVerdict(
        ok=ok,
        quality=quality,
        warnings=warnings,
        report=report,
        issues=issues,
    )


def load_report_from_zip_member(data: bytes) -> dict:
    return json.loads(data.decode("utf-8"))