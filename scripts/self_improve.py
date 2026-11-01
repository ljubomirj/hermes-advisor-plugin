#!/usr/bin/env python3
"""Autonomous Self-Improvement Analyzer for Advisor Plugin.

Queries state.db for recent verify_turn decisions (past 8-48 hours),
computes empirical performance metrics, categorizes failure modes,
and outputs actionable improvement targets for the autonomous self-improvement loop.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import time
from pathlib import Path

HERMES_HOME = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))
DB_PATH = Path(os.environ.get("HERMES_STATE_DB", HERMES_HOME / "state.db"))


def analyze_recent_performance(lookback_hours: float = 24.0) -> dict:
    """Query state.db for verify_turn tool results within the lookback window."""
    if not DB_PATH.exists():
        return {"error": f"Database not found at {DB_PATH}"}

    since_ts = time.time() - (lookback_hours * 3600)
    try:
        with sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT session_id, content, timestamp 
                FROM messages 
                WHERE role = 'tool' AND timestamp >= ? 
                  AND (content LIKE '%"status": "approved"%' OR content LIKE '%"status": "rejected"%')
                ORDER BY timestamp ASC
            """, (since_ts,))
            rows = cursor.fetchall()
    except Exception as exc:
        return {"error": f"Failed to query state.db: {exc}"}

    total_audits = len(rows)
    if total_audits == 0:
        return {
            "lookback_hours": lookback_hours,
            "total_audits": 0,
            "approved": 0,
            "rejected": 0,
            "approval_rate_pct": 0.0,
            "categories": {},
            "top_rejection_reasons": [],
            "actionable_recommendations": ["No recent audits found in window. Continue monitoring."]
        }

    approved_count = 0
    rejected_count = 0
    reasons = []

    for _, content, _ in rows:
        is_approved = '"status": "approved"' in content
        is_rejected = '"status": "rejected"' in content
        if is_approved:
            approved_count += 1
        elif is_rejected:
            rejected_count += 1
            try:
                parsed = json.loads(content)
                reasons.append(parsed.get("reason", ""))
            except Exception:
                reasons.append(content)

    pat_unsupported = re.compile(r"(?i)\b(aren't|isn't|not|no)\s+(backed|supported|tool)\b|overstate|unverified|unsupported|never checked|stronger than|unrun|never finished checking")
    pat_omitted = re.compile(r"(?i)\b(omitted|missing (context|details|trade-off|comparison)|skips|leaves out|one-sided)\b")
    pat_directive = re.compile(r"(?i)\b(directive|standing rule|explicit instruction|user asked to not|violated)\b")
    pat_arithmetic = re.compile(r"(?i)\b(arithmetic|miscount|math|miscalculated|wrong sum)\b")
    pat_contradiction = re.compile(r"(?i)\b(contradict|misreport|false claim|invalid fix)\b")
    pat_unanswered = re.compile(r"(?i)\b(does not|didn't|doesn't|never)\s+answer\b")
    pat_unsanctioned = re.compile(r"(?i)\b(unsanctioned|without asking|without running|admin-merge|bypassing|missing.*gate)\b")
    pat_empty = re.compile(r"(?i)\b(no draft|placeholder|empty draft)\b")
    pat_style = re.compile(r"(?i)\b(bureaucratic|affidavit|defensive|pedantic|jargon)\b")

    categories = {
        "unsupported_claim_or_overstatement": 0,
        "user_directive_violation": 0,
        "omitted_context_or_tradeoff": 0,
        "arithmetic_or_count_error": 0,
        "contradiction_or_invalid_fix": 0,
        "unanswered_user_ask": 0,
        "unsanctioned_action_or_skipped_gate": 0,
        "empty_or_placeholder_draft": 0,
        "style_or_bureaucratic_tone": 0,
        "other": 0,
    }

    for r in reasons:
        if pat_unsupported.search(r):
            categories["unsupported_claim_or_overstatement"] += 1
        elif pat_directive.search(r):
            categories["user_directive_violation"] += 1
        elif pat_omitted.search(r):
            categories["omitted_context_or_tradeoff"] += 1
        elif pat_arithmetic.search(r):
            categories["arithmetic_or_count_error"] += 1
        elif pat_contradiction.search(r):
            categories["contradiction_or_invalid_fix"] += 1
        elif pat_unanswered.search(r):
            categories["unanswered_user_ask"] += 1
        elif pat_unsanctioned.search(r):
            categories["unsanctioned_action_or_skipped_gate"] += 1
        elif pat_empty.search(r):
            categories["empty_or_placeholder_draft"] += 1
        elif pat_style.search(r):
            categories["style_or_bureaucratic_tone"] += 1
        else:
            categories["other"] += 1

    recommendations = []
    ranked_cats = [
        (cat, count) for cat, count in categories.items() if cat != "other" and count > 0
    ]
    ranked_cats.sort(key=lambda x: x[1], reverse=True)
    if ranked_cats:
        top_cat, top_count = ranked_cats[0]
        if top_cat == "unsupported_claim_or_overstatement":
            recommendations.append(
                f"Unsupported claims/overstatements are leading failure mode ({top_count} occurrences): "
                "add deterministic ledger rules in ledger.py or expand active read-only probes in prober.py."
            )
        elif top_cat == "user_directive_violation":
            recommendations.append(
                f"User directive violations are leading failure mode ({top_count} occurrences): "
                "strengthen directive extraction and active constraint checks."
            )
        elif top_cat == "omitted_context_or_tradeoff":
            recommendations.append(
                f"Omitted context/trade-offs are leading failure mode ({top_count} occurrences): "
                "ensure comparative evaluations surface both sides."
            )

    return {
        "lookback_hours": lookback_hours,
        "total_audits": total_audits,
        "approved": approved_count,
        "rejected": rejected_count,
        "approval_rate_pct": round((approved_count / total_audits) * 100, 1),
        "rejection_rate_pct": round((rejected_count / total_audits) * 100, 1),
        "categories": categories,
        "sample_rejection_reasons": reasons[-5:] if reasons else [],
        "actionable_recommendations": recommendations,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze recent advisor performance metrics.")
    parser.add_argument("--hours", type=float, default=24.0, help="Lookback window in hours (default: 24.0)")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    args = parser.parse_args()

    metrics = analyze_recent_performance(lookback_hours=args.hours)
    if args.json:
        print(json.dumps(metrics, indent=2))
        return

    print("=" * 60)
    print(f"ADVISOR PERFORMANCE REPORT (Past {metrics.get('lookback_hours')} Hours)")
    print("=" * 60)
    if "error" in metrics:
        print(f"Error: {metrics['error']}")
        sys.exit(1)

    print(f"Total Audits:    {metrics['total_audits']}")
    print(f"Approved:        {metrics['approved']} ({metrics['approval_rate_pct']}%)")
    print(f"Rejected:        {metrics['rejected']} ({metrics['rejection_rate_pct']}%)")
    print("\nCategorized Failure Modes:")
    for cat, count in sorted(metrics["categories"].items(), key=lambda x: x[1], reverse=True):
        pct = (count / max(1, metrics['rejected'])) * 100
        print(f"  - {cat:28}: {count:3} ({pct:.1f}% of rejections)")

    print("\nActionable Improvement Targets:")
    for rec in metrics.get("actionable_recommendations", []):
        print(f"  * {rec}")

    if metrics.get("sample_rejection_reasons"):
        print("\nRecent Rejection Snippets:")
        for i, s in enumerate(metrics["sample_rejection_reasons"][-3:], 1):
            clean_s = s.strip().replace("\n", " ")[:150]
            print(f"  {i}. {clean_s}...")


if __name__ == "__main__":
    main()
