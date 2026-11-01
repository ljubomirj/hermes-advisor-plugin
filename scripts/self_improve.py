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

    categories = {
        "user_directive_violation": 0,
        "unverified_assertion": 0,
        "tool_output_contradiction": 0,
        "style_or_pedantic_omission": 0,
        "missing_action_or_cron": 0,
        "other": 0,
    }

    for r in reasons:
        rl = r.lower()
        if "directive" in rl or "user said" in rl or "user asked" in rl or "alejandro" in rl or "constraint" in rl:
            categories["user_directive_violation"] += 1
        elif "unverified" in rl or "no tool" in rl or "not supported" in rl or "never checked" in rl or "unrun" in rl:
            categories["unverified_assertion"] += 1
        elif "contradict" in rl or "misreport" in rl or "false" in rl or "failed" in rl:
            categories["tool_output_contradiction"] += 1
        elif "cron" in rl or "action" in rl or "scheduled" in rl:
            categories["missing_action_or_cron"] += 1
        elif "pedantic" in rl or "style" in rl or "paragraph" in rl or "bureaucratic" in rl:
            categories["style_or_pedantic_omission"] += 1
        else:
            categories["other"] += 1

    recommendations = []
    if categories["user_directive_violation"] > categories["unverified_assertion"]:
        recommendations.append(
            "User directive violations are leading failure mode: strengthen directive extraction "
            "and check for negative lookaheads in agent assumptions."
        )
    if categories["unverified_assertion"] > 0:
        recommendations.append(
            "Unverified assertions detected: identify recurring diagnostic commands and add "
            "deterministic ledger rules or active prober checks in prober.py."
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
