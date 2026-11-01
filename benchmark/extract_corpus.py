#!/usr/bin/env python3
"""Extract a benchmark dataset of historical verify_turn scenarios from state.db."""

import json
import os
import re
import sqlite3
from pathlib import Path

DB_PATH = Path(os.environ.get("HERMES_STATE_DB", Path.home() / ".hermes" / "state.db"))
OUTPUT_PATH = Path(__file__).parent / "dataset.json"


def extract_benchmark_cases(limit_per_session=15):
    with sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True) as conn:
        c = conn.cursor()
        # Find sessions with significant verify_turn usage
        c.execute("""
            SELECT session_id, count(*) 
            FROM messages 
            WHERE role = 'tool' AND (content LIKE '%"status": "approved"%' OR content LIKE '%"status": "rejected"%')
            GROUP BY session_id
            HAVING count(*) >= 10
            ORDER BY count(*) DESC
        """)
        top_sessions = [r[0] for r in c.fetchall()]

        cases = []
        for sess in top_sessions:
            c.execute("""
                SELECT id, role, content, tool_calls, timestamp 
                FROM messages 
                WHERE session_id = ? 
                ORDER BY timestamp ASC
            """, (sess,))
            rows = c.fetchall()

            session_cases = []
            for i, r in enumerate(rows):
                msg_id, role, content, tool_calls, ts = r
                if role == "assistant" and tool_calls and "verify_turn" in tool_calls:
                    # Parse draft
                    draft = ""
                    evidence = ""
                    try:
                        tc = json.loads(tool_calls)
                        for call in tc:
                            fn = call.get("function", {})
                            name = fn.get("name")
                            args = json.loads(fn.get("arguments", "{}"))
                            if name == "verify_turn":
                                draft = args.get("draft", "")
                                evidence = args.get("evidence", "")
                            elif name == "tool_call":
                                for sub in args.get("calls", []):
                                    if sub.get("name") == "verify_turn":
                                        sub_args = sub.get("arguments", {})
                                        draft = sub_args.get("draft", "")
                                        evidence = sub_args.get("evidence", "")
                    except Exception:
                        pass

                    if not draft:
                        continue

                    # Look for auditor response in immediate subsequent tool messages
                    auditor_status = None
                    auditor_reason = ""
                    for j in range(i + 1, min(i + 6, len(rows))):
                        next_role, next_content = rows[j][1], rows[j][2]
                        if next_role == "tool" and next_content and ("status" in next_content):
                            try:
                                d = json.loads(next_content)
                                if "status" in d:
                                    auditor_status = d.get("status")
                                    auditor_reason = d.get("reason", "")
                                    break
                            except Exception:
                                pass

                    if not auditor_status:
                        continue

                    # Find prior tool calls in this turn (tools run before this verify_turn)
                    turn_tools = []
                    k = i - 1
                    while k >= 0:
                        prev_role, prev_content, prev_tc = rows[k][1], rows[k][2], rows[k][3]
                        if prev_role == "user":
                            break
                        if prev_role == "tool" and prev_content:
                            turn_tools.append({"role": "tool", "content": prev_content[:2000]})
                        elif prev_role == "assistant" and prev_tc:
                            turn_tools.append({"role": "assistant_tools", "calls": prev_tc[:1000]})
                        k -= 1
                    turn_tools.reverse()

                    # Find latest user prompt before this turn
                    user_prompt = ""
                    recent_dialogue = []
                    k = i - 1
                    user_count = 0
                    while k >= 0 and user_count < 4:
                        prev_role, prev_content = rows[k][1], rows[k][2]
                        if prev_role == "user" and prev_content:
                            if not user_prompt:
                                user_prompt = prev_content
                            user_count += 1
                            recent_dialogue.append({"role": "user", "content": prev_content[:500]})
                        elif prev_role == "assistant" and prev_content and len(prev_content) > 10:
                            recent_dialogue.append({"role": "assistant", "content": prev_content[:500]})
                        k -= 1
                    recent_dialogue.reverse()

                    # Classify if the historical rejection caught a substantive bug
                    reason_low = auditor_reason.lower()
                    is_substantive = False
                    if auditor_status == "rejected":
                        substantive_markers = [
                            "contradict", "unverified", "no tool output shows", "not supported",
                            "never checked", "misreport", "false claim", "exited 124", "permission denied",
                            "did not review", "unauthorized", "without asking", "failed"
                        ]
                        if any(m in reason_low for m in substantive_markers):
                            is_substantive = True

                    session_cases.append({
                        "case_id": f"{sess}_{msg_id}",
                        "session_id": sess,
                        "user_prompt": user_prompt[:800],
                        "recent_dialogue": recent_dialogue,
                        "turn_tools": turn_tools,
                        "draft": draft,
                        "evidence": evidence,
                        "historical_status": auditor_status,
                        "historical_reason": auditor_reason,
                        "is_substantive_error": is_substantive,
                    })

                    if len(session_cases) >= limit_per_session:
                        break

            cases.extend(session_cases)

        print(f"Extracted {len(cases)} benchmark cases across {len(top_sessions)} sessions.")
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(cases, f, indent=2)
        print(f"Saved dataset to {OUTPUT_PATH}")
        return cases


if __name__ == "__main__":
    extract_benchmark_cases()
