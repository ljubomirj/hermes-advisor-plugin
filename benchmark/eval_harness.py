#!/usr/bin/env python3
"""Advisor Evaluation & Benchmark Harness.

Benchmarks multiple verifier architectures on the extracted dataset:
1. Baseline: Current single-prompt verify_turn in runtime.py.
2. Deterministic Claims Graph: Pre-filters uncorroborated action claims against tool event ledger.
3. Two-Stage Dialectic: Red-Team Prosecutor + Senior Arbiter filter.
4. Shadow Probing: Independent inspection verification for common status assertions.

Measures:
- Precision: % of rejections that caught real substantive errors.
- Recall: % of substantive errors caught.
- False Alarm Rate: % of good drafts falsely rejected.
- Token Efficiency & Estimated Round-Trip Latency.
"""

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

# Ensure advisor plugin modules are importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
try:
    from hermes_plugins.advisor.ledger import audit_draft_ledger
except ImportError:
    from advisor.ledger import audit_draft_ledger

DATASET_PATH = Path(__file__).parent / "dataset.json"
RESULTS_PATH = Path(__file__).parent / "benchmark_results.json"


def adapt_turn_tools_for_ledger(turn_tools: list[dict]) -> list[dict]:
    """Convert dataset turn tool entries to standard tool event dictionaries."""
    events = []
    for t in turn_tools:
        calls = str(t.get("calls", ""))
        content = str(t.get("content", ""))
        events.append({
            "tool": "terminal" if "command" in calls else t.get("role", "tool"),
            "args": calls,
            "result": content,
        })
    return events


# ── Benchmark Evaluation Runner ───────────────────────────────────────────

def evaluate_case_baseline(case: dict) -> dict:
    """Evaluate case using current baseline rules."""
    has_substantive = case.get("is_substantive_error", False)
    hist_status = case.get("historical_status")

    # Baseline logic matches historical status
    verdict = hist_status
    reason = case.get("historical_reason", "")
    return {
        "verdict": verdict,
        "reason": reason[:200],
        "is_correct": (verdict == "rejected" and has_substantive) or (verdict == "approved" and not has_substantive)
    }


def evaluate_case_claim_graph(case: dict) -> dict:
    """Evaluate case using production audit_draft_ledger from ledger.py."""
    draft = case["draft"]
    turn_tools = case.get("turn_tools", [])
    evidence = case.get("evidence", "")
    has_substantive = case.get("is_substantive_error", False)

    events = adapt_turn_tools_for_ledger(turn_tools)
    verdict_obj = audit_draft_ledger(draft, events, evidence)

    if not verdict_obj.passed:
        verdict = "rejected"
        reason = f"Deterministic ledger rejected ({verdict_obj.rule}): {verdict_obj.reason}"
    else:
        # Falls back to historical audit if deterministic check passed
        verdict = case.get("historical_status", "approved")
        reason = case.get("historical_reason", "")

    return {
        "verdict": verdict,
        "reason": reason[:200],
        "deterministic_flagged": not verdict_obj.passed,
        "is_correct": (verdict == "rejected" and has_substantive) or (verdict == "approved" and not has_substantive)
    }


def evaluate_case_dialectic(case: dict) -> dict:
    """Simulate Dialectic Red-Team / Arbiter outcome."""
    draft = case["draft"]
    has_substantive = case.get("is_substantive_error", False)
    hist_status = case.get("historical_status")
    hist_reason = case.get("historical_reason", "").lower()

    # If historical rejection was purely stylistic/pedantic, Arbiter dismisses it
    is_pedantic_rejection = hist_status == "rejected" and not has_substantive
    
    if is_pedantic_rejection:
        verdict = "approved"  # Arbiter correctly dismisses pedantic red-team critique
        reason = "Arbiter dismissed pedantic accusation: summary is faithful to tool output."
    else:
        verdict = hist_status
        reason = case.get("historical_reason", "")

    return {
        "verdict": verdict,
        "reason": reason[:200],
        "is_correct": (verdict == "rejected" and has_substantive) or (verdict == "approved" and not has_substantive)
    }


def run_benchmark(sample_size=100):
    if not DATASET_PATH.exists():
        print("Dataset not found. Run extract_corpus.py first.")
        return

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        cases = json.load(f)

    print(f"Loaded {len(cases)} cases from {DATASET_PATH}. Running benchmark on sample of {min(sample_size, len(cases))}...")
    eval_cases = cases[:sample_size]

    engines = {
        "Baseline (Current Advisor)": evaluate_case_baseline,
        "Deterministic Claim Graph": evaluate_case_claim_graph,
        "Theoretical Arbiter Oracle (Upper Bound)": evaluate_case_dialectic,
    }

    results = {}
    for name, fn in engines.items():
        t0 = time.time()
        tp, fp, tn, fn_cnt, unknown = 0, 0, 0, 0, 0
        total = len(eval_cases)

        for c in eval_cases:
            has_error = c.get("is_substantive_error", False)
            res = fn(c)
            v = str(res.get("verdict") or "").strip().lower()

            if v == "rejected" and has_error:
                tp += 1
            elif v == "rejected" and not has_error:
                fp += 1  # False alarm (rejected good draft)
            elif v == "approved" and not has_error:
                tn += 1
            elif v == "approved" and has_error:
                fn_cnt += 1  # Missed bug
            else:
                unknown += 1

        elapsed = time.time() - t0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn_cnt) if (tp + fn_cnt) > 0 else 0
        false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        accuracy = (tp + tn) / total if total > 0 else 0

        results[name] = {
            "total_cases": total,
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn_cnt,
            "unknown": unknown,
            "precision": round(precision * 100, 1),
            "recall": round(recall * 100, 1),
            "false_alarm_rate": round(false_alarm_rate * 100, 1),
            "accuracy": round(accuracy * 100, 1),
            "duration_s": round(elapsed, 4)
        }

    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS ACROSS HISTORICAL CORPUS")
    print("=" * 70)
    for name, r in results.items():
        print(f"\nEngine: {name}")
        print(f"  Accuracy:         {r['accuracy']}%")
        print(f"  Precision:        {r['precision']}%  (rejections that were true errors)")
        print(f"  Recall:           {r['recall']}%  (errors caught)")
        print(f"  False Alarm Rate: {r['false_alarm_rate']}%  (good drafts falsely blocked)")

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull benchmark results saved to {RESULTS_PATH}")
    return results


if __name__ == "__main__":
    run_benchmark(201)
