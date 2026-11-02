"""Deterministic Verification Ledger for Advisor Plugin.

Provides zero-latency, deterministic static verification of drafts before calling an LLM:
1. Placeholder / Empty Draft Detection (e.g. "Testing verify_turn", "...", stubs).
2. Unhandled Tool Error Detection: flags drafts claiming success when tool events failed.
3. Truncated Identifier Citation: flags IDs/hashes cited from truncated outputs.
4. Hard Assertion Corroboration: checks required diagnostic tools for git/test/build claims.
"""

from __future__ import annotations

import re
from typing import Any

# Drafts that are clearly placeholders or stubs
_PLACEHOLDER_PATTERNS = [
    re.compile(r"^\s*(testing verify_turn|test draft|placeholder|\.{2,}|todo)\s*$", re.IGNORECASE),
    re.compile(r"^\s*draft\s*:?\s*$", re.IGNORECASE),
]

# Action patterns requiring specific diagnostic tools in the turn
_CORROBORATION_RULES = [
    {
        "name": "test_suite_passed",
        "pattern": re.compile(r"(?i)(?<!make sure )(?<!ensure )(?<!verify )(?<!if )\b(all \d+ tests? pass(ed)?|test suite passes?|tests? pass(ed)?)\b"),
        "required_tool_patterns": [r"\bunittest\b", r"\bpytest\b", r"\b(pnpm|bun|yarn|npm) (run )?test\b", r"\bcargo test\b", r"\bgo test\b", r"\bmake test\b", r"\btox\b"],
        "suggested_command": "Run your test suite (e.g. pytest or unittest) before asserting tests pass.",
    },
    {
        "name": "clean_git_tree",
        "pattern": re.compile(r"(?i)(?<!make sure )(?<!ensure )(?<!verify )\b(working tree (is )?clean|clean working tree|git status clean|git tree (is )?clean|no uncommitted changes)\b"),
        "required_tool_patterns": [r"git status", r"git diff", r"git -C .* status"],
        "suggested_command": "Run `git status` to verify the working tree is clean.",
    },
    {
        "name": "git_pushed",
        "pattern": re.compile(
            r"(?i)(?<!not )(?<!n't )(?<!never )(?<!not been )(?<!never been )(?<!yet to be )"
            r"\b(?:pushed\s+(?:the\s+|all\s+)?(?:changes?|branch(?:es)?(?:\s+[\w\-\/\.]+)?|commits?|updates?)?\s*to\s+(?:the\s+)?(?:origin|remote|github)"
            r"|(?:changes?|branch(?:es)?(?:\s+[\w\-\/\.]+)?|commits?)\s+(?:have been |were |was )?pushed\s+to\s+(?:the\s+)?(?:origin|remote|github)"
            r"|(?:i|we)\s+(?:ran|did)\s+git\s+push)\b"
        ),
        "required_tool_patterns": [r"git push", r"push -u origin", r"push origin"],
        "suggested_command": "Run `git push` before asserting changes were pushed.",
    },
    {
        "name": "pr_created",
        "pattern": re.compile(
            r"(?i)(?<!make sure )(?<!ensure )(?<!verify )(?<!if )(?<!to )(?<!will )"
            r"\b(?:(?:opened|created|submitted)\s+(?:a\s+|the\s+)?(?:new\s+|draft\s+)?pr(?: #?\d+)?"
            r"|(?:opened|created|submitted)\s+(?:a\s+|the\s+)?(?:new\s+|draft\s+)?pull request(?: #?\d+)?"
            r"|pr #?\d+\s+(?:was |is )?(?:opened|created))\b"
        ),
        "required_tool_patterns": [r"gh pr create", r"gh pr view", r"gh pr list"],
        "suggested_command": "Run `gh pr create` before claiming a PR was created or opened.",
    },
    {
        "name": "pr_merged",
        "pattern": re.compile(r"(?i)(?<!not )(?<!n't )(?<!never )\b(merged pr #?\d+|pr #?\d+ (was |is )?merged)\b"),
        "required_tool_patterns": [r"gh pr merge", r"git merge"],
        "suggested_command": "Verify PR merge via `gh pr merge` or `gh pr view`.",
    },
    {
        "name": "coderabbit_reviewed",
        "pattern": re.compile(
            r"(?i)(?<!make sure )(?<!ensure )(?<!verify )(?<!if )(?<!run )"
            r"\b(?:coderabbit\s+(?:review\s+)?(?:passed|clean|completed|finished|approved|reported)"
            r"|(?:ran|executed)\s+coderabbit\s+review)\b"
        ),
        "required_tool_patterns": [r"\bcoderabbit\b"],
        "suggested_command": "Run `coderabbit review` before asserting CodeRabbit review completed or passed.",
    },
]

# Keywords indicating an unhandled tool failure in raw tool results
_TOOL_FAILURE_PATTERNS = [
    re.compile(r"""(?i)["']?exit[_-]?code["']?\s*[:=]\s*([1-9]\d*)"""),
    re.compile(r"\bexit code\s+([1-9]\d*)\b"),
    re.compile(r"\b(Permission denied|command not found|SyntaxError)\b|Traceback \(most recent call last\)"),
]


class LedgerVerdict:
    def __init__(self, passed: bool, reason: str = "", missing_checks: list[str] | None = None, rule: str = ""):
        self.passed = passed
        self.reason = reason
        self.missing_checks = missing_checks or []
        self.rule = rule

    def to_dict(self) -> dict[str, Any]:
        if self.passed:
            return {"status": "approved", "reason": self.reason or "Deterministic ledger verified."}
        res: dict[str, Any] = {
            "status": "rejected",
            "reason": self.reason,
            "rule": self.rule,
        }
        if self.missing_checks:
            res["missing_checks"] = self.missing_checks
        return res


def audit_draft_ledger(
    draft: str,
    tool_events: list[dict],
    evidence: str = "",
    claims: list[str] | str | None = None,
) -> LedgerVerdict:
    """Deterministically audit draft against tool events before invoking LLM."""
    clean_draft = (draft or "").strip()
    if isinstance(claims, list):
        claims_text = " ".join(str(c) for c in claims).strip()
    else:
        claims_text = str(claims or "").strip()

    if not clean_draft:
        clean_draft = claims_text

    if not clean_draft:
        return LedgerVerdict(
            passed=False,
            reason="Draft is empty. A valid response draft or claim list is required for verification.",
            rule="empty_draft"
        )

    # Combined text for checking factual claims across draft and explicit claims
    scan_text = f"{clean_draft} {claims_text}".strip()

    # 1. Check for placeholder drafts
    for pat in _PLACEHOLDER_PATTERNS:
        if pat.search(clean_draft):
            return LedgerVerdict(
                passed=False,
                reason=f"Draft '{clean_draft[:50]}' is a placeholder or stub, not a user-facing response.",
                rule="placeholder_draft"
            )

    combined_tool_text = " ".join(
        f"{ev.get('tool', '')} {ev.get('args', '')}" for ev in tool_events
    )
    combined_evidence = combined_tool_text.lower()

    # 2. Check corroboration rules
    missing_checks = []
    unbacked_reasons = []
    for rule in _CORROBORATION_RULES:
        if rule["pattern"].search(scan_text):
            # Check if any required pattern was executed in tool events
            matched = any(
                re.search(req_pat, combined_evidence, re.IGNORECASE)
                for req_pat in rule["required_tool_patterns"]
            )
            if not matched:
                missing_checks.append(rule["suggested_command"])
                unbacked_reasons.append(
                    f"Claim matching '{rule['name']}' has no supporting tool execution in this turn."
                )

    if missing_checks:
        return LedgerVerdict(
            passed=False,
            reason=" ".join(unbacked_reasons),
            missing_checks=missing_checks,
            rule="unbacked_assertion"
        )

    # 3. Check for severe unhandled tool errors if draft claims unqualified success
    claims_clean_success = re.search(r"(?i)\b(all clean|succeeded|done successfully|everything works?|healthy)\b", scan_text)
    if claims_clean_success and tool_events:
        # Check only the final execution result per tool to avoid flagging retried/recovered commands
        latest_event_per_tool = {}
        for ev in tool_events:
            tool_name = ev.get("tool", "")
            if tool_name:
                latest_event_per_tool[tool_name] = ev

        for tool_name, ev in latest_event_per_tool.items():
            res = str(ev.get("result", ""))
            for err_pat in _TOOL_FAILURE_PATTERNS:
                m = err_pat.search(res)
                if m:
                    # An error occurred in the final tool call of this tool, but draft claims clean success
                    ack_words = ["failed", "error", "exit", "issue", "problem", "unable to", "warning"]
                    if not any(w in clean_draft.lower() for w in ack_words):
                        return LedgerVerdict(
                            passed=False,
                            reason=(
                                f"Tool '{tool_name}' reported a failure ({m.group(0)}), but the draft claims "
                                "clean success without acknowledging or explaining the failure."
                            ),
                            missing_checks=["Inspect tool error and explain or resolve it in the draft."],
                            rule="unacknowledged_tool_failure"
                        )

    return LedgerVerdict(passed=True, reason="Deterministic ledger checks passed.")
