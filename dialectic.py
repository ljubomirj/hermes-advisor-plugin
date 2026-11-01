"""Two-Stage Dialectic Verifier for Advisor Plugin.

Architectural design separating the audit into two distinct phases:
1. Phase 1: Adversarial Red-Team Challenger (prosecutes potential discrepancies).
2. Phase 2: Senior Staff Arbiter (evaluates accusations, dismisses pedantic nitpicks, upholds true blockers).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

CHALLENGER_SYSTEM_PROMPT = """You are an Adversarial Red-Team Auditor. Your ONLY goal is to challenge the draft by finding:
1. Factual claims in the draft that have no corroborating evidence in the tool execution events.
2. Direct contradictions between the draft and tool outputs.
3. Violations of explicit user directives or standing constraints.
4. Unhandled edge cases, hidden failures, or premature claims of completion.

Be thorough and unsparing. Do NOT approve or apologize.
Output ONLY a JSON array of accusations matching this schema:
[
  {
    "claim": "the specific sentence or claim in the draft",
    "category": "unverified_claim" | "contradiction" | "directive_violation" | "hidden_failure" | "stylistic_omission",
    "evidence_gap": "exact reason why tool outputs do not corroborate this"
  }
]
If there are genuinely zero issues, return an empty array: []"""

ARBITER_SYSTEM_PROMPT = """You are a Senior Staff Arbiter. You evaluate the Adversarial Red-Team Auditor's accusations against the user's actual prompt, recorded tool outputs, and SOUL.md communication guidelines.

Your mandate:
1. FILTER OUT pedantic nitpicks and false alarms:
   - Dismiss accusations demanding additional unasked-for explanations or educational essays.
   - Dismiss accusations demanding internal plumbing details (like exit codes, temp files, or tab IDs) in the user-facing draft when the core action succeeded.
   - Dismiss accusations based on pure styling preferences.
2. UPHOLD genuine blockers:
   - Uphold claims of tests passing, services running, or files committed when tool events do not show them.
   - Uphold direct contradictions with tool outputs or user directives.
   - Uphold hidden failures or unhandled critical errors.
3. If all accusations are dismissed as pedantic or baseless:
   Return: {"status": "approved", "reason": "All substantive claims are verified. Pedantic accusations dismissed."}
4. If one or more genuine blockers are upheld:
   Return:
   {
     "status": "rejected",
     "reason": "Clear explanation of the real blocker",
     "missing_checks": ["Exact tool command needed, e.g. git status"],
     "unbacked_claims": ["Specific claim that must be verified or dropped"]
   }"""


def parse_json_safely(text: str) -> Any:
    """Extract and parse first JSON object or array from LLM response text."""
    if not text:
        return None
    text = text.strip()
    # Try direct parse
    try:
        return json.loads(text)
    except Exception:
        pass
    # Try regex match for object or array
    m = re.search(r"(\[.*\]|\{.*\})", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    return None


class DialecticAuditor:
    """Orchestrates the two-stage Challenger -> Arbiter dialectic."""

    def __init__(self, llm_complete_fn):
        self.llm_complete = llm_complete_fn

    def run_challenger(self, draft: str, tool_events_text: str, directives_text: str, history_text: str) -> list[dict] | None:
        """Phase 1: Red-team attacks the draft."""
        user_prompt = (
            f"### Proposed Draft Response:\n{draft}\n\n"
            f"### Directives & Constraints:\n{directives_text or 'None'}\n\n"
            f"### Tool Execution Events This Turn:\n{tool_events_text or 'No tools ran this turn.'}\n\n"
            f"### Recent Conversation Context:\n{history_text or 'None'}\n\n"
            "Produce your JSON list of accusations:"
        )
        messages = [
            {"role": "system", "content": CHALLENGER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        try:
            res = self.llm_complete(messages=messages)
            parsed = parse_json_safely(getattr(res, "text", str(res)))
            if isinstance(parsed, list):
                return [a for a in parsed if isinstance(a, dict)]
        except Exception as e:
            logger.warning("Advisor Challenger failed: %s", e)
        return None

    def run_arbiter(
        self,
        draft: str,
        tool_events_text: str,
        directives_text: str,
        accusations: list[dict] | None,
    ) -> dict[str, Any]:
        """Phase 2: Senior arbiter rules on accusations."""
        if accusations is None:
            return {
                "status": "error",
                "reason": "Challenger stage failed to produce a valid evaluation.",
            }

        if len(accusations) == 0:
            return {
                "status": "approved",
                "reason": "Challenger found zero issues. Draft is corroborated.",
            }

        user_prompt = (
            f"### Proposed Draft Response:\n{draft}\n\n"
            f"### Tool Execution Events This Turn:\n{tool_events_text or 'No tools ran this turn.'}\n\n"
            f"### Directives & Constraints:\n{directives_text or 'None'}\n\n"
            f"### Challenger Accusations:\n{json.dumps(accusations, indent=2)}\n\n"
            "Rule on these accusations. Output ONLY JSON with status 'approved' or 'rejected':"
        )
        messages = [
            {"role": "system", "content": ARBITER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        try:
            res = self.llm_complete(messages=messages)
            parsed = parse_json_safely(getattr(res, "text", str(res)))
            if isinstance(parsed, dict) and "status" in parsed:
                parsed["status"] = str(parsed["status"]).lower().strip()
                return parsed
        except Exception as e:
            logger.warning("Advisor Arbiter failed: %s", e)
            return {"status": "error", "reason": f"Arbiter stage failed: {e}"}

        # Fallback if parsing failed
        valid_accusations = [a for a in accusations if isinstance(a, dict)]
        return {
            "status": "rejected",
            "reason": f"Auditor flagged {len(valid_accusations)} potential issues: " + "; ".join(str(a.get("evidence_gap", "")) for a in valid_accusations[:2]),
            "unbacked_claims": [str(a.get("claim", "")) for a in valid_accusations[:3]],
        }
