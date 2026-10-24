"""Advisor data models."""

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

# A review note is prose advice. These markers mean the model returned leaked
# machine output instead — a tool call it tried to make, or raw structured
# data. Such a note is dropped (with a WARNING) rather than delivered or held:
# a malformed review must cost silence, not garbage in the agent's context.
_TOOL_CALL_MARKERS = (
    "<tool_call>",
    "</tool_call>",
    "<function",
    "<parameter",
    "<tool_use",
    "<invoke",
)
MAX_NOTE_CHARS = 1200


def is_malformed_note(note: str) -> bool:
    """True when a note is leaked machine output rather than prose advice."""
    if not note or not note.strip():
        return True
    stripped = note.strip()
    if len(stripped) > MAX_NOTE_CHARS:
        return True
    lowered = stripped.casefold()
    if any(marker in lowered for marker in _TOOL_CALL_MARKERS):
        return True
    # Bare JSON payload — a dict/list serialised straight into the review.
    if stripped[0] in "{[":
        try:
            json.loads(stripped)
            return True
        except (ValueError, TypeError):
            pass
    return False


class Severity(str, Enum):
    NIT = "nit"
    CONCERN = "concern"
    BLOCKER = "blocker"


SEVERITY_RANK = {Severity.NIT: 1, Severity.CONCERN: 2, Severity.BLOCKER: 3}


@dataclass
class Advice:
    """A single piece of advice with severity."""

    note: str
    severity: Severity = Severity.NIT

    def tag(self) -> str:
        return f"[{self.severity.value.upper()}]"


@dataclass
class TurnDelta:
    """Data about one agent turn."""

    session_id: str
    turn_id: str
    user_message: str
    assistant_response: str
    conversation_history: list  # full message list
    model: str
    context: Any = None


@dataclass
class AdvisorState:
    """Persistent state for the advisor plugin."""

    enabled: bool = False
    held_notes: list[dict] = field(default_factory=list)
    # Per-advisor model override — empty means inherit primary model
    model: str = ""
    provider: str = ""

    def dedupe_key(self, note: str) -> str:
        return re.sub(r"\s+", " ", note.strip()).casefold()

    def has_held(self) -> bool:
        return len(self.held_notes) > 0

    def format_reconfirm_preamble(self) -> str:
        """Build a reconfirmation preamble from held notes."""
        if not self.held_notes:
            return ""
        items = "\n".join(
            f"- [{h['severity'].upper()}] {h['note']}"
            for h in self.held_notes
        )
        return (
            "### Held advisories — reconfirm\n\n"
            "You raised these on an earlier step; they were held pending reconfirmation, "
            "because by now the agent may have already addressed them. Re-check each "
            "against the latest activity below.\n\n"
            "For every item that STILL applies, raise it again — same severity, or higher "
            "if it's gotten worse. Say nothing for the rest — silence drops them.\n\n"
            f"{items}\n\n---"
        )

    def parse_response(self, text: str) -> list[Advice]:
        """Parse a review and return only advice that should be delivered.

        Nits are delivered immediately. A concern or blocker is held on first
        sight and becomes deliverable only when a later review raises the same
        normalized note again. Held notes omitted by the latest review are
        considered resolved and removed.
        """
        previous_held = {
            self.dedupe_key(item.get("note", "")): item
            for item in self.held_notes
            if item.get("note")
        }
        if not text or not text.strip():
            self.held_notes = []
            return []

        text = text.strip()

        # Check for silence signal
        if "nothing to flag" in text.lower():
            self.held_notes = []
            return []

        parsed: list[Advice] = []

        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            for sev in Severity:
                tag = f"[{sev.value.upper()}]"
                if tag in line:
                    note = line.replace(tag, "").strip().strip(":").strip()
                    if note and is_malformed_note(note):
                        logger.warning(
                            "Advisor: dropping malformed %s note (%d chars): %.120s",
                            sev.value,
                            len(note),
                            note,
                        )
                    elif note:
                        parsed.append(Advice(note=note, severity=sev))
                    break

        if not parsed:
            # Unstructured response — a prose reply with substance is a concern,
            # but leaked tool-call or structured output is machine noise, not
            # advice. Drop it so a bad review costs silence, not context.
            if len(text) > 20:
                if is_malformed_note(text):
                    logger.warning(
                        "Advisor: dropping malformed unstructured review "
                        "(%d chars): %.120s",
                        len(text),
                        text,
                    )
                else:
                    parsed.append(Advice(note=text, severity=Severity.CONCERN))

        deliverable: list[Advice] = []
        next_held: dict[str, dict] = {}
        for advice in parsed:
            if advice.severity == Severity.NIT:
                deliverable.append(advice)
                continue

            key = self.dedupe_key(advice.note)
            if key in previous_held:
                deliverable.append(advice)

            current = next_held.get(key)
            if current is None:
                next_held[key] = {
                    "note": advice.note,
                    "severity": advice.severity.value,
                }
                continue

            old_rank = SEVERITY_RANK[Severity(current["severity"])]
            if SEVERITY_RANK[advice.severity] > old_rank:
                current["severity"] = advice.severity.value

        self.held_notes = list(next_held.values())
        return deliverable

    def serialize(self) -> dict:
        return {
            "enabled": self.enabled,
            "held_notes": self.held_notes,
            "model": self.model,
            "provider": self.provider,
        }

    @classmethod
    def deserialize(cls, data: dict) -> "AdvisorState":
        return cls(
            enabled=data.get("enabled", True),
            held_notes=data.get("held_notes", []),
            model=data.get("model", ""),
            provider=data.get("provider", ""),
        )
