"""Advisor runtime — turn tracking, state, review triggers."""

import contextlib
import hashlib
import json
import logging
import os
import queue
import re
import threading
from contextvars import copy_context
from pathlib import Path
from typing import Any

from .advisor_prompt import ADVISOR_SYSTEM_PROMPT
from .models import AdvisorState, Advice, Severity, TurnDelta
from .ledger import audit_draft_ledger
from .prober import run_active_probes_for_draft

logger = logging.getLogger(__name__)

# Env var to skip live reviews (keeps /advisor test path for manual testing)
ADVISOR_NO_REVIEW = "ADVISOR_NO_REVIEW"
WATCHDOG_FILENAME = "WATCHDOG.md"
REVIEW_TIMEOUT_SECONDS = 90
SHUTDOWN_GRACE_SECONDS = 1.0

_SECRET_PATTERNS = (
    re.compile(r"sk-[a-zA-Z0-9_-]{20,}"),
    re.compile(r"ghp_[a-zA-Z0-9]{36}"),
    re.compile(r"xoxb-[0-9]{10,13}-[0-9]{10,13}-[a-zA-Z0-9]{24}"),
    re.compile(r"AIza[0-9A-Za-z_-]{35}"),
    re.compile(r"Bearer\s+[A-Za-z0-9\-._~+/]{20,}"),
    re.compile(r"-----BEGIN (?:RSA |EC )?PRIVATE KEY-----[\s\S]*?-----END (?:RSA |EC )?PRIVATE KEY-----"),
    re.compile(
        r"(?i)([A-Za-z0-9_]*(?:api[_-]?key|secret|password|passwd|token))['\"]?\s*[:=]\s*['\"]?([^\s'\",}]{8,})['\"]?"
    ),
)


def _redact_secrets(text: str) -> str:
    """Scrub common tokens, keys, and credentials before LLM dispatch."""
    if not text:
        return ""
    for pat in _SECRET_PATTERNS:
        text = pat.sub("[REDACTED SECRET]", text)
    return text


def _redact_secrets_obj(val: Any) -> Any:
    """Recursively scrub secrets in str, dict, list, and tuple values."""
    if isinstance(val, str):
        return _redact_secrets(val)
    if isinstance(val, dict):
        return {k: _redact_secrets_obj(v) for k, v in val.items()}
    if isinstance(val, list):
        return [_redact_secrets_obj(item) for item in val]
    if isinstance(val, tuple):
        return tuple(_redact_secrets_obj(item) for item in val)
    return val


def _find_guidelines(cwd: str) -> str:
    """Find WATCHDOG.md, AGENTS.md, or CLAUDE.md by walking up from cwd to git root."""
    try:
        base = Path(cwd).resolve()
        for d in [base, *base.parents]:
            for name in (WATCHDOG_FILENAME, "AGENTS.md", "CLAUDE.md"):
                p = d / name
                if p.is_file():
                    try:
                        return p.read_text(encoding="utf-8", errors="replace").strip()[:4000]
                    except OSError:
                        pass
            if (d / ".git").exists():
                break
    except Exception:
        pass
    return ""


def _is_substantive_turn(user_msg: str, asst_resp: str, history: list) -> bool:
    """True when this turn involved tool calls, code, or substantive technical explanation."""
    last_user = max((i for i, m in enumerate(history) if isinstance(m, dict) and m.get("role") == "user"), default=-1)
    if last_user >= 0:
        recent = history[last_user + 1:]
    else:
        # If history contains no user message, do not assume prior messages belong to this turn
        recent = []
    for msg in recent:
        if isinstance(msg, dict):
            if msg.get("role") == "tool" or msg.get("tool_calls"):
                return True
    text = (asst_resp or "").strip()
    if "```" in text or len(text) >= 200:
        return True
    return False


ADVISOR_INJECTION_PREFIX = "◆ Advisor review"


def _advisor_depth(messages: list) -> int:
    """Count consecutive advisor injections since the last genuine human user message."""
    depth = 0
    for m in reversed(messages or []):
        if not isinstance(m, dict):
            continue
        if m.get("role") != "user":
            continue
        content = (m.get("content") or "").lstrip()
        if content.startswith(ADVISOR_INJECTION_PREFIX):
            depth += 1
        else:
            break
    return depth


MAX_DIRECTIVES_CHARS = 8000
ANCHOR_CAP = 2000
SNIPPET_CAP = 600


def _get_git_diff(cwd: str, paths: list[str], max_chars: int = 12000) -> str | None:
    """Run git diff on changed paths safely, including new, untracked, and deleted files."""
    try:
        import os
        import subprocess
        from pathlib import Path

        diff_chunks = []
        total_len = 0

        for p in paths:
            if not p:
                continue
            full_path = (Path(p) if os.path.isabs(p) else Path(cwd) / p).resolve()
            # Skip internal .git directory files, but allow .github/, .gitignore, etc.
            if ".git" in full_path.parts:
                continue

            parent_dir = full_path.parent if full_path.parent.exists() else Path(cwd)

            # Resolve git repo root for this specific file
            res_root = subprocess.run(
                ["git", "-C", str(parent_dir), "rev-parse", "--show-toplevel"],
                capture_output=True,
                text=True,
                timeout=3,
            )
            raw_repo = res_root.stdout.strip() if res_root.returncode == 0 else None
            repo_dir = None
            if raw_repo:
                repo_path = Path(raw_repo).resolve()
                plugin_root = Path(__file__).resolve().parent
                # Ensure the file is actually inside the detected repo, and avoid auditing
                # the advisor plugin's own repo for unrelated tasks
                if full_path.resolve().is_relative_to(repo_path):
                    if repo_path != plugin_root or full_path.resolve().is_relative_to(plugin_root):
                        repo_dir = raw_repo

            chunk = ""
            if not full_path.exists():
                # Handle deleted file
                if repo_dir:
                    rel_p = os.path.relpath(full_path, repo_dir)
                    cmd = ["git", "-C", repo_dir, "diff", "HEAD", "--", rel_p]
                    res = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                    chunk = res.stdout.strip() if res.returncode == 0 else ""
                if not chunk:
                    chunk = f"deleted file: {p}"
            elif repo_dir:
                # File exists within a git repo
                rel_p = os.path.relpath(full_path, repo_dir)
                res_track = subprocess.run(
                    ["git", "-C", repo_dir, "ls-files", "--error-unmatch", "--", rel_p],
                    capture_output=True,
                    text=True,
                    timeout=3,
                )
                if res_track.returncode == 0:
                    # Tracked: diff against HEAD (staged + unstaged edits)
                    cmd = ["git", "-C", repo_dir, "diff", "HEAD", "--", rel_p]
                    res = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                    chunk = res.stdout.strip() if res.returncode == 0 else ""
                    if not chunk:
                        cmd_c = ["git", "-C", repo_dir, "diff", "--", rel_p]
                        res_c = subprocess.run(cmd_c, capture_output=True, text=True, timeout=5)
                        chunk = res_c.stdout.strip() if res_c.returncode == 0 else ""
                else:
                    # Untracked in repo: diff against /dev/null
                    cmd = ["git", "-C", repo_dir, "diff", "--no-index", "--", "/dev/null", rel_p]
                    res = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                    chunk = res.stdout.strip()
            else:
                # Outside git repo entirely: diff against /dev/null
                cmd = ["git", "diff", "--no-index", "--", "/dev/null", str(full_path)]
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                chunk = res.stdout.strip()

            if chunk:
                diff_chunks.append(chunk)
                total_len += len(chunk)
                if total_len > max_chars:
                    diff_chunks.append("\n... [further changed paths omitted due to diff size limit] ...\n")
                    break

        if not diff_chunks:
            if paths:
                logger.warning("Advisor: changed_paths non-empty (%s) but diff was empty in %s", paths, cwd)
            return None

        full_diff = "\n\n".join(diff_chunks)
        if len(full_diff) > max_chars:
            full_diff = full_diff[:4000] + f"\n\n... [diff truncated - {len(full_diff)} chars total] ...\n\n" + full_diff[-6000:]
        return full_diff
    except Exception as exc:
        logger.debug("Advisor git diff failed for %s in %s: %s", paths, cwd, exc)
        return None


def _clip_directive(text: str, cap: int) -> str:
    """Clip a directive preserving both leading context and trailing instructions."""
    text = text.strip()
    if len(text) <= cap:
        return text
    head = cap * 2 // 5
    tail = cap - head - 20
    return f"{text[:head]} …[{len(text) - cap} chars]… {text[-tail:]}"


def _collect_directives(user_msgs: list[str]) -> list[str]:
    """Collect human user directives with an overall character budget.

    Preserves the initial task framing (first message, up to ANCHOR_CAP)
    and walks newest-to-oldest with balanced head-and-tail clipping so
    trailing instructions in long pastes survive.
    """
    if not user_msgs:
        return []

    cleaned = []
    for m in user_msgs:
        m_str = (m or "").strip()
        if (
            not m_str
            or m_str in {"...", "…"}
            or m_str.startswith("◆ Advisor review")
            or m_str.startswith("[CONTEXT COMPACTION")
            or m_str.startswith("[System note:")
            or m_str.startswith("[Verification")
        ):
            continue
        if m_str.startswith("[OUT-OF-BAND USER MESSAGE"):
            # Unwrap out-of-band message to keep user's actual steering text
            import re
            inner = re.sub(r"^\[OUT-OF-BAND USER MESSAGE[^\]]*\]\s*", "", m_str)
            inner = re.sub(r"\s*\[/OUT-OF-BAND USER MESSAGE\]$", "", inner).strip()
            if inner:
                m_str = inner
            else:
                continue
        cleaned.append(m_str)

    if not cleaned:
        return []

    first_snippet = _clip_directive(cleaned[0], ANCHOR_CAP)
    rest = cleaned[1:]
    kept_rev = []
    used = len(first_snippet)

    for m in reversed(rest):
        snippet = _clip_directive(m, SNIPPET_CAP)
        if used + len(snippet) > MAX_DIRECTIVES_CHARS:
            break
        kept_rev.append(snippet)
        used += len(snippet)

    return [first_snippet] + list(reversed(kept_rev))


class AdvisorRuntime:
    """Tracks turns and triggers reviews via the advisor model.

    Wired into Hermes plugin hooks:
      - ``post_llm_call`` — fires once per turn at end, carries full history.
        This is our ``turn_end`` equivalent.
    """

    def __init__(self, ctx):
        self.ctx = ctx
        from hermes_constants import get_hermes_home

        self.state_file = get_hermes_home() / "advisor" / "state.json"
        self.sessions_dir = self.state_file.parent / "sessions"
        self.legacy_state_file = Path(__file__).parent / "state.json"
        self._state_lock = threading.RLock()
        self.state = self._load_state()
        self._session_states: dict[str, AdvisorState] = {}
        self._turn_tool_events: dict[str, list[dict]] = {}
        self._turn_verified: dict[str, bool] = {}
        self._session_history: dict[str, list[dict]] = {}

        # Current turn tracking — populated from post_llm_call kwargs
        self.last_turn_key: tuple[str, str] | None = None
        self._review_queue: queue.Queue[TurnDelta | None] = queue.Queue(maxsize=1)
        self._queue_lock = threading.Lock()
        self._worker: threading.Thread | None = None
        self._idle = threading.Event()
        self._idle.set()

    # ── state persistence ────────────────────────────────────────────────

    def _load_state(self) -> AdvisorState:
        for path in (self.state_file, self.legacy_state_file):
            try:
                state = AdvisorState.deserialize(json.loads(path.read_text()))
            except FileNotFoundError:
                continue
            except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
                logger.warning("Advisor: could not load state from %s: %s", path, exc)
                continue

            if path == self.legacy_state_file:
                self.state = state
                self._save_state()
            return state
        return AdvisorState(enabled=True)

    def _save_state(self):
        from utils import atomic_json_write

        with self._state_lock:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            atomic_json_write(
                self.state_file,
                {
                    **self.state.serialize(),
                    "held_notes": [],
                },
                indent=2,
                mode=0o600,
            )

    def _session_path(self, session_id: str) -> Path:
        digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:24]
        return self.sessions_dir / f"{digest}.json"

    def _session_state(self, session_id: str) -> AdvisorState:
        key = session_id or "default"
        with self._state_lock:
            existing = self._session_states.get(key)
            if existing is not None:
                return existing
            try:
                data = json.loads(self._session_path(key).read_text())
                state = AdvisorState.deserialize(data)
            except FileNotFoundError:
                state = AdvisorState(enabled=True)
            except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
                logger.warning("Advisor: could not load session state: %s", exc)
                state = AdvisorState(enabled=True)
            self._session_states[key] = state
            return state

    def _save_session_state(self, session_id: str, state: AdvisorState) -> None:
        from utils import atomic_json_write

        key = session_id or "default"
        with self._state_lock:
            path = self._session_path(key)
            path.parent.mkdir(parents=True, exist_ok=True)
            atomic_json_write(
                path,
                {
                    "session_id": key,
                    "held_notes": state.held_notes,
                },
                indent=2,
                mode=0o600,
            )

    def _held_count(self) -> int:
        count = 0
        for path in self.sessions_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text())
                count += len(data.get("held_notes") or [])
            except (json.JSONDecodeError, OSError, TypeError) as exc:
                logger.warning("Advisor: could not count held notes in %s: %s", path, exc)
        return count

    def _clear_held_notes(self) -> None:
        from utils import atomic_json_write

        with self._state_lock:
            # Clear in-memory session states and track which disk files we cover
            cleared_digests: set[str] = set()
            for key, state in self._session_states.items():
                state.held_notes = []
                self._save_session_state(key, state)
                cleared_digests.add(self._session_path(key).name)

            # Catch orphaned session files not yet loaded into memory
            for path in self.sessions_dir.glob("*.json"):
                if path.name in cleared_digests:
                    continue  # already handled above
                try:
                    atomic_json_write(
                        path,
                        {"session_id": "(migrated)", "held_notes": []},
                        indent=2,
                        mode=0o600,
                    )
                except (OSError, TypeError) as exc:
                    logger.warning("Advisor: could not clear %s: %s", path, exc)

    # ── hook: end of each agent turn ─────────────────────────────────────

    def on_post_llm_call(
        self, *,
        session_id: str = "",
        turn_id: str = "",
        user_message: str = "",
        assistant_response: str = "",
        conversation_history: list | None = None,
        model: str = "",
        **kwargs,
    ):
        """Deprecated post-turn injection: disabled in favor of in-turn verify_turn and pre_verify."""
        return

    def _enqueue_review(self, turn: TurnDelta) -> None:
        """Queue the newest completed turn without blocking hook dispatch."""
        self._ensure_worker()
        with self._queue_lock:
            self._idle.clear()
            try:
                self._review_queue.put_nowait(turn)
                return
            except queue.Full:
                pass

            try:
                dropped = self._review_queue.get_nowait()
                self._review_queue.task_done()
                if dropped is not None:
                    logger.info(
                        "Advisor: dropped stale queued turn %s in favor of %s",
                        dropped.turn_id,
                        turn.turn_id,
                    )
            except queue.Empty:
                pass
            self._review_queue.put_nowait(turn)

    def _ensure_worker(self) -> None:
        with self._state_lock:
            if self._worker is not None and self._worker.is_alive():
                return
            self._worker = threading.Thread(
                target=self._worker_loop,
                name="hermes-advisor",
                daemon=True,
            )
            self._worker.start()

    def _worker_loop(self) -> None:
        while True:
            turn = self._review_queue.get()
            try:
                if turn is None:
                    return
                if getattr(turn, "context", None) is not None:
                    turn.context.run(self._review_turn, turn)
                else:
                    self._review_turn(turn)
            finally:
                with self._queue_lock:
                    self._review_queue.task_done()
                    if self._review_queue.unfinished_tasks == 0:
                        self._idle.set()

    def _review_turn(self, turn: TurnDelta) -> None:
        try:
            advice_list = self._run_review(
                user_message=turn.user_message,
                assistant_response=turn.assistant_response,
                conversation_history=turn.conversation_history,
                model=turn.model,
                turn_id=turn.turn_id,
                session_id=turn.session_id,
            )
        except Exception as exc:
            logger.warning("Advisor review failed for turn %s: %s", turn.turn_id, exc)
            return

        if not advice_list:
            logger.debug("Advisor: nothing to deliver for turn %s", turn.turn_id)
            return
        self._deliver_advice(advice_list, session_id=turn.session_id)

    def on_session_finalize(self, session_id: str = "", **_kwargs) -> None:
        """Clean up in-memory session caches and give in-flight reviews a small shutdown grace period."""
        sess = session_id or "default"
        with self._state_lock:
            self._session_history.pop(sess, None)
            self._turn_tool_events.pop(sess, None)
            self._turn_verified.pop(sess, None)
            self._session_states.pop(sess, None)

        if not self._idle.wait(SHUTDOWN_GRACE_SECONDS):
            logger.info(
                "Advisor: review still running after %.1fs shutdown grace; "
                "daemon worker will not delay exit",
                SHUTDOWN_GRACE_SECONDS,
            )

    def wait_for_idle(self, timeout: float = 5.0) -> bool:
        """Wait for queued work to finish. Intended for tests and diagnostics."""
        return self._idle.wait(timeout)

    # ── run the review ────────────────────────────────────────────────────

    def _run_review(
        self,
        *,
        user_message: str,
        assistant_response: str,
        conversation_history: list,
        model: str,
        turn_id: str,
        session_id: str,
    ) -> list[Advice]:
        """Build the prompt, call the advisor model, parse the result."""

        with self._state_lock:
            if not self.state.enabled:
                return []
            review_state = self._session_state(session_id)
            messages = self._build_review_prompt(
                user_message=user_message,
                assistant_response=assistant_response,
                conversation_history=conversation_history,
                cwd=os.getcwd(),
                review_state=review_state,
            )
            advisor_model = self.state.model
            advisor_provider = self.state.provider

        # Call the advisor model — use configured override or inherit primary
        kwargs = {"messages": messages, "timeout": REVIEW_TIMEOUT_SECONDS}
        if advisor_model:
            kwargs["model"] = advisor_model
        if advisor_provider:
            kwargs["provider"] = advisor_provider

        result = self.ctx.llm.complete(**kwargs)

        logger.debug(
            "Advisor: review complete, provider=%s model=%s tokens=%d",
            result.provider, result.model, result.usage.total_tokens if result.usage else 0,
        )

        with self._state_lock:
            if not self.state.enabled:
                return []
            advice = review_state.parse_response(result.text)
            self._save_session_state(session_id, review_state)
            return advice

    def _session_cwd(self, session_id: str = "") -> str:
        """Resolve session working directory from state.db or fallback to getcwd."""
        if not session_id:
            return os.getcwd()
        try:
            import sqlite3
            from hermes_constants import get_hermes_home
            db_path = get_hermes_home() / "state.db"
            if db_path.exists():
                with contextlib.closing(sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)) as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT cwd FROM sessions WHERE id = ? LIMIT 1", (session_id,))
                    row = cursor.fetchone()
                    if row and row[0]:
                        return row[0]
        except Exception:
            pass
        return os.getcwd()

    def consult_advisor(self, *, task: str, proposed_approach: str = "", session_id: str = "") -> str:
        """On-demand senior architect consultation during a turn."""
        cwd = self._session_cwd(session_id) or os.getcwd()
        guidelines = _find_guidelines(cwd)

        # Prefix caching optimization: static instructions & repo guidelines come FIRST
        prompt_parts = [
            "You are a Staff Software Architect paired with a coding agent. "
            "The agent is consulting you for architectural and design guidance before taking action.\n"
        ]
        if guidelines:
            prompt_parts.append(f"### Project Guidelines & Architecture (from repo):\n{guidelines}\n")

        prompt_parts.append(
            "### Instructions:\n"
            "Provide high-signal, concise (300-600 words) architectural direction:\n"
            "1. Recommended Pattern & Approach: The cleanest, most robust, idiomatic way to solve this.\n"
            "2. Potential Pitfalls & Edge Cases: Subtleties to watch out for.\n"
            "3. Existing Libraries / Primitives: Any standard or battle-tested libraries to prefer over custom code.\n"
            "Be direct, practical, and specific.\n"
        )
        prompt_parts.append(f"### Task / Goal:\n{task}\n")
        if proposed_approach:
            prompt_parts.append(f"### Proposed Approach:\n{proposed_approach}\n")

        prompt = "\n".join(prompt_parts)

        messages = [
            {"role": "system", "content": "You are a senior Staff Software Architect advising a coding agent."},
            {"role": "user", "content": prompt},
        ]

        kwargs = {"messages": messages, "timeout": REVIEW_TIMEOUT_SECONDS}
        if self.state.model:
            kwargs["model"] = self.state.model
        if self.state.provider:
            kwargs["provider"] = self.state.provider

        result = self.ctx.llm.complete(**kwargs)
        return result.text.strip()

    def record_tool_call(
        self, *, tool_name: str = "", args: dict | None = None, result: Any = None, session_id: str = "", **kwargs
    ) -> None:
        """Record tool execution for the active turn in a thread-safe buffer."""
        if tool_name in ("verify_turn", "consult_advisor", "tool_search", "tool_describe"):
            return
        sess = session_id or "default"
        with self._state_lock:
            if sess not in self._turn_tool_events:
                self._turn_tool_events[sess] = []
            # Any new action invalidates a prior approval for this turn.
            self._turn_verified[sess] = False
            res_str = str(result) if result is not None else ""
            res_str = _redact_secrets(res_str)
            if len(res_str) > 2000:
                head_len = 650
                tail_len = 1350
                res_str = (
                    res_str[:head_len]
                    + f"\n... [truncated {len(res_str) - 2000} chars] ...\n"
                    + res_str[-tail_len:]
                )
            clean_args = _redact_secrets_obj(args) if args else {}
            self._turn_tool_events[sess].append({
                "tool": tool_name,
                "args": clean_args,
                "result": res_str,
            })

    def _session_messages(self, session_id: str = "", limit: int = 20) -> list[dict]:
        """Fetch recent session messages from state.db for rich context."""
        if not session_id:
            return []
        try:
            import sqlite3
            from hermes_constants import get_hermes_home
            db_path = get_hermes_home() / "state.db"
            if db_path.exists():
                with contextlib.closing(sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)) as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT role, content FROM messages WHERE session_id = ? AND active = 1 ORDER BY timestamp DESC LIMIT ?",
                        (session_id, limit),
                    )
                    rows = cursor.fetchall()
                    return [{"role": r[0], "content": r[1]} for r in reversed(rows)]
        except Exception:
            pass
        return []

    def on_pre_llm_call(self, *, session_id: str = "", **kwargs) -> dict:
        """Inject verification directive into active turn context for all sessions."""
        sess = session_id or "default"
        history = kwargs.get("conversation_history")
        with self._state_lock:
            self._turn_tool_events[sess] = []
            self._turn_verified[sess] = False
            if history:
                self._session_history[sess] = list(history[-20:])

        if not self.state.enabled or os.environ.get(ADVISOR_NO_REVIEW):
            return {}

        return {
            "context": (
                "[Verification Gate]: On any turn where you executed tools or performed actions, you must call "
                'verify_turn(draft="<your natural response for user>", evidence="<optional technical verification notes for auditor>") '
                "before delivering your final answer. The draft should be natural, direct, and human (no robotic compliance text or routine tool plumbing; "
                "surface material failures or decisions plainly); raw technical proofs and tool details belong in evidence or tool events."
            )
        }

    def verify_turn(
        self,
        *,
        draft: str = "",
        claims: list[str] | str | None = None,
        evidence: str = "",
        session_id: str = "",
    ) -> dict:
        """Audit draft response and tool actions against full session history and constraints."""
        if not self.state.enabled or os.environ.get(ADVISOR_NO_REVIEW):
            return {"status": "approved", "reason": "Advisor disabled"}
        sess = session_id or "default"
        with self._state_lock:
            events = list(self._turn_tool_events.get(sess, []))
            raw_history = list(self._session_history.get(sess, []))
            history_msgs = [m for m in raw_history if isinstance(m, dict)]

        # Directives across the entire session from past human messages
        directives = self._session_user_directives(session_id)
        # Recent conversation history fallback if memory cache not yet populated or has no valid dicts
        if not history_msgs:
            history_msgs = self._session_messages(session_id, limit=10)

        # Find the latest user query from history
        latest_user_query = ""
        for m in reversed(history_msgs):
            if isinstance(m, dict) and m.get("role") == "user":
                latest_user_query = m.get("content", "")
                break

        # Check project guidelines
        cwd = self._session_cwd(session_id) or os.getcwd()
        guidelines = _find_guidelines(cwd)

        # Active autonomous probing (read-only verification of verifiable claims)
        claims_text = " ".join(str(c) for c in claims) if isinstance(claims, list) else str(claims or "")
        scan_draft = f"{draft} {claims_text}".strip()
        active_probes = run_active_probes_for_draft(scan_draft, cwd)
        probe_notes = []
        for p in active_probes:
            probe_notes.append(p.format_evidence())
            if p.success:
                events.append({
                    "tool": "active_probe",
                    "args": p.command,
                    "result": p.output,
                })

        combined_evidence = f"{evidence}\n" + "\n".join(probe_notes) if probe_notes else evidence

        # Deterministic ledger pre-flight verification
        ledger_verdict = audit_draft_ledger(draft, events, combined_evidence, claims=claims)
        if not ledger_verdict.passed:
            verdict = ledger_verdict.to_dict()
            if probe_notes:
                verdict["reason"] = f"{verdict.get('reason', '')} Active probe notes: " + " ".join(probe_notes)
            return verdict

        system_prompt = (
            "You are an objective auditor and senior coworker verifying an AI agent's actions and draft response.\n"
            "You have access to the full session context: user directives, recent conversation history, and the raw tool execution events of this turn.\n\n"
            "Evaluate the agent's work against these strict verification and communication rules:\n"
            "1. Faithful & Truthful: Did the agent make claims contradicted by the tool outputs (e.g. claim success when tool errored, or claim 0 failures when errors occurred)? If so, REJECT.\n"
            "2. Corroborated Evidence: Do NOT treat agent-supplied `evidence` as verified ground truth on its own. Technical claims must be corroborated by recorded tool execution events or session history; treat uncorroborated evidence as an assertion to check.\n"
            "3. User Directives & Invariants: Did the agent violate any standing user directives or explicit constraints from this session? If so, REJECT.\n"
            "4. Natural Communication Style & Material Caveats (SOUL.md):\n"
            "   - The `draft` is user-facing communication for Alejandro. It must be natural, direct, and concise (plain claims, no filler, no robotic templates, no narrating routine tool calls, tab IDs, or exit codes).\n"
            "   - Surface real blockers, material failures, or required decisions plainly in the draft. Do not hide them in evidence.\n"
            "   - Do NOT reject a draft simply because it cleanly summarizes outcomes without reciting routine technical plumbing, as long as the summary is truthful and corroborated by tool events.\n"
            "   - Reject drafts that read like bureaucratic compliance affidavits, legal disclaimers, or defensive apologies. The agent is a peer engineer speaking directly to Alejandro, not writing a compliance filing.\n"
            "5. Action-First Guidance:\n"
            "   - If a claim lacks evidence or was not checked, instruct the agent to RUN THE TOOL to verify it, or DROP the claim entirely.\n"
            "   - Never demand that the agent write a defensive paragraph admitting it didn't verify something. Action comes first.\n"
            "6. If the draft is sound, truthful, answers the user's ask, and is written naturally for a human, APPROVE.\n\n"
            'Respond in JSON format: {"status": "approved" | "rejected", "reason": "...", "missing_checks": [...], "unbacked_claims": [...]}'
        )

        user_prompt_parts = []
        if guidelines:
            user_prompt_parts.append(f"### Project Guidelines:\n{guidelines}")
        if directives:
            user_prompt_parts.append("### User Directives & Constraints (Session Context):\n" + "\n".join(f"- {d}" for d in directives))
        if latest_user_query:
            user_prompt_parts.append(f"### Current User Request:\n{latest_user_query}")

        # Recent dialogue context leading into this turn
        if history_msgs:
            recent_turns = history_msgs[-8:]
            formatted_history = []
            for m in recent_turns:
                if not isinstance(m, dict):
                    continue
                role = m.get("role", "unknown")
                if role == "system":
                    continue
                content = str(m.get("content") or "").strip()
                if len(content) > 1000:
                    content = content[:500] + " ... [truncated]"
                content = _redact_secrets(content)
                tc = m.get("tool_calls")
                tools_info = ""
                if tc and isinstance(tc, list):
                    names = [c.get("function", {}).get("name") for c in tc if isinstance(c, dict) and c.get("function")]
                    names = [n for n in names if n]
                    if names:
                        tools_info = f" (called tools: {', '.join(names)})"
                if content or tools_info:
                    formatted_history.append(f"[{role}]{tools_info}: {content}")
            if formatted_history:
                user_prompt_parts.append("### Recent Conversation Dialogue:\n" + "\n\n".join(formatted_history))

        events_formatted = []
        for i, ev in enumerate(events, 1):
            events_formatted.append(f"{i}. Tool: {ev['tool']}\n   Args: {ev['args']}\n   Result: {ev['result']}")
        user_prompt_parts.append("### Tool Execution Events (This Turn):\n" + ("\n\n".join(events_formatted) if events_formatted else "None (no tools called)."))

        if draft:
            user_prompt_parts.append(f"### Proposed User-Facing Response (Draft):\n{draft}")
        if combined_evidence:
            user_prompt_parts.append(f"### Technical Verification Evidence (Internal Notes for Auditor):\n{_redact_secrets(str(combined_evidence))}")
        if claims:
            if isinstance(claims, str):
                claims = [claims]
            claims_clean = [str(c).strip() for c in claims if str(c).strip()]
            if claims_clean:
                user_prompt_parts.append("### Specific Asserted Claims:\n" + "\n".join(f"- {c}" for c in claims_clean))

        user_prompt = "\n\n".join(user_prompt_parts)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            kwargs = {"messages": messages, "timeout": REVIEW_TIMEOUT_SECONDS}
            if self.state.model:
                kwargs["model"] = self.state.model
            if self.state.provider:
                kwargs["provider"] = self.state.provider
            try:
                result = self.ctx.llm.complete(**kwargs)
            except Exception as e:
                if "cannot override the provider" in str(e).lower() and "provider" in kwargs:
                    logger.warning("Advisor: provider override rejected by host, falling back to primary provider")
                    kwargs.pop("provider", None)
                    result = self.ctx.llm.complete(**kwargs)
                else:
                    raise
            raw = (result.text or "").strip()
            import json, re
            parsed = None
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                try:
                    parsed = json.loads(m.group(0))
                except Exception:
                    pass
            if not isinstance(parsed, dict):
                status = "approved" if re.match(r"^\W*approved\b", raw, re.IGNORECASE) else "rejected"
                parsed = {"status": status, "reason": raw}

            parsed["status"] = str(parsed.get("status", "")).strip().lower() or "rejected"
            if parsed["status"] == "approved":
                with self._state_lock:
                    self._turn_verified[sess] = True
            return parsed
        except Exception as exc:
            logger.warning("verify_turn audit failed: %s", exc)
            return {"status": "error", "reason": f"Advisor evaluation failed: {exc}"}

    def on_transform_llm_output(
        self,
        *,
        response_text: str = "",
        session_id: str = "",
        model: str = "",
        platform: str = "",
        turn_id: str = "",
        **kwargs,
    ) -> str | None:
        """Enforce verification: if tools ran this turn but verify_turn was skipped, audit before delivery."""
        if not self.state.enabled or os.environ.get(ADVISOR_NO_REVIEW):
            return None
        sess = session_id or "default"
        with self._state_lock:
            tools_ran = bool(self._turn_tool_events.get(sess))
            already_verified = self._turn_verified.get(sess, False)

        if not tools_ran or already_verified:
            return None

        # Tools were executed, but the agent skipped calling verify_turn!
        logger.info("Forced verification running via transform_llm_output for session %s", sess)
        audit = self.verify_turn(draft=response_text, session_id=session_id)
        if audit.get("status") == "approved":
            return None  # Verified and clean, deliver response unchanged
        if audit.get("status") == "error":
            logger.warning("transform_llm_output audit encountered error: %s", audit.get("reason"))
            return None  # Fail open on provider error rather than injecting false contradiction alert

        # Auditor flagged discrepancies, omissions, or unverified claims!
        reason = audit.get("reason", "Discrepancy detected between claims and tool outputs.")
        return (
            f"{response_text}\n\n"
            f"[Auditor note: {reason}]"
        )

    def _session_user_directives(self, session_id: str = "") -> list[str]:
        """Pull user directives from state.db for the given session."""
        if not session_id:
            return []
        try:
            import sqlite3
            from hermes_constants import get_hermes_home
            db_path = get_hermes_home() / "state.db"
            if db_path.exists():
                with contextlib.closing(sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)) as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT content FROM messages WHERE session_id = ? AND role = 'user' AND active = 1 ORDER BY timestamp ASC",
                        (session_id,)
                    )
                    msgs = [r[0] for r in cursor.fetchall() if r[0]]
                    return _collect_directives(msgs)
        except Exception:
            pass
        return []

    def on_pre_verify(
        self,
        *,
        session_id: str = "",
        platform: str = "",
        model: str = "",
        coding: bool = False,
        attempt: int = 0,
        final_response: str = "",
        changed_paths: list[str] | None = None,
        **kwargs,
    ) -> dict[str, str] | None:
        """In-loop stop gate: audits git diff before the agent finishes/delivers."""
        with self._state_lock:
            if not self.state.enabled:
                return None
            if self._turn_verified.get(session_id or "default", False):
                logger.debug("Advisor pre_verify: session %s already approved by verify_turn", session_id)
                return None
        if os.environ.get(ADVISOR_NO_REVIEW):
            return None

        # Cap self-correction nudges to 2 attempts. When capped, turn finishes and
        # any remaining issues are handled by the transform_llm_output safety net.
        if attempt >= 2:
            logger.debug(
                "Advisor pre_verify: reached max nudge attempt (%d >= 2), allowing turn to finish",
                attempt,
            )
            return None

        if not changed_paths:
            return None

        cwd = self._session_cwd(session_id) or os.getcwd()
        diff_text = _get_git_diff(cwd, changed_paths)
        if not diff_text:
            logger.debug("Advisor pre_verify: empty diff for %s, skipping", changed_paths)
            return None
        diff_text = _redact_secrets(diff_text)

        guidelines = _find_guidelines(cwd)
        directives = self._session_user_directives(session_id)

        prompt_parts = [
            "You are a senior Staff Software Engineer and Architect performing an in-loop verification gate.",
            "The coding agent has modified files and is preparing to report completion.",
            "Review the git diff below against the user's task and project guidelines before the agent is allowed to finish.\n"
        ]
        if guidelines:
            prompt_parts.append(f"### Project Guidelines & Architecture:\n{guidelines}\n")
        if directives:
            prompt_parts.append("### User Directives & Constraints (Session Context):\n" + "\n".join(f"- User: {d}" for d in directives) + "\n")

        prompt_parts.append(f"### Modified Files:\n" + "\n".join(f"- {p}" for p in changed_paths) + "\n")
        prompt_parts.append(f"### Git Diff:\n```diff\n{diff_text}\n```\n")
        if final_response:
            clipped_resp = final_response if len(final_response) <= 2000 else final_response[:800] + "\n... [truncated] ...\n" + final_response[-1200:]
            prompt_parts.append(f"### Agent's Proposed Completion Report:\n{clipped_resp}\n")

        prompt_parts.append(
            "### Audit Instructions:\n"
            "- Check: Does this diff introduce bugs, syntax errors, security flaws, broken contracts, or unverified claims?\n"
            "- Check: Did the agent claim something works without running real tests or checks?\n"
            "- Check: Does the diff violate any explicit user directive or project guideline?\n"
            "- If the changes are sound, tested, minimal, and ready to ship, reply EXACTLY: nothing to flag\n"
            "- If there is a real bug, regression, or missing check, reply with [BLOCKER] or [CONCERN] and the exact fix required.\n"
        )

        prompt = "\n".join(prompt_parts)
        messages = [
            {"role": "system", "content": "You are a Staff Software Engineer acting as a pre-completion verification gate."},
            {"role": "user", "content": prompt},
        ]

        kwargs_llm = {"messages": messages, "timeout": REVIEW_TIMEOUT_SECONDS}
        if self.state.model:
            kwargs_llm["model"] = self.state.model
        if self.state.provider:
            kwargs_llm["provider"] = self.state.provider

        try:
            result = self.ctx.llm.complete(**kwargs_llm)
            review_text = (result.text or "").strip()
        except Exception as exc:
            logger.warning("Advisor pre_verify LLM call failed: %s", exc)
            return None

        if not review_text:
            logger.debug("Advisor pre_verify: empty review response, inconclusive")
            return None

        if review_text.lower().startswith("nothing to flag"):
            logger.info("Advisor pre_verify: diff approved ('nothing to flag')")
            with self._state_lock:
                self._turn_verified[session_id or "default"] = True
            return None

        if "[BLOCKER]" in review_text or "[CONCERN]" in review_text:
            logger.info("Advisor pre_verify: caught issue, nudging agent (attempt %d/2)", attempt + 1)
            return {
                "action": "continue",
                "message": (
                    f"Advisor caught issue before finish (attempt {attempt + 1}/2):\n"
                    f"{review_text}\n"
                    f"Fix this issue and verify before reporting completion."
                ),
            }

        logger.debug("Advisor pre_verify: review was inconclusive (%s)", review_text[:100])
        return None

    def _build_review_prompt(
        self,
        *,
        user_message: str,
        assistant_response: str,
        conversation_history: list,
        cwd: str,
        review_state: AdvisorState | None = None,
    ) -> list[dict]:
        """Build the message list for the advisor model."""

        # Base system prompt
        system_prompt = ADVISOR_SYSTEM_PROMPT

        # Append project guidelines if present (WATCHDOG.md, AGENTS.md, or CLAUDE.md)
        guidelines = _find_guidelines(cwd)
        if guidelines:
            system_prompt += (
                f"\n\nEspecially pay attention to project guidelines:\n"
                f"<guidelines>\n{guidelines}\n</guidelines>"
            )

        # Build the user content: reconfirm preamble + turn transcript
        user_content_parts = []

        # Reconfirm preamble for held notes
        preamble = (review_state or self.state).format_reconfirm_preamble()
        if preamble:
            user_content_parts.append(preamble)

        # Format the conversation history as a readable turn transcript
        transcript = self._format_history(
            user_message=user_message,
            response=assistant_response,
            history=conversation_history,
        )
        if transcript:
            user_content_parts.append(transcript)

        if not user_content_parts:
            return [{"role": "system", "content": system_prompt}]

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "\n\n".join(user_content_parts)},
        ]

    @staticmethod
    def _format_history(
        *, user_message: str, response: str, history: list
    ) -> str:
        """Format conversation history into a markdown transcript for review.

        Shows the user prompt, tool calls/results made in THIS turn, and the assistant response.
        Brief context from recent prior turns is included without smearing old tool executions.
        """
        parts = []

        # Find where this turn started (last user message)
        last_user = max(
            (i for i, m in enumerate(history) if isinstance(m, dict) and m.get("role") == "user"),
            default=-1,
        )

        # Collect human user directives across the session, budget-bounded
        # User messages carry crucial decisions, authorizations, and constraints.
        user_msgs = []
        if last_user > 0:
            for msg in history[:last_user]:
                if isinstance(msg, dict) and msg.get("role") == "user":
                    user_msgs.append(msg.get("content") or "")

        directives = _collect_directives(user_msgs)
        if directives:
            formatted_directives = [f"- User: {d}" for d in directives]
            parts.append("### User Directives & History (Session Context)\n" + "\n".join(formatted_directives))

        # Brief summary of recent assistant turns for continuity (last 3 assistant responses)
        recent_asst = []
        if last_user > 0:
            for msg in history[:last_user]:
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    c = (msg.get("content") or "").strip()
                    if c:
                        recent_asst.append(f"- Assistant: {c[:250]}")
            if recent_asst:
                parts.append("### Recent Assistant Context\n" + "\n".join(recent_asst[-3:]))

        # Current turn user message
        parts.append(f"### Current Turn Being Reviewed\n#### User\n\n{(user_message or '').strip()}")

        # Build tool-call and result summary for the CURRENT TURN ONLY
        tool_calls: list[str] = []
        tool_results: list[str] = []

        current_turn_msgs = history[last_user + 1:] if last_user >= 0 else history
        for msg in current_turn_msgs:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "assistant":
                for tool_call in msg.get("tool_calls") or []:
                    if not isinstance(tool_call, dict):
                        continue
                    function = tool_call.get("function") or {}
                    tc_name = function.get("name") or tool_call.get("name") or "?"
                    tc_args = function.get("arguments", tool_call.get("arguments", {}))
                    if isinstance(tc_args, str):
                        try:
                            tc_args = json.loads(tc_args)
                        except json.JSONDecodeError:
                            pass
                    if isinstance(tc_args, (dict, list)):
                        tc_text = json.dumps(tc_args, ensure_ascii=False, indent=1)
                    else:
                        tc_text = str(tc_args)
                    tool_calls.append(f"→ tool `{tc_name}`: {tc_text[:800]}")

                # Retain compatibility with providers that use content blocks.
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict):
                            if block.get("type") in {"toolCall", "tool_use"}:
                                tc_name = block.get("name", "?")
                                tc_args = block.get("arguments", block.get("input", {}))
                                tc_str = json.dumps(
                                    tc_args, ensure_ascii=False, indent=1
                                )[:800]
                                tool_calls.append(f"→ tool `{tc_name}`: {tc_str}")
            elif role == "tool":
                if isinstance(content, list):
                    text = " ".join(
                        b.get("text", "") for b in content
                        if isinstance(b, dict) and b.get("type") == "text"
                    )
                elif isinstance(content, dict):
                    text = json.dumps(content, ensure_ascii=False)
                else:
                    text = str(content)
                if text.strip():
                    tool_name = msg.get("name") or "tool"
                    tool_results.append(
                        f"→ `{tool_name}` result: {text.strip()[:1200]}"
                    )

        if tool_calls:
            parts.append("#### Tool calls in this turn\n\n" + "\n".join(tool_calls))
        if tool_results:
            parts.append("#### Tool results in this turn\n\n" + "\n".join(tool_results))

        # Assistant response (clipped to 4000 chars, preserving head and tail)
        if response and response.strip():
            resp_text = response.strip()
            if len(resp_text) > 4000:
                resp_text = resp_text[:1500] + "\n... [truncated] ...\n" + resp_text[-2500:]
            parts.append(f"#### Assistant\n\n{resp_text}")

        return "\n\n".join(parts)

    # ── deliver advice back to the conversation ──────────────────────────

    def _deliver_advice(self, advice_list: list[Advice], session_id: str = ""):
        """Inject advice into the active conversation.

        Uses ctx.inject_message(). Delivers to CLI, Desktop/TUI, or Gateway.
        """
        if not advice_list:
            return

        lines = []
        for a in advice_list:
            lines.append(f"{a.tag()} {a.note}")

        advisory_text = "\n".join(lines)
        full_msg = (
            f"◆ Advisor review (automated reviewer, NOT the user)\n\n"
            f"{advisory_text}\n\n"
            f"[Directive: This message was generated by an automated background reviewer, NOT the user. "
            f"First answer the user's most recent actual message (the one before this review). "
            f"Apply any technical fixes above secondarily and briefly. Never attribute this review to the user.]"
        )

        ok = self.ctx.inject_message(full_msg, role="user", session_key=session_id or None)
        if ok:
            logger.info("Advisor: injected %d item(s) into conversation", len(advice_list))
        else:
            logger.info(
                "Advisor: %d item(s) not injected (delivery unavailable).",
                len(advice_list),
            )

    # ── interactive model selector ───────────────────────────────────────

    def _interactive_select(self) -> str | None:
        """Open Hermes' native provider/model modal for the advisor slot."""

        def apply_selection(result) -> str:
            if not result.success:
                return f"Advisor model selection failed: {result.error_message}"
            with self._state_lock:
                self.state.model = result.new_model
                self.state.provider = result.target_provider
                self._save_state()
            return (
                f"Advisor model set to: {result.new_model} "
                f"({result.provider_label or result.target_provider})"
            )

        # Optional seam: builds without ``request_model_selection`` keep the
        # text commands (``/advisor provider``, ``/advisor model <name>``) and
        # say so, instead of raising AttributeError on every picker attempt.
        request_selection = getattr(self.ctx, "request_model_selection", None)
        if not callable(request_selection):
            return (
                "Interactive advisor model selection is not available in this "
                "Hermes build.\nUse /advisor provider <name> and "
                "/advisor model <name> here."
            )

        with self._state_lock:
            current_provider = self.state.provider
            current_model = self.state.model
        opened = request_selection(
            apply_selection,
            current_provider=current_provider,
            current_model=current_model,
        )
        if opened:
            return None
        return (
            "Interactive advisor model selection is available in the Hermes CLI.\n"
            "Use /advisor model <name> and /advisor provider <name> here."
        )

    # ── slash command ─────────────────────────────────────────────────────

    def handle_command(self, args: str) -> str | None:
        """Handle /advisor [on|off|status|model|provider|providers|models|test].

        Keywords match case-insensitively; values (model names, provider
        slugs, test notes) keep the case the user typed.
        """
        raw = args.strip()
        tokens = raw.split(None, 1)
        head = tokens[0].lower() if tokens else ""
        value = tokens[1].strip() if len(tokens) > 1 else ""

        # ── status ──
        if head in ("", "status", "config") and not (head == "config" and value):
            return self._format_status()

        # ── on ──
        if head == "on":
            self.state.enabled = True
            self._save_state()
            return "Advisor on."

        # ── off ──
        if head == "off":
            self.state.enabled = False
            self._clear_held_notes()
            self._save_state()
            return "Advisor off."

        # ── model (no args) — open the native provider/model picker ──
        if head == "model" and not value:
            return self._interactive_select()

        # ── model <name> ──
        if head == "model":
            self.state.model = value
            self._save_state()
            return f"Advisor model set to: {value}"

        # Provider is selected as the first stage of /advisor model.
        if head == "provider" and not value:
            return "Provider selection is part of /advisor model."

        # ── provider <name> ──
        if head == "provider":
            self.state.provider = value
            self._save_state()
            return f"Advisor provider set to: {value}"

        # ── config <key> <value> ──
        if head == "config":
            sub_tokens = value.split(None, 1)
            sub = sub_tokens[0].lower() if sub_tokens else ""
            sub_value = sub_tokens[1].strip() if len(sub_tokens) > 1 else ""
            if sub == "model" and sub_value:
                self.state.model = sub_value
                self._save_state()
                return f"Advisor model set to: {sub_value}"
            if sub == "provider" and sub_value:
                self.state.provider = sub_value
                self._save_state()
                return f"Advisor provider set to: {sub_value}"
            return "Usage: /advisor config <model|provider> <value>"

        # ── providers — the picker's provider rows (no private catalog) ──
        if head in ("providers", "list-providers"):
            return self._list_providers()

        # ── models [provider] — the picker's model rows (no private catalog) ──
        if head in ("models", "list-models"):
            return self._list_models(value)

        # ── rounds <n> ──
        if head in ("rounds", "max-rounds"):
            try:
                n = int(value)
                if n < 1 or n > 20:
                    return "Rounds must be between 1 and 20."
                self.state.max_rounds = n
                self._save_state()
                return f"Advisor max rounds set to: {n}"
            except ValueError:
                return "Usage: /advisor rounds <1-20>"

        # ── test — inject a test advice message ──
        if head == "test":
            import re

            m = re.match(r"(nit|concern|blocker)\s+([\s\S]+)$", value, re.IGNORECASE)
            if m:
                sev = Severity(m.group(1).lower())
                note = m.group(2).strip()
                self._deliver_advice([Advice(note=note, severity=sev)])
                return f"Advisor: delivered test {sev.value}."
            return "Usage: /advisor test <nit|concern|blocker> <note>"

        return "Usage: /advisor [on|off|status|config|model|provider|providers|models|rounds|test]"

    def _format_status(self) -> str:
        state = "enabled" if self.state.enabled else "disabled"
        model = self.state.model or "(inherit primary)"
        provider = self.state.provider or "(inherit primary)"
        max_rounds = getattr(self.state, "max_rounds", 2)
        held = self._held_count()
        return (
            f"Advisor {state}.\n"
            f"  model:      {model}\n"
            f"  provider:   {provider}\n"
            f"  max rounds: {max_rounds}\n"
            f"  held:       {held}\n"
            f"Usage: /advisor [on|off|status|config|model|provider|rounds]"
        )

    def _provider_rows(self) -> list[dict]:
        """Provider rows for the text listings, from the documented plugin seam.

        ``providers`` (``list_providers`` / ``get_provider_profile``) is the
        documented extension point — the same provider registry the native
        ``/model`` picker is built from — whereas ``hermes_cli.inventory`` is
        core-internal and carries no compatibility promise (catalog rule 9).

        The interactive path is unaffected: ``/advisor model`` opens Hermes' own
        picker through ``ctx.request_model_selection``, so no catalog of the
        advisor's own is ever needed for selection. These rows only feed the
        text listings, for surfaces where a modal cannot open (gateway, plain
        output).
        """
        try:
            from providers import list_providers

            profiles = list_providers()
        except Exception as exc:
            logger.warning("Advisor: could not load the provider registry: %s", exc)
            return []

        with self._state_lock:
            current = (self.state.provider or "").strip().lower()
        rows = []
        for profile in profiles:
            slug = (getattr(profile, "name", "") or "").strip()
            if not slug:
                continue
            label = (getattr(profile, "display_name", "") or "").strip() or slug
            rows.append({
                "slug": slug,
                "name": label,
                "is_current": bool(current) and slug.lower() == current,
                "models": list(getattr(profile, "fallback_models", []) or []),
            })
        return rows

    def _list_providers(self) -> str:
        providers = self._provider_rows()
        if not providers:
            return (
                "No providers found in the plugin provider registry. "
                "Add one via hermes setup or model.provider in config.yaml."
            )
        lines = ["Providers available to /advisor model:"]
        for p in providers:
            slug = p.get("slug") or "?"
            label = p.get("name") or slug
            marker = "  (current)" if p.get("is_current") else ""
            lines.append(f"  {slug}  —  {label}{marker}")
        return "\n".join(lines)

    def _list_models(self, provider_arg: str) -> str:
        target = (provider_arg or "").strip()
        with self._state_lock:
            target = target or self.state.provider or ""
        if not target:
            return "Usage: /advisor models <provider>\n(run /advisor providers first)"
        try:
            from providers import get_provider_profile

            profile = get_provider_profile(target)
        except Exception as exc:
            logger.warning("Advisor: provider lookup failed for %r: %s", target, exc)
            profile = None
        if profile is None:
            return (
                f"No provider '{target}' found. Run /advisor providers for valid names."
            )
        # A profile owns its catalog: fetch live ids (public contract, the profile
        # resolves its own credentials), else fall back to its curated list.
        models: list[str] = []
        try:
            models = [m for m in (profile.fetch_models() or []) if isinstance(m, str)]
        except Exception as exc:
            logger.debug("Advisor: catalog fetch failed for %s: %s", target, exc)
        if not models:
            models = [m for m in (getattr(profile, "fallback_models", []) or []) if isinstance(m, str)]
        total = None
        if not models:
            return (
                f"No models listed for '{target}'. Open /advisor model to pick "
                "interactively, or refresh with /model --refresh."
            )
        lines = [f"Models for {target}:"]
        shown = sorted(models)[:30]
        for m in shown:
            lines.append(f"  {m}")
        hidden = len(models) - len(shown)
        capped = (total - len(models)) if isinstance(total, int) else 0
        if hidden or capped > 0:
            lines.append(f"  ... and {hidden + max(capped, 0)} more")
        return "\n".join(lines)
