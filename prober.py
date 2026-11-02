"""Active Autonomous Probe Engine for Advisor Plugin.

Provides read-only, non-mutating active probes to verify common system and repo assertions
without requiring an extra agent turn:
- Repo clean state (git status --porcelain)
- File existence / non-empty checks
- Service / Process presence (launchctl / pgrep)

All probes are bounded by strict timeouts (<= 2.0s) and executed in a read-only manner.
"""

from __future__ import annotations

import os
import re
import subprocess
from typing import Any

PROBE_TIMEOUT_SECONDS = 2.0


class ProbeResult:
    def __init__(self, executed: bool, success: bool, claim: str, command: str, output: str):
        self.executed = executed
        self.success = success
        self.claim = claim
        self.command = command
        self.output = output.strip()

    def format_evidence(self) -> str:
        if not self.executed:
            status_str = "INCONCLUSIVE"
        else:
            status_str = "CONFIRMED" if self.success else "CONTRADICTED"
        return f"[Active Probe {status_str}]: `{self.command}` -> {self.output[:200]}"


def probe_git_clean_status(cwd: str) -> ProbeResult | None:
    """Safely probe if working tree is clean."""
    if not os.path.isdir(os.path.join(cwd, ".git")):
        # Check if parent is git
        try:
            res = subprocess.run(
                ["git", "-C", cwd, "rev-parse", "--is-inside-work-tree"],
                capture_output=True,
                text=True,
                timeout=PROBE_TIMEOUT_SECONDS,
            )
            if res.returncode != 0:
                return None
        except Exception:
            return None

    try:
        proc = subprocess.run(
            ["git", "--no-optional-locks", "-C", cwd, "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=PROBE_TIMEOUT_SECONDS,
        )
        if proc.returncode == 0:
            is_clean = len(proc.stdout.strip()) == 0
            out = "working tree clean" if is_clean else f"uncommitted changes present: {proc.stdout.strip()[:100]}"
            return ProbeResult(
                executed=True,
                success=is_clean,
                claim="working tree clean",
                command=f"git -C {cwd} status --porcelain",
                output=out,
            )
    except Exception as e:
        return ProbeResult(
            executed=False,
            success=False,
            claim="working tree clean",
            command=f"git -C {cwd} status --porcelain",
            output=f"Probe inconclusive: {e}",
        )
    return None


def probe_git_branch(cwd: str) -> ProbeResult | None:
    """Safely probe current git branch."""
    if not os.path.isdir(os.path.join(cwd, ".git")):
        try:
            res = subprocess.run(
                ["git", "-C", cwd, "rev-parse", "--is-inside-work-tree"],
                capture_output=True,
                text=True,
                timeout=PROBE_TIMEOUT_SECONDS,
            )
            if res.returncode != 0:
                return None
        except Exception:
            return None

    try:
        proc = subprocess.run(
            ["git", "--no-optional-locks", "-C", cwd, "branch", "--show-current"],
            capture_output=True,
            text=True,
            timeout=PROBE_TIMEOUT_SECONDS,
        )
        if proc.returncode == 0:
            branch = proc.stdout.strip()
            return ProbeResult(
                executed=True,
                success=bool(branch),
                claim=f"current git branch: {branch}",
                command=f"git -C {cwd} branch --show-current",
                output=branch,
            )
    except Exception as e:
        return ProbeResult(
            executed=False,
            success=False,
            claim="current git branch",
            command=f"git -C {cwd} branch --show-current",
            output=f"Probe inconclusive: {e}",
        )
    return None


def probe_file_existence(path: str) -> ProbeResult:
    """Safely probe if a file exists on disk."""
    exists = os.path.exists(path)
    return ProbeResult(
        executed=True,
        success=exists,
        claim=f"file exists: {path}",
        command=f"test -e {path}",
        output=f"exists={exists}",
    )


def probe_service_running(label: str) -> ProbeResult | None:
    """Safely probe if a launchd service is alive on macOS."""
    try:
        proc = subprocess.run(
            ["launchctl", "list"],
            capture_output=True,
            text=True,
            timeout=PROBE_TIMEOUT_SECONDS,
        )
        if proc.returncode == 0:
            for line in proc.stdout.splitlines():
                parts = line.split()
                if parts and parts[-1] == label:
                    pid = parts[0]
                    is_alive = pid.isdigit() and int(pid) > 0
                    return ProbeResult(
                        executed=True,
                        success=is_alive,
                        claim=f"service {label} running",
                        command=f"launchctl list | grep {label}",
                        output=f"entry={line.strip()} (alive={is_alive})",
                    )
    except Exception:
        pass
    return None


def run_active_probes_for_draft(draft: str, cwd: str) -> list[ProbeResult]:
    """Inspect draft for verifiable assertions and run safe read-only probes."""
    probes: list[ProbeResult] = []

    # 1. Probe git clean status if asserted
    if re.search(r"(?i)\b(working tree (is )?clean|git status clean|clean working tree|git tree (is )?clean|no uncommitted changes)\b", draft):
        res = probe_git_clean_status(cwd)
        if res:
            probes.append(res)

    # 2. Probe service running if asserted
    if re.search(r"(?i)\b(ai\.hermes\.gateway|gateway is running|gateway restarted)\b", draft):
        res = probe_service_running("ai.hermes.gateway")
        if res:
            probes.append(res)

    # 3. Probe git branch if claimed
    branch_m = re.search(r"(?i)\b(?:on|switched to|checked out|on feature|on branch)\s+(?:branch\s+)?([a-zA-Z0-9_\-\.\/]+)\b", draft)
    if branch_m:
        claimed_branch = branch_m.group(1).strip()
        if claimed_branch.lower() not in {"the", "a", "an", "our", "this", "that", "clean", "dirty", "main", "master"}:
            res = probe_git_branch(cwd)
            if res and res.executed:
                res.success = (res.output == claimed_branch)
                res.claim = f"on branch {claimed_branch}"
                probes.append(res)

    # 4. Probe file existence if explicit file creation or existence is asserted with a file path
    file_matches = re.findall(
        r"(?i)(?:created|written|saved|added|exists?)\s+(?:in|to|at)?\s*[`'\"]?([/~][a-zA-Z0-9_\-./]+(?:\.[a-zA-Z0-9_\-]+))[`'\"]?",
        draft,
    )
    for p in file_matches:
        expanded_path = os.path.expanduser(p)
        if os.path.isabs(expanded_path):
            probes.append(probe_file_existence(expanded_path))

    return probes
