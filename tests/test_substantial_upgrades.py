"""Unit tests for substantial verifier upgrades: ledger, prober, dialectic."""

import os
import subprocess
import tempfile
import unittest
try:
    from hermes_plugins.advisor.dialectic import DialecticAuditor, parse_json_safely
    from hermes_plugins.advisor.ledger import audit_draft_ledger
    from hermes_plugins.advisor.prober import (
        probe_file_existence,
        probe_git_clean_status,
        probe_service_running,
        run_active_probes_for_draft,
    )
except ImportError:
    from advisor.dialectic import DialecticAuditor, parse_json_safely
    from advisor.ledger import audit_draft_ledger
    from advisor.prober import (
        probe_file_existence,
        probe_git_clean_status,
        probe_service_running,
        run_active_probes_for_draft,
    )


class TestDeterministicLedger(unittest.TestCase):
    def test_rejects_empty_and_placeholder_drafts(self):
        v = audit_draft_ledger("", [])
        self.assertFalse(v.passed)
        self.assertEqual(v.rule, "empty_draft")

        v = audit_draft_ledger("Testing verify_turn", [])
        self.assertFalse(v.passed)
        self.assertEqual(v.rule, "placeholder_draft")

        v = audit_draft_ledger("...", [])
        self.assertFalse(v.passed)
        self.assertEqual(v.rule, "placeholder_draft")

    def test_corroboration_rule_catches_unrun_test_claims(self):
        # Claim tests pass, but no test tools ran
        v = audit_draft_ledger("All 36 tests pass cleanly.", tool_events=[])
        self.assertFalse(v.passed)
        self.assertEqual(v.rule, "unbacked_assertion")
        self.assertTrue(len(v.missing_checks) > 0)
        self.assertIn("pytest or unittest", v.missing_checks[0])

        # Claim tests pass, and unittest was run in tool_events
        v_ok = audit_draft_ledger(
            "All 36 tests pass cleanly.",
            tool_events=[{"tool": "terminal", "args": "python -m unittest", "result": "Ran 36 tests. OK"}]
        )
        self.assertTrue(v_ok.passed)

    def test_corroboration_rule_catches_unrun_git_status_claim(self):
        v = audit_draft_ledger("The working tree is clean and ready.", tool_events=[])
        self.assertFalse(v.passed)
        self.assertEqual(v.rule, "unbacked_assertion")
        self.assertIn("git status", v.missing_checks[0])

        v_ok = audit_draft_ledger(
            "The working tree is clean and ready.",
            tool_events=[{"tool": "terminal", "args": "git status", "result": "nothing to commit, working tree clean"}]
        )
        self.assertTrue(v_ok.passed)

        # Expanded git clean phrasing: 'no uncommitted changes'
        v_no_uncommitted = audit_draft_ledger("There are no uncommitted changes.", tool_events=[])
        self.assertFalse(v_no_uncommitted.passed)
        self.assertEqual(v_no_uncommitted.rule, "unbacked_assertion")

    def test_corroboration_rule_catches_unrun_git_pushed_expanded(self):
        # Unrun push claim with various phrasings
        for draft in [
            "Pushed changes to origin.",
            "Pushed the branch to remote.",
            "Changes pushed to origin successfully.",
        ]:
            v = audit_draft_ledger(draft, tool_events=[])
            self.assertFalse(v.passed, f"Expected failure for: {draft}")
            self.assertEqual(v.rule, "unbacked_assertion")
            self.assertIn("git push", v.missing_checks[0])

        # Supported push
        v_ok = audit_draft_ledger(
            "Pushed changes to origin.",
            tool_events=[{"tool": "terminal", "args": "git push origin feat/x", "result": "Everything up-to-date"}]
        )
        self.assertTrue(v_ok.passed)

    def test_corroboration_rule_catches_unrun_pr_created(self):
        v = audit_draft_ledger("Opened PR #15 on GitHub.", tool_events=[])
        self.assertFalse(v.passed)
        self.assertEqual(v.rule, "unbacked_assertion")
        self.assertIn("gh pr create", v.missing_checks[0])

        v_ok = audit_draft_ledger(
            "Opened PR #15 on GitHub.",
            tool_events=[{"tool": "terminal", "args": "gh pr create --title 'x' --body 'y'", "result": "https://github.com/foo/bar/pull/15"}]
        )
        self.assertTrue(v_ok.passed)

    def test_corroboration_rule_catches_unrun_coderabbit_reviewed(self):
        v = audit_draft_ledger("CodeRabbit review passed cleanly.", tool_events=[])
        self.assertFalse(v.passed)
        self.assertEqual(v.rule, "unbacked_assertion")
        self.assertIn("coderabbit review", v.missing_checks[0])

        v_ok = audit_draft_ledger(
            "CodeRabbit review passed cleanly.",
            tool_events=[{"tool": "terminal", "args": "coderabbit review --agent", "result": "Review completed. No issues found."}]
        )
        self.assertTrue(v_ok.passed)

    def test_catches_unacknowledged_tool_errors(self):
        tool_events = [
            {"tool": "terminal", "args": "git push", "result": "error: failed to push some refs, exit_code: 1"}
        ]
        # Draft claims unqualified success without mentioning failure
        v = audit_draft_ledger("All clean and done successfully.", tool_events=tool_events)
        self.assertFalse(v.passed)
        self.assertEqual(v.rule, "unacknowledged_tool_failure")

        # Draft acknowledges the failure
        v_ack = audit_draft_ledger("Done successfully, except the push failed with an exit error.", tool_events=tool_events)
        self.assertTrue(v_ack.passed)

        # Test traceback pattern in tool errors
        tb_event = [{"tool": "terminal", "args": "python run.py", "result": "Traceback (most recent call last):\n  File 'x.py', line 1\nValueError: boom"}]
        v_tb = audit_draft_ledger("Everything works and is healthy.", tool_events=tb_event)
        self.assertFalse(v_tb.passed)
        self.assertEqual(v_tb.rule, "unacknowledged_tool_failure")

        # Test JSON tool output exit code matching
        json_err_event = [{"tool": "terminal", "args": "git push", "result": '{"output": "remote rejected", "exit_code": 1}'}]
        v_json = audit_draft_ledger("Everything works and is healthy.", tool_events=json_err_event)
        self.assertFalse(v_json.passed)
        self.assertEqual(v_json.rule, "unacknowledged_tool_failure")

        # Test claims text is checked when draft is non-empty
        v_claims = audit_draft_ledger(draft="Done.", claims=["All 36 tests pass"], tool_events=[])
        self.assertFalse(v_claims.passed)
        self.assertEqual(v_claims.rule, "unbacked_assertion")


class TestActiveProber(unittest.TestCase):
    def test_probe_file_existence(self):
        with tempfile.NamedTemporaryFile() as tmp:
            res = probe_file_existence(tmp.name)
            self.assertTrue(res.executed)
            self.assertTrue(res.success)

        res_missing = probe_file_existence("/tmp/non_existent_file_xyz_123.txt")
        self.assertTrue(res_missing.executed)
        self.assertFalse(res_missing.success)

    def test_probe_git_clean_status(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            subprocess.run(["git", "init"], cwd=tmp_dir, check=True, capture_output=True)
            res = probe_git_clean_status(tmp_dir)
            self.assertIsNotNone(res)
            self.assertTrue(res.executed)
            self.assertTrue(res.success)

    def test_run_active_probes_for_draft(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            subprocess.run(["git", "init"], cwd=tmp_dir, check=True, capture_output=True)
            draft = "The working tree is clean and ready."
            probes = run_active_probes_for_draft(draft, tmp_dir)
            self.assertTrue(len(probes) >= 1)
            self.assertIn("working tree clean", probes[0].claim)
            self.assertTrue(probes[0].success)

            # Test expanded phrasing 'no uncommitted changes'
            draft_expanded = "There are no uncommitted changes."
            probes_expanded = run_active_probes_for_draft(draft_expanded, tmp_dir)
            self.assertTrue(len(probes_expanded) >= 1)
            self.assertTrue(probes_expanded[0].success)


class TestDialecticAuditor(unittest.TestCase):
    def test_arbiter_dismisses_pedantic_stylistic_accusations(self):
        class MockLLMResult:
            def __init__(self, text):
                self.text = text

        # Challenger returns a pedantic stylistic accusation
        challenger_text = """[
            {
                "claim": "Area assignment succeeded",
                "category": "stylistic_omission",
                "evidence_gap": "The agent did not explain how IR remote limits work in detail."
            }
        ]"""
        # Arbiter dismisses it and approves
        arbiter_text = """{
            "status": "approved",
            "reason": "All core claims verified. Pedantic accusation dismissed."
        }"""

        calls = []
        def mock_complete(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                return MockLLMResult(challenger_text)
            return MockLLMResult(arbiter_text)

        auditor = DialecticAuditor(mock_complete)
        accusations = auditor.run_challenger("Area assignment succeeded", "success=True", "", "")
        self.assertEqual(len(accusations), 1)

        decision = auditor.run_arbiter("Area assignment succeeded", "success=True", "", accusations)
        self.assertEqual(decision["status"], "approved")

    def test_arbiter_upholds_substantive_blocker(self):
        class MockLLMResult:
            def __init__(self, text):
                self.text = text

        challenger_text = """[
            {
                "claim": "All 36 tests pass",
                "category": "unverified_claim",
                "evidence_gap": "No test runner was executed in this turn."
            }
        ]"""
        arbiter_text = """{
            "status": "rejected",
            "reason": "Draft claims tests pass, but no test runner was executed.",
            "missing_checks": ["Run python -m unittest"],
            "unbacked_claims": ["All 36 tests pass"]
        }"""

        calls = []
        def mock_complete(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                return MockLLMResult(challenger_text)
            return MockLLMResult(arbiter_text)

        auditor = DialecticAuditor(mock_complete)
        accusations = auditor.run_challenger("All 36 tests pass", "No tools ran", "", "")
        decision = auditor.run_arbiter("All 36 tests pass", "No tools ran", "", accusations)
        self.assertEqual(decision["status"], "rejected")
        self.assertEqual(decision["missing_checks"], ["Run python -m unittest"])

    def test_dialectic_handles_challenger_failure_and_non_dict_items(self):
        class MockLLMResult:
            def __init__(self, text):
                self.text = text

        # Test non-dict item filtering
        auditor = DialecticAuditor(lambda **kw: MockLLMResult('["not_a_dict", {"claim": "c", "category": "cat", "evidence_gap": "gap"}]'))
        accusations = auditor.run_challenger("draft", "tools", "", "")
        self.assertIsNotNone(accusations)
        self.assertEqual(len(accusations), 1)
        self.assertEqual(accusations[0]["claim"], "c")

        # Test failure returns None and arbiter returns error status
        auditor_fail = DialecticAuditor(lambda **kw: (_ for _ in ()).throw(RuntimeError("LLM exploded")))
        accusations_fail = auditor_fail.run_challenger("draft", "tools", "", "")
        self.assertIsNone(accusations_fail)

        res_err = auditor_fail.run_arbiter("draft", "tools", "", accusations=None)
        self.assertEqual(res_err["status"], "error")

        # Test arbiter failure returns error status
        res_arbiter_err = auditor_fail.run_arbiter("draft", "tools", "", accusations=[{"claim": "c", "category": "cat", "evidence_gap": "gap"}])
        self.assertEqual(res_arbiter_err["status"], "error")


if __name__ == "__main__":
    unittest.main()
