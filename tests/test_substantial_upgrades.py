"""Unit tests for substantial verifier upgrades: ledger, prober, dialectic."""

import os
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

    def test_catches_unacknowledged_tool_errors(self):
        tool_events = [
            {"tool": "terminal", "args": "git push", "result": "error: failed to push some refs, exit_code: 1"}
        ]
        # Draft claims unqualified success without mentioning failure
        v = audit_draft_ledger("All clean and done successfully.", tool_events=tool_events)
        self.assertFalse(v.passed)
        self.assertEqual(v.rule, "unacknowledged_tool_failure")

        # Draft acknowledges the failure
        v_ack = audit_draft_ledger("The push failed with an exit error, so I backed off.", tool_events=tool_events)
        self.assertTrue(v_ack.passed)


class TestActiveProber(unittest.TestCase):
    def test_probe_file_existence(self):
        res = probe_file_existence("/Users/alejandro/.hermes/plugins/advisor/plugin.yaml")
        self.assertTrue(res.executed)
        self.assertTrue(res.success)

        res_missing = probe_file_existence("/tmp/non_existent_file_xyz_123.txt")
        self.assertTrue(res_missing.executed)
        self.assertFalse(res_missing.success)

    def test_probe_git_clean_status(self):
        # /Users/alejandro/.hermes/plugins/advisor is a git repo
        cwd = "/Users/alejandro/.hermes/plugins/advisor"
        res = probe_git_clean_status(cwd)
        self.assertIsNotNone(res)
        self.assertTrue(res.executed)

    def test_run_active_probes_for_draft(self):
        cwd = "/Users/alejandro/.hermes/plugins/advisor"
        draft = "The working tree is clean and ready."
        probes = run_active_probes_for_draft(draft, cwd)
        self.assertTrue(len(probes) >= 1)
        self.assertIn("working tree clean", probes[0].claim)


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


if __name__ == "__main__":
    unittest.main()
