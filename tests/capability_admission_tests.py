#!/usr/bin/env python3
"""Focused checks for the non-production v2 capability admission prototype."""

import copy
import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_capability_admission.py"
EXAMPLES = ROOT / "docs" / "spec" / "capability-vocabulary-v1.examples.json"

spec = importlib.util.spec_from_file_location("capability_admission", SCRIPT)
admission = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = admission
spec.loader.exec_module(admission)


class CapabilityAdmissionTests(unittest.TestCase):
    def test_examples_match_recorded_decisions(self) -> None:
        cases = admission.load_cases(EXAMPLES)
        self.assertEqual(len(cases), 4)
        for case in cases:
            self.assertEqual(admission.assess(case)["decision"], case["expected_decision"])

    def test_known_numerics_mismatch_is_structured_and_offers_alternatives(self) -> None:
        case = admission.load_cases(EXAMPLES)[-1]
        result = admission.assess(case)
        self.assertEqual(result["status"], "unsupported")
        self.assertEqual(result["incompatibility"], {
            "owner": "provider",
            "requirement": "numerics_profile",
            "required": "nvcr.fx.v1",
            "available": ["native"],
            "stage_id": "i_prior",
        })
        self.assertEqual(
            [alternative["provider_id"] for alternative in result["alternatives"]],
            ["onnxruntime-cpu", "tensorrt"],
        )

    def test_untested_evidence_holds_instead_of_rejecting(self) -> None:
        case = copy.deepcopy(admission.load_cases(EXAMPLES)[0])
        case["conformance"] = {"status": "untested", "evidence": "pending"}
        result = admission.assess(case)
        self.assertEqual(result["decision"], "hold")
        self.assertEqual(result["status"], "untested")

    def test_malformed_coverage_fails_cleanly(self) -> None:
        case = copy.deepcopy(admission.load_cases(EXAMPLES)[0])
        case["artifact"]["visible_size"]["min_width"] = "one"
        with self.assertRaisesRegex(admission.InputError, "positive integers"):
            admission.assess(case)


if __name__ == "__main__":
    unittest.main()
