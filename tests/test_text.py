#!/usr/bin/env python3
"""
Test suite for UltraTextDetector – validates the real app.py API contract.
"""

import unittest
import os
import sys

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import UltraTextDetector


class TestTextDetector(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Instantiate detector once for all tests."""
        cls.detector = UltraTextDetector()

    # ── tests ────────────────────────────────────────────────
    def test_detector_initialization(self):
        """Detector should be the correct class."""
        self.assertIsInstance(self.detector, UltraTextDetector)

    def test_analyze_text_returns_expected_keys(self):
        """analyze_text() must return the documented key set."""
        result = self.detector.analyze_text(
            "This is a normal news article about current events.")
        for key in ('risk_score', 'is_authentic', 'verdict', 'confidence',
                    'findings', 'kenya_warnings', 'details'):
            self.assertIn(key, result, f"Missing key: {key}")

    def test_risk_score_range(self):
        """Risk score should be between 0 and 100."""
        result = self.detector.analyze_text("Some text to analyse.")
        self.assertGreaterEqual(result['risk_score'], 0)
        self.assertLessEqual(result['risk_score'], 100)

    def test_verdict_values(self):
        """Verdict must be one of the three allowed strings."""
        result = self.detector.analyze_text("Testing verdict values.")
        self.assertIn(result['verdict'],
                      {'LIKELY_DEEPFAKE', 'AUTHENTIC', 'REVIEW_REQUIRED'})

    def test_findings_is_list(self):
        """Findings should always be a list."""
        result = self.detector.analyze_text("A quick test string.")
        self.assertIsInstance(result['findings'], list)

    def test_details_contains_ai_fields(self):
        """Details dict must include AI label and score."""
        result = self.detector.analyze_text("Checking AI details.")
        self.assertIn('ai_label', result['details'])
        self.assertIn('ai_score', result['details'])

    def test_clickbait_boosts_risk(self):
        """Text with clickbait keywords ('exposed', 'shocking', 'secret') should
        receive the +20 clickbait penalty, pushing risk_score above a baseline."""
        clickbait = self.detector.analyze_text(
            "SHOCKING secret exposed! You won't believe what was exposed!")
        # The clickbait penalty adds up to +20 — so the score should be non-trivial
        self.assertGreaterEqual(clickbait['risk_score'], 20,
                                "Clickbait text should have a meaningful risk score")

    def test_is_authentic_matches_risk(self):
        """is_authentic should agree with the risk score threshold."""
        result = self.detector.analyze_text("A neutral factual sentence.")
        if result['risk_score'] < 50:
            self.assertTrue(result['is_authentic'])
        else:
            self.assertFalse(result['is_authentic'])

    def test_kenya_warnings_is_list(self):
        """Kenya warnings must be a (possibly empty) list."""
        result = self.detector.analyze_text("Test message.")
        self.assertIsInstance(result['kenya_warnings'], list)


if __name__ == '__main__':
    unittest.main()
