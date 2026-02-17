#!/usr/bin/env python3
"""
Test suite for UltraImageDetector – validates the real app.py API contract.
"""

import unittest
import os
import sys
import tempfile
from PIL import Image

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import UltraImageDetector


class TestImageDetector(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Instantiate detector once for all tests (model loading is expensive)."""
        cls.detector = UltraImageDetector()

    # ── helpers ──────────────────────────────────────────────
    @staticmethod
    def _make_test_image(width=200, height=200, color='red'):
        """Create a temporary JPEG and return its path."""
        img = Image.new('RGB', (width, height), color=color)
        fd, path = tempfile.mkstemp(suffix='.jpg')
        os.close(fd)
        img.save(path)
        return path

    # ── tests ────────────────────────────────────────────────
    def test_detector_initialization(self):
        """Detector should be the correct class."""
        self.assertIsInstance(self.detector, UltraImageDetector)

    def test_analyze_image_returns_expected_keys(self):
        """analyze_image() must return the documented key set."""
        path = self._make_test_image()
        try:
            result = self.detector.analyze_image(path)
            for key in ('risk_score', 'verdict', 'confidence', 'findings',
                        'kenya_warnings', 'details'):
                self.assertIn(key, result, f"Missing key: {key}")
        finally:
            os.remove(path)

    def test_risk_score_range(self):
        """Risk score should be between 0 and 100."""
        path = self._make_test_image()
        try:
            result = self.detector.analyze_image(path)
            self.assertGreaterEqual(result['risk_score'], 0)
            self.assertLessEqual(result['risk_score'], 100)
        finally:
            os.remove(path)

    def test_verdict_values(self):
        """Verdict must be one of the three allowed strings."""
        path = self._make_test_image()
        try:
            result = self.detector.analyze_image(path)
            self.assertIn(result['verdict'],
                          {'LIKELY_DEEPFAKE', 'AUTHENTIC', 'REVIEW_REQUIRED'})
        finally:
            os.remove(path)

    def test_findings_is_list(self):
        """Findings should always be a list."""
        path = self._make_test_image()
        try:
            result = self.detector.analyze_image(path)
            self.assertIsInstance(result['findings'], list)
        finally:
            os.remove(path)

    def test_kenya_warnings_is_list(self):
        """Kenya warnings must be a (possibly empty) list."""
        path = self._make_test_image()
        try:
            result = self.detector.analyze_image(path)
            self.assertIsInstance(result['kenya_warnings'], list)
        finally:
            os.remove(path)

    def test_details_contains_expected_fields(self):
        """Details dict must include AI confidence and ELA score."""
        path = self._make_test_image()
        try:
            details = self.detector.analyze_image(path)['details']
            self.assertIn('ai_confidence', details)
            self.assertIn('ela_score', details)
        finally:
            os.remove(path)

    def test_confidence_is_numeric(self):
        """Confidence should be a float between 0 and 1."""
        path = self._make_test_image()
        try:
            result = self.detector.analyze_image(path)
            self.assertIsInstance(result['confidence'], float)
            self.assertGreaterEqual(result['confidence'], 0)
            self.assertLessEqual(result['confidence'], 1)
        finally:
            os.remove(path)

    def test_solid_color_image_is_low_risk(self):
        """A plain solid-colour image should not be flagged as deepfake."""
        path = self._make_test_image(color='white')
        try:
            result = self.detector.analyze_image(path)
            self.assertLess(result['risk_score'], 80,
                            "Plain white image should not be high risk")
        finally:
            os.remove(path)


if __name__ == '__main__':
    unittest.main()
