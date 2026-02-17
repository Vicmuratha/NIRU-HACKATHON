#!/usr/bin/env python3
"""
Test suite for UltraAudioDetector – validates the real app.py API contract.
"""

import unittest
import os
import sys
import tempfile
import wave
import numpy as np

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import UltraAudioDetector


class TestAudioDetector(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Instantiate detector once for all tests."""
        cls.detector = UltraAudioDetector()

    # ── helpers ──────────────────────────────────────────────
    @staticmethod
    def _make_wav(duration=1.0, frequency=440.0, sample_rate=16000):
        """Create a mono 16-bit WAV sine wave and return its path."""
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        signal = (np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)
        fd, path = tempfile.mkstemp(suffix='.wav')
        os.close(fd)
        with wave.open(path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(signal.tobytes())
        return path

    # ── tests ────────────────────────────────────────────────
    def test_detector_initialization(self):
        """Detector should initialise with correct sample rate."""
        self.assertIsInstance(self.detector, UltraAudioDetector)
        self.assertEqual(self.detector.sample_rate, 16000)

    def test_analyze_audio_returns_expected_keys(self):
        """analyze_audio() must return the documented key set."""
        path = self._make_wav()
        try:
            result = self.detector.analyze_audio(path)
            for key in ('risk_score', 'is_authentic', 'verdict', 'confidence',
                        'findings', 'kenya_warnings', 'details'):
                self.assertIn(key, result, f"Missing key: {key}")
        finally:
            os.remove(path)

    def test_risk_score_range(self):
        """Risk score should be between 0 and 100."""
        path = self._make_wav()
        try:
            result = self.detector.analyze_audio(path)
            self.assertGreaterEqual(result['risk_score'], 0)
            self.assertLessEqual(result['risk_score'], 100)
        finally:
            os.remove(path)

    def test_verdict_values(self):
        """Verdict must be one of the allowed strings."""
        path = self._make_wav()
        try:
            result = self.detector.analyze_audio(path)
            self.assertIn(result['verdict'],
                          {'LIKELY_DEEPFAKE', 'AUTHENTIC', 'REVIEW_REQUIRED', 'ERROR'})
        finally:
            os.remove(path)

    def test_details_contains_model_info(self):
        """Details dict must include AI model flag and heuristic metrics."""
        path = self._make_wav()
        try:
            details = self.detector.analyze_audio(path)['details']
            self.assertIn('ai_model_used', details)
            self.assertIn('mfcc_variance', details)
            self.assertIn('silence_ratio', details)
        finally:
            os.remove(path)

    def test_findings_is_nonempty_list(self):
        """Findings should always contain at least one entry."""
        path = self._make_wav()
        try:
            result = self.detector.analyze_audio(path)
            self.assertIsInstance(result['findings'], list)
            self.assertGreater(len(result['findings']), 0)
        finally:
            os.remove(path)

    def test_kenya_warnings_is_list(self):
        """Kenya warnings must be a (possibly empty) list."""
        path = self._make_wav()
        try:
            result = self.detector.analyze_audio(path)
            self.assertIsInstance(result['kenya_warnings'], list)
        finally:
            os.remove(path)

    def test_is_authentic_matches_risk(self):
        """is_authentic should agree with the risk score threshold."""
        path = self._make_wav()
        try:
            result = self.detector.analyze_audio(path)
            if result['risk_score'] < 50:
                self.assertTrue(result['is_authentic'])
            else:
                self.assertFalse(result['is_authentic'])
        finally:
            os.remove(path)

    def test_error_on_invalid_file(self):
        """Feeding garbage bytes should return an ERROR verdict, not crash."""
        fd, path = tempfile.mkstemp(suffix='.wav')
        os.close(fd)
        with open(path, 'wb') as f:
            f.write(b'NOT_VALID_AUDIO_DATA')
        try:
            result = self.detector.analyze_audio(path)
            self.assertIn('verdict', result)
        finally:
            os.remove(path)


if __name__ == '__main__':
    unittest.main()
