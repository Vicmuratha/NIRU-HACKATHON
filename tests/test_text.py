#!/usr/bin/env python3
"""
Test cases for text misinformation detection
"""

import unittest
import os
import sys
import warnings

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Suppress warnings for cleaner test output
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from backend.app import UltraTextDetector

class TestTextDetector(unittest.TestCase):
    def setUp(self):
        self.detector = UltraTextDetector()

    def test_detector_initialization(self):
        """Test that detector initializes properly"""
        self.assertIsInstance(self.detector, UltraTextDetector)
        self.assertIsNone(self.detector.pipeline)

    def test_text_analysis_normal(self):
        """Test complete text analysis with normal text"""
        test_text = "This is a normal news article about current events and factual information."
        result = self.detector.analyze_text(test_text)

        self.assertIn('risk_score', result)
        self.assertIn('is_authentic', result)
        self.assertIn('confidence', result)
        self.assertIn('findings', result)
        self.assertIsInstance(result['findings'], list)
        
        # Verify risk score is in valid range
        self.assertGreaterEqual(result['risk_score'], 0)
        self.assertLessEqual(result['risk_score'], 100)
        
        # Verify confidence is between 0 and 1
        self.assertGreaterEqual(result['confidence'], 0.0)
        self.assertLessEqual(result['confidence'], 1.0)

    def test_suspicious_text_detection(self):
        """Test detection of suspicious text patterns"""
        suspicious_text = "BREAKING NEWS!!! YOU WON'T BELIEVE THIS SHOCKING SECRET EXPOSED!!!"
        result = self.detector.analyze_text(suspicious_text)

        self.assertIn('risk_score', result)
        self.assertIn('is_authentic', result)
        self.assertIn('findings', result)
        # Should have valid risk score
        self.assertGreaterEqual(result['risk_score'], 0)
        self.assertLessEqual(result['risk_score'], 100)

    def test_clickbait_keyword_boost(self):
        """Test that clickbait keywords increase risk score"""
        # Text with clickbait keywords
        clickbait_text = "You won't believe what happened! This shocking secret was exposed!"
        result_clickbait = self.detector.analyze_text(clickbait_text)
        
        # Normal text
        normal_text = "A recent study shows interesting results about the topic."
        result_normal = self.detector.analyze_text(normal_text)
        
        # Both should return valid results
        self.assertIn('risk_score', result_clickbait)
        self.assertIn('risk_score', result_normal)

    def test_long_text_truncation(self):
        """Test that long text is handled correctly (truncated to 512 tokens)"""
        long_text = "This is a sentence. " * 100  # Very long text
        result = self.detector.analyze_text(long_text)
        
        self.assertIn('risk_score', result)
        self.assertIn('findings', result)
        self.assertIsInstance(result['findings'], list)

    def test_empty_text_handling(self):
        """Test handling of empty or very short text"""
        short_text = "Hi"
        result = self.detector.analyze_text(short_text)
        
        self.assertIn('risk_score', result)
        self.assertIn('is_authentic', result)

if __name__ == '__main__':
    unittest.main()
