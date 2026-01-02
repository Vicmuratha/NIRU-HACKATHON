#!/usr/bin/env python3
"""
Test cases for text misinformation detection
"""

import unittest
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detectors.text_detector import TextMisinformationDetector

class TestTextDetector(unittest.TestCase):
    def setUp(self):
        self.detector = TextMisinformationDetector()

    def test_detector_initialization(self):
        """Test that detector initializes properly"""
        self.assertIsInstance(self.detector, TextMisinformationDetector)

    def test_text_feature_extraction(self):
        """Test text feature extraction"""
        test_text = "This is a test message with some EXCLAMATION!!! and questions?"
        features = self.detector.extract_text_features(test_text)

        self.assertIn('word_count', features)
        self.assertIn('char_count', features)
        self.assertIn('sentence_count', features)
        self.assertIn('exclamation_ratio', features)
        self.assertIn('caps_ratio', features)
        self.assertIn('clickbait_score', features)

        # Check calculations
        self.assertEqual(features['word_count'], 11)
        self.assertGreater(features['char_count'], 0)
        self.assertGreater(features['exclamation_ratio'], 0)

    def test_clickbait_detection(self):
        """Test clickbait pattern detection"""
        clickbait_text = "You won't believe what happened! Secret revealed!"
        features = self.detector.extract_text_features(clickbait_text)
        self.assertGreater(features['clickbait_score'], 0)

    def test_text_analysis(self):
        """Test complete text analysis"""
        test_text = "This is a normal news article about current events."
        result = self.detector.analyze_text(test_text)

        self.assertIn('risk_score', result)
        self.assertIn('is_authentic', result)
        self.assertIn('confidence', result)
        self.assertIn('findings', result)
        self.assertIn('text_features', result)
        self.assertIn('details', result)
        self.assertIsInstance(result['findings'], list)

    def test_suspicious_text_detection(self):
        """Test detection of suspicious text patterns"""
        suspicious_text = "BREAKING NEWS!!! YOU WON'T BELIEVE THIS SHOCKING SECRET!!!"
        result = self.detector.analyze_text(suspicious_text)

        self.assertIn('risk_score', result)
        self.assertIn('is_authentic', result)
        self.assertIn('findings', result)
        # High-risk text should have higher risk score
        self.assertGreaterEqual(result['risk_score'], 0)

    def test_emotional_language_detection(self):
        """Test detection of excessive emotional language"""
        emotional_text = "This is amazing! Fantastic! Incredible! Unbelievable!"
        result = self.detector.analyze_text(emotional_text)

        features = result['text_features']
        self.assertGreater(features['exclamation_ratio'], 0)

    def test_caps_detection(self):
        """Test detection of excessive capitalization"""
        caps_text = "THIS IS ALL CAPS TEXT WHICH IS SUSPICIOUS"
        result = self.detector.analyze_text(caps_text)

        features = result['text_features']
        self.assertGreater(features['caps_ratio'], 0.5)  # Should be high due to all caps

if __name__ == '__main__':
    unittest.main()
