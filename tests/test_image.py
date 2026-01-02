#!/usr/bin/env python3
"""
Test cases for image deepfake detection
"""

import unittest
import os
import sys
from PIL import Image
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detectors.image_detector import ImageDeepfakeDetector

class TestImageDetector(unittest.TestCase):
    def setUp(self):
        self.detector = ImageDeepfakeDetector()

    def test_detector_initialization(self):
        """Test that detector initializes properly"""
        self.assertIsInstance(self.detector, ImageDeepfakeDetector)
        self.assertIsNotNone(self.detector.device)

    def test_error_level_analysis(self):
        """Test Error Level Analysis functionality"""
        # Create a test image
        test_image = Image.new('RGB', (100, 100), color='red')
        test_path = 'test_image.jpg'
        test_image.save(test_path)

        try:
            result = self.detector.error_level_analysis(test_path)
            self.assertIn('ela_score', result)
            self.assertIn('suspicious', result)
            self.assertIn('confidence', result)
            self.assertIsInstance(result['ela_score'], float)
        finally:
            if os.path.exists(test_path):
                os.remove(test_path)

    def test_metadata_extraction(self):
        """Test metadata extraction"""
        # Create a test image
        test_image = Image.new('RGB', (100, 100), color='blue')
        test_path = 'test_metadata.jpg'
        test_image.save(test_path)

        try:
            result = self.detector.extract_metadata(test_path)
            self.assertIn('has_metadata', result)
            self.assertIn('metadata_count', result)
            self.assertIn('integrity', result)
        finally:
            if os.path.exists(test_path):
                os.remove(test_path)

    def test_face_detection(self):
        """Test face detection functionality"""
        # Create a test image (no faces expected)
        test_image = Image.new('RGB', (100, 100), color='gray')
        test_path = 'test_face.jpg'
        test_image.save(test_path)

        try:
            result = self.detector.detect_face_manipulation(test_path)
            self.assertIn('faces_detected', result)
            self.assertIn('face_verification', result)
            self.assertIsInstance(result['faces_detected'], int)
        finally:
            if os.path.exists(test_path):
                os.remove(test_path)

    def test_full_analysis(self):
        """Test complete image analysis"""
        # Create a test image
        test_image = Image.new('RGB', (200, 200), color='green')
        test_path = 'test_full.jpg'
        test_image.save(test_path)

        try:
            result = self.detector.analyze_image(test_path)
            self.assertIn('risk_score', result)
            self.assertIn('is_authentic', result)
            self.assertIn('confidence', result)
            self.assertIn('findings', result)
            self.assertIn('ela_analysis', result)
            self.assertIn('metadata_analysis', result)
            self.assertIn('face_analysis', result)
            self.assertIsInstance(result['findings'], list)
        finally:
            if os.path.exists(test_path):
                os.remove(test_path)

if __name__ == '__main__':
    unittest.main()
