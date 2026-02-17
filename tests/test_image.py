#!/usr/bin/env python3
"""
Test cases for image deepfake detection
"""

import unittest
import os
import sys
from PIL import Image
import numpy as np
import tempfile
import warnings

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Suppress warnings for cleaner test output
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from backend.app import UltraImageDetector

class TestImageDetector(unittest.TestCase):
    def setUp(self):
        self.detector = UltraImageDetector()

    def test_detector_initialization(self):
        """Test that detector initializes properly"""
        self.assertIsInstance(self.detector, UltraImageDetector)

    def test_error_level_analysis(self):
        """Test Error Level Analysis functionality"""
        # Create a test image
        test_image = Image.new('RGB', (100, 100), color='red')
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            test_path = temp_file.name
            test_image.save(test_path)

        try:
            result = self.detector.error_level_analysis(test_path)
            self.assertIn('ela_score', result)
            self.assertIn('assessment', result)
            self.assertIn('risk', result)
            self.assertIsInstance(result['ela_score'], float)
        finally:
            if os.path.exists(test_path):
                os.remove(test_path)

    def test_metadata_extraction(self):
        """Test metadata extraction"""
        # Create a test image
        test_image = Image.new('RGB', (100, 100), color='blue')
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            test_path = temp_file.name
            test_image.save(test_path)

        try:
            result = self.detector.analyze_metadata(test_path)
            self.assertIn('has_metadata', result)
            self.assertIn('metadata_count', result)
            self.assertIn('risk', result)
        finally:
            if os.path.exists(test_path):
                os.remove(test_path)

    def test_face_texture_analysis(self):
        """Test face texture analysis functionality"""
        # Create a test image (no faces expected in solid color)
        test_image = Image.new('RGB', (100, 100), color='gray')
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            test_path = temp_file.name
            test_image.save(test_path)

        try:
            sharpness = self.detector.get_sharpness(test_path)
            result = self.detector.analyze_face_texture(test_path, sharpness)
            self.assertIn('faces_detected', result)
            self.assertIn('risk', result)
            self.assertIn('assessment', result)
            self.assertIsInstance(result['faces_detected'], int)
        finally:
            if os.path.exists(test_path):
                os.remove(test_path)

    def test_full_analysis(self):
        """Test complete image analysis"""
        # Create a test image
        test_image = Image.new('RGB', (200, 200), color='green')
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            test_path = temp_file.name
            test_image.save(test_path)

        try:
            result = self.detector.analyze_image(test_path)
            self.assertIn('risk_score', result)
            self.assertIn('verdict', result)
            self.assertIn('confidence', result)
            self.assertIn('findings', result)
            self.assertIsInstance(result['findings'], list)
            
            # Verify risk score is in valid range
            self.assertGreaterEqual(result['risk_score'], 0)
            self.assertLessEqual(result['risk_score'], 100)
            
            # Verify confidence is between 0 and 1
            self.assertGreaterEqual(result['confidence'], 0.0)
            self.assertLessEqual(result['confidence'], 1.0)
        finally:
            if os.path.exists(test_path):
                os.remove(test_path)

if __name__ == '__main__':
    unittest.main()
