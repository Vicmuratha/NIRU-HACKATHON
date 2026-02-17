#!/usr/bin/env python3
"""
Test cases for audio deepfake detection
"""

import unittest
import os
import sys
import numpy as np
import tempfile
import warnings

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Suppress warnings for cleaner test output
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from backend.app import UltraAudioDetector

class TestAudioDetector(unittest.TestCase):
    def setUp(self):
        self.detector = UltraAudioDetector()

    def test_detector_initialization(self):
        """Test that detector initializes properly"""
        self.assertIsInstance(self.detector, UltraAudioDetector)
        self.assertEqual(self.detector.sample_rate, 16000)
        self.assertIsNone(self.detector.ai_model)
        self.assertIsNone(self.detector.ai_processor)

    def test_full_analysis(self):
        """Test complete audio analysis with WavLM integration"""
        # Create a simple test audio signal (sine wave)
        sample_rate = 16000
        duration = 1.0
        frequency = 440.0
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio_signal = np.sin(frequency * 2 * np.pi * t).astype(np.float32)

        # Save as WAV file using soundfile
        import soundfile as sf
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_path = temp_file.name
            sf.write(temp_path, audio_signal, sample_rate)

        try:
            result = self.detector.analyze_audio(temp_path)
            
            # Verify expected keys in result
            self.assertIn('risk_score', result)
            self.assertIn('is_authentic', result)
            self.assertIn('confidence', result)
            self.assertIn('findings', result)
            self.assertIn('kenya_warnings', result)
            self.assertIn('kenya_audio_context', result)
            
            # Verify types
            self.assertIsInstance(result['risk_score'], (int, float))
            self.assertIsInstance(result['is_authentic'], bool)
            self.assertIsInstance(result['confidence'], float)
            self.assertIsInstance(result['findings'], list)
            
            # Risk score should be in valid range
            self.assertGreaterEqual(result['risk_score'], 0)
            self.assertLessEqual(result['risk_score'], 100)
            
            # Confidence should be between 0 and 1
            self.assertGreaterEqual(result['confidence'], 0.0)
            self.assertLessEqual(result['confidence'], 1.0)
            
            # Should have at least one finding
            self.assertGreater(len(result['findings']), 0)
            
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_analysis_with_short_audio(self):
        """Test analysis handles short audio files correctly"""
        sample_rate = 16000
        duration = 0.1  # Very short audio
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio_signal = np.sin(440 * 2 * np.pi * t).astype(np.float32)

        import soundfile as sf
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_path = temp_file.name
            sf.write(temp_path, audio_signal, sample_rate)

        try:
            result = self.detector.analyze_audio(temp_path)
            self.assertIn('risk_score', result)
            self.assertIsInstance(result['findings'], list)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_kenya_warnings_generation(self):
        """Test that Kenya-specific warnings are generated appropriately"""
        # Create test audio
        sample_rate = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio_signal = np.sin(440 * 2 * np.pi * t).astype(np.float32)

        import soundfile as sf
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_path = temp_file.name
            sf.write(temp_path, audio_signal, sample_rate)

        try:
            result = self.detector.analyze_audio(temp_path)
            
            # Check Kenya warnings structure
            self.assertIn('kenya_warnings', result)
            self.assertIsInstance(result['kenya_warnings'], list)
            
            # If there are warnings, verify they have required fields
            for warning in result['kenya_warnings']:
                self.assertIn('type', warning)
                self.assertIn('severity', warning)
                self.assertIn('warning', warning)
                self.assertIn('action', warning)
                
            # Check Kenya audio context
            self.assertIn('kenya_audio_context', result)
            self.assertIsInstance(result['kenya_audio_context'], dict)
            
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

if __name__ == '__main__':
    unittest.main()
