#!/usr/bin/env python3
"""
Test cases for audio deepfake detection
"""

import unittest
import os
import sys
import numpy as np
import tempfile

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detectors.audio_detector import AudioDeepfakeDetector

class TestAudioDetector(unittest.TestCase):
    def setUp(self):
        self.detector = AudioDeepfakeDetector()

    def test_detector_initialization(self):
        """Test that detector initializes properly"""
        self.assertIsInstance(self.detector, AudioDeepfakeDetector)
        self.assertEqual(self.detector.sample_rate, 16000)

    def test_audio_feature_extraction(self):
        """Test audio feature extraction"""
        # Create a simple test audio signal (sine wave)
        sample_rate = 16000
        duration = 1.0
        frequency = 440.0
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio_signal = np.sin(frequency * 2 * np.pi * t).astype(np.float32)

        # Save as WAV file
        import wave
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_path = temp_file.name
            with wave.open(temp_path, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes((audio_signal * 32767).astype(np.int16).tobytes())

        try:
            features = self.detector.extract_audio_features(temp_path)
            self.assertIn('mfcc_mean', features)
            self.assertIn('mfcc_std', features)
            self.assertIn('spectral_centroid_mean', features)
            self.assertIn('zero_crossing_rate', features)
            self.assertIn('pitch_mean', features)
            self.assertIn('pitch_std', features)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_spoofing_detection(self):
        """Test spoofing detection logic"""
        # Test with normal audio features
        normal_features = {
            'mfcc_std': [1.0, 1.1, 1.2],
            'spectral_centroid_mean': 2000.0,
            'pitch_std': 50.0,
            'zero_crossing_rate': 0.05
        }

        result = self.detector.detect_spoofing(normal_features)
        self.assertIn('spoofing_score', result)
        self.assertIn('suspicious_indicators', result)
        self.assertIn('is_spoofed', result)
        self.assertIsInstance(result['suspicious_indicators'], list)

    def test_full_analysis(self):
        """Test complete audio analysis"""
        # Create a simple test audio signal
        sample_rate = 16000
        duration = 1.0
        frequency = 440.0
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio_signal = np.sin(frequency * 2 * np.pi * t).astype(np.float32)

        # Save as WAV file
        import wave
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_path = temp_file.name
            with wave.open(temp_path, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes((audio_signal * 32767).astype(np.int16).tobytes())

        try:
            result = self.detector.analyze_audio(temp_path)
            self.assertIn('risk_score', result)
            self.assertIn('is_authentic', result)
            self.assertIn('confidence', result)
            self.assertIn('findings', result)
            self.assertIn('audio_features', result)
            self.assertIn('spoofing_analysis', result)
            self.assertIn('details', result)
            self.assertIsInstance(result['findings'], list)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

if __name__ == '__main__':
    unittest.main()
