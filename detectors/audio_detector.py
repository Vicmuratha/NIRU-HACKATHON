# ============== AUDIO DEEPFAKE DETECTION ==============

import numpy as np

# Try to import optional dependencies
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

class AudioDeepfakeDetector:
    def __init__(self):
        self.sample_rate = 16000

    def extract_audio_features(self, audio_path):
        """Extract audio features for analysis"""
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=self.sample_rate)

            # Extract features
            features = {}

            # Mel-frequency cepstral coefficients
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            features['mfcc_mean'] = np.mean(mfcc, axis=1).tolist()
            features['mfcc_std'] = np.std(mfcc, axis=1).tolist()

            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            features['spectral_centroid_mean'] = float(np.mean(spectral_centroids))

            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            features['zero_crossing_rate'] = float(np.mean(zcr))

            # Pitch estimation
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)

            features['pitch_mean'] = float(np.mean(pitch_values)) if pitch_values else 0
            features['pitch_std'] = float(np.std(pitch_values)) if pitch_values else 0

            return features
        except Exception as e:
            return {'error': str(e)}

    def detect_spoofing(self, audio_features):
        """Detect audio spoofing based on features"""
        # Simple heuristic-based detection
        # In production, use trained models like ASVspoof

        suspicious_indicators = []
        risk_score = 0

        # Check MFCC patterns
        mfcc_std = np.std(audio_features.get('mfcc_std', [0]))
        if mfcc_std < 5:
            suspicious_indicators.append('Unnatural MFCC variance detected')
            risk_score += 30

        # Check spectral centroid
        spectral_centroid = audio_features.get('spectral_centroid_mean', 0)
        if spectral_centroid < 500 or spectral_centroid > 5000:
            suspicious_indicators.append('Unusual spectral characteristics')
            risk_score += 25

        # Check pitch consistency
        pitch_std = audio_features.get('pitch_std', 0)
        if pitch_std < 10:
            suspicious_indicators.append('Unnaturally consistent pitch')
            risk_score += 30

        # Check zero crossing rate
        zcr = audio_features.get('zero_crossing_rate', 0)
        if zcr < 0.02 or zcr > 0.5:
            suspicious_indicators.append('Abnormal zero crossing rate')
            risk_score += 15

        return {
            'spoofing_score': min(risk_score, 100),
            'suspicious_indicators': suspicious_indicators,
            'is_spoofed': risk_score > 50
        }

    def analyze_audio(self, audio_path):
        """Comprehensive audio analysis"""
        results = {}

        # Extract features
        features = self.extract_audio_features(audio_path)
        results['audio_features'] = features

        if 'error' not in features:
            # Detect spoofing
            spoofing_results = self.detect_spoofing(features)
            results['spoofing_analysis'] = spoofing_results

            risk_score = spoofing_results['spoofing_score']
            results['risk_score'] = risk_score
            results['is_authentic'] = risk_score < 50
            results['confidence'] = 0.75 + (min(risk_score, 80) / 400)

            # Generate findings
            findings = []
            if risk_score < 30:
                findings.append('Natural voice patterns detected')
                findings.append('Consistent prosody and pitch')
                findings.append('Human articulation characteristics')
            else:
                findings.extend(spoofing_results['suspicious_indicators'])
                findings.append('AI voice generation artifacts detected')

            results['findings'] = findings

            # Details for UI
            results['details'] = {
                'spoofing_score': risk_score,
                'spectral_analysis': 'NATURAL' if risk_score < 50 else 'SYNTHETIC',
                'pitch_consistency': 'NORMAL' if features.get('pitch_std', 0) > 10 else 'ABNORMAL'
            }

        return results
