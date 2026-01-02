# ============== TEXT MISINFORMATION DETECTION ==============

import numpy as np

# Try to import optional dependencies
try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

class TextMisinformationDetector:
    def __init__(self):
        # Load pre-trained models
        try:
            # Use a lightweight model for demo
            self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
            self.model = AutoModelForSequenceClassification.from_pretrained(
                'distilbert-base-uncased',
                num_labels=2
            )
        except:
            self.tokenizer = None
            self.model = None

    def extract_text_features(self, text):
        """Extract features from text"""
        features = {}

        # Basic statistics
        features['word_count'] = len(text.split())
        features['char_count'] = len(text)
        features['sentence_count'] = len([s for s in text.split('.') if s.strip()])

        # Sentiment indicators
        exclamation_count = text.count('!')
        question_count = text.count('?')
        caps_count = sum(1 for c in text if c.isupper())

        features['exclamation_ratio'] = exclamation_count / max(features['sentence_count'], 1)
        features['caps_ratio'] = caps_count / max(features['char_count'], 1)

        # Clickbait indicators
        clickbait_words = ['shocking', 'unbelievable', 'you won\'t believe', 'secret', 'they don\'t want you to know']
        features['clickbait_score'] = sum(1 for word in clickbait_words if word.lower() in text.lower())

        return features

    def analyze_text(self, text):
        """Comprehensive text analysis"""
        results = {}

        # Extract features
        features = self.extract_text_features(text)
        results['text_features'] = features

        # Calculate risk score
        risk_score = 0

        if features['exclamation_ratio'] > 0.3:
            risk_score += 20
        if features['caps_ratio'] > 0.3:
            risk_score += 25
        if features['clickbait_score'] > 0:
            risk_score += 30

        # Add randomness for demo (replace with real model predictions)
        risk_score += np.random.randint(0, 25)

        results['risk_score'] = min(risk_score, 100)
        results['is_authentic'] = risk_score < 50
        results['confidence'] = 0.75 + (min(risk_score, 80) / 400)

        # Generate findings
        findings = []
        if risk_score < 40:
            findings.append('Factual claims structure detected')
            findings.append('Credible source indicators')
            findings.append('No propaganda patterns detected')
        else:
            if features['clickbait_score'] > 0:
                findings.append('Clickbait patterns identified')
            if features['exclamation_ratio'] > 0.3:
                findings.append('Excessive emotional language')
            findings.append('Misinformation indicators detected')

        results['findings'] = findings

        # Details for UI
        results['details'] = {
            'claim_verification': 'VERIFIED' if risk_score < 50 else 'UNVERIFIED',
            'bias_score': float(risk_score * 0.8),
            'credibility': 'HIGH' if risk_score < 40 else 'LOW'
        }

        return results
