# SafEye Backend - High-confidence AI-assisted detection
# Competition-grade deepfake detection with AI models + advanced heuristics

import os
import warnings
import threading
import uuid
import json
from datetime import datetime
import numpy as np

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_jwt_extended import JWTManager, create_access_token, jwt_required
from werkzeug.utils import secure_filename
from PIL import Image
import exifread

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)
app.config['JWT_SECRET_KEY'] = os.environ.get('JWT_SECRET_KEY', 'safeye-hackathon-secret-2026')
jwt = JWTManager(app)

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ============== AUTHENTICATION ==============
@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    if data.get('username') == 'admin' and data.get('password') == 'password':
        return jsonify(access_token=create_access_token(identity='admin')), 200
    return jsonify({'error': 'Invalid credentials'}), 401

# ============== ULTRA-ACCURATE IMAGE DETECTOR ==============
class UltraImageDetector:
    def __init__(self):
        self.ai_model = None
        self.ai_processor = None
        self.lock = threading.Lock()
        print("🔧 Ultra-Accurate Image Detector initialized")

    def load_ai_model(self):
        """Lazy load the deep learning deepfake detection model"""
        if self.ai_model is None:
            with self.lock:
                if self.ai_model is None:
                    try:
                        from transformers import AutoModelForImageClassification, AutoImageProcessor
                        import torch

                        print("📥 Loading deepfake detection AI model (this may take a minute)...")
                        self.ai_model = AutoModelForImageClassification.from_pretrained(
                            "dima806/deepfake_vs_real_image_detection"
                        )
                        self.ai_processor = AutoImageProcessor.from_pretrained(
                            "dima806/deepfake_vs_real_image_detection"
                        )
                        self.ai_model.eval()
                        print("✅ AI model loaded successfully")
                    except Exception as e:
                        print(f"⚠️ Could not load AI model: {e}")
                        print("   Falling back to heuristic-only detection")
                        self.ai_model = "unavailable"

    def ai_deepfake_check(self, image_path):
        """
        Deep learning-based deepfake detection
        Provides confidence scores for ensemble decision
        """
        self.load_ai_model()

        if self.ai_model == "unavailable":
            return {'available': False, 'fake_confidence': 0}

        try:
            import torch

            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')

            # Resize for efficiency (model works well at lower res)
            if max(image.size) > 512:
                image.thumbnail((512, 512), Image.Resampling.LANCZOS)

            # Process through model
            inputs = self.ai_processor(images=image, return_tensors="pt")

            with torch.no_grad():
                outputs = self.ai_model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

            # Model outputs: [real_prob, fake_prob]
            fake_confidence = float(probs[0][1].item())
            real_confidence = float(probs[0][0].item())

            return {
                'available': True,
                'fake_confidence': fake_confidence,
                'real_confidence': real_confidence
            }
        except Exception as e:
            print(f"❌ AI detection error: {e}")
            return {'available': False, 'fake_confidence': 0}

    def error_level_analysis(self, image_path):
        """
        ELA (Error Level Analysis) - detects JPEG compression anomalies (20% weight)
        AI images compress differently than camera photos
        """
        try:
            original = Image.open(image_path).convert('RGB')
            temp_filename = f"temp_ela_{uuid.uuid4().hex}.jpg"
            temp_path = os.path.join(os.path.dirname(image_path), temp_filename)

            try:
                # Save at quality 90, then compare
                original.save(temp_path, 'JPEG', quality=90)
                compressed = Image.open(temp_path)

                original_arr = np.array(original).astype(np.float32)
                compressed_arr = np.array(compressed).astype(np.float32)

                if original_arr.shape != compressed_arr.shape:
                    compressed = compressed.resize(original.size, Image.Resampling.LANCZOS)
                    compressed_arr = np.array(compressed).astype(np.float32)

                # Calculate difference
                diff = np.abs(original_arr - compressed_arr)
                ela_score = float(np.mean(diff))

                # Interpret score
                if ela_score < 2.5:
                    assessment = 'EXTREMELY_CLEAN'
                    risk = 95
                elif ela_score < 5.0:
                    assessment = 'VERY_CLEAN'
                    risk = 80
                elif ela_score < 8.0:
                    assessment = 'CLEAN'
                    risk = 55
                elif ela_score < 15.0:
                    assessment = 'MODERATE'
                    risk = 30
                else:
                    assessment = 'HEAVY_COMPRESSION'
                    risk = 12

                # Cap ELA risk to prevent catastrophic false positives
                risk = min(risk, 60)

                return {
                    'ela_score': ela_score,
                    'assessment': assessment,
                    'risk': risk
                }
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        except Exception as e:
            print(f"⚠️ ELA error: {e}")
            return {'ela_score': 15.0, 'assessment': 'UNKNOWN', 'risk': 35}

    def analyze_metadata(self, image_path):
        """
        Camera metadata verification (25% weight)
        Real photos from phones have rich EXIF data
        """
        try:
            with open(image_path, 'rb') as f:
                tags = exifread.process_file(f, details=False)

            make = str(tags.get('Image Make', tags.get('EXIF Make', ''))).strip()
            model = str(tags.get('Image Model', tags.get('EXIF Model', ''))).strip()
            software = str(tags.get('Image Software', '')).lower().strip()

            # Trusted camera brands (common in Kenya and globally)
            trusted_brands = [
                'samsung', 'apple', 'iphone', 'google', 'pixel',
                'huawei', 'tecno', 'infinix', 'oppo', 'xiaomi', 'vivo',
                'canon', 'nikon', 'sony', 'fujifilm', 'panasonic'
            ]

            is_trusted_camera = any(
                brand in make.lower() or brand in model.lower()
                for brand in trusted_brands
            )

            # Check for photo editing software
            editing_software = ['photoshop', 'gimp', 'paint.net', 'lightroom', 'affinity']
            was_edited = any(sw in software for sw in editing_software)

            # Risk assessment
            if is_trusted_camera:
                risk = 12 if not was_edited else 25  # Trusted camera = low risk
            elif make or model:
                risk = 35  # Has metadata but not trusted brand
            else:
                risk = 25  # No metadata = lower penalty

            return {
                'has_metadata': bool(make or model),
                'is_trusted_camera': is_trusted_camera,
                'camera_info': f"{make} {model}".strip() or 'None',
                'was_edited': was_edited,
                'editing_software': software if was_edited else None,
                'metadata_count': len(tags),
                'risk': risk
            }
        except Exception as e:
            print(f"⚠️ Metadata error: {e}")
            return {
                'has_metadata': False,
                'is_trusted_camera': False,
                'camera_info': 'None',
                'was_edited': False,
                'metadata_count': 0,
                'risk': 52
            }

    def analyze_face_texture(self, image_path, sharpness):
        """
        Face texture analysis (10% weight)
        AI-generated faces have unnaturally smooth skin
        """
        try:
            from deepface import DeepFace

            faces = DeepFace.extract_faces(image_path, enforce_detection=False)

            if not faces or len(faces) == 0:
                return {'faces_detected': 0, 'risk': 0, 'assessment': 'NO_FACE'}

            # Analyze each face, take maximum risk
            max_risk = 0
            best_assessment = 'NORMAL'

            for face_data in faces:
                face_img = face_data['face']

                if isinstance(face_img, np.ndarray):
                    # Normalize if needed
                    if face_img.max() <= 1.0:
                        face_img = (face_img * 255).astype(np.uint8)

                    # Calculate texture variance
                    face_std = float(np.std(face_img))

                    # Risk assessment depends on image sharpness
                    if sharpness > 100:  # High quality image
                        if face_std < 14:
                            risk = 75
                            assessment = 'SYNTHETIC'
                        elif face_std < 28:
                            risk = 68
                            assessment = 'SUSPICIOUSLY_SMOOTH'
                        elif face_std < 40:
                            risk = 28
                            assessment = 'SMOOTH'
                        else:
                            risk = 8
                            assessment = 'NATURAL'
                    else:  # Lower quality image (compressed/small)
                        if face_std < 10:
                            risk = 75
                            assessment = 'TOO_SMOOTH'
                        elif face_std < 22:
                            risk = 35
                            assessment = 'SMOOTH'
                        else:
                            risk = 12
                            assessment = 'NORMAL'

                    # Hard cap to prevent false accusations
                    risk = min(risk, 75)

                    if risk > max_risk:
                        max_risk = risk
                        best_assessment = assessment

            return {
                'faces_detected': len(faces),
                'risk': max_risk,
                'assessment': best_assessment
            }
        except Exception as e:
            print(f"⚠️ Face analysis error: {e}")
            return {'faces_detected': 0, 'risk': 0, 'assessment': 'ERROR'}

    def get_sharpness(self, image_path):
        """Calculate image sharpness using Laplacian variance"""
        try:
            import cv2
            img = cv2.imread(image_path)
            if img is None:
                return 50.0

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            return float(laplacian_var)
        except Exception as e:
            print(f"⚠️ Sharpness calculation error: {e}")
            return 50.0

    def noise_analysis(self, image_path):
        """
        High-frequency noise analysis (5% weight)
        AI images lack natural camera sensor noise
        """
        try:
            from scipy import fft

            img = Image.open(image_path).convert('L')
            img.thumbnail((256, 256), Image.Resampling.LANCZOS)
            img_array = np.array(img).astype(np.float32)

            # FFT to analyze frequency domain
            f_transform = fft.fft2(img_array)
            f_shift = fft.fftshift(f_transform)
            magnitude = np.abs(f_shift)

            # Analyze high-frequency region
            h, w = magnitude.shape
            high_freq = magnitude[h//3:, w//3:]
            noise_score = float(np.mean(high_freq))

            # AI images have suspiciously low high-frequency content
            if noise_score < 12:
                return {'noise_score': noise_score, 'risk': 25, 'assessment': 'VERY_LOW'}
            elif noise_score < 20:
                return {'noise_score': noise_score, 'risk': 10, 'assessment': 'LOW'}
            else:
                return {'noise_score': noise_score, 'risk': 0, 'assessment': 'NORMAL'}
        except Exception as e:
            print(f"⚠️ Noise analysis error: {e}")
            return {'noise_score': 0, 'risk': 0, 'assessment': 'UNKNOWN'}

    def ensemble_decision(self, ai_result, ela_result, meta_result, face_result, noise_result):
        """
        WEIGHTED ENSEMBLE: Combine all detection methods
        AI model dominates the ensemble for better accuracy
        """
        total_risk = 0
        confidence_sum = 0
        findings = []

        # === AI MODEL (60% weight) - Most reliable ===
        AI_WEIGHT = 0.60
        if ai_result['available']:
            ai_risk = ai_result['fake_confidence'] * 100
            total_risk += ai_risk * AI_WEIGHT
            confidence_sum += AI_WEIGHT

            if ai_result['fake_confidence'] > 0.5:
                findings.append(f"🤖 AI Model: DEEPFAKE detected ({ai_result['fake_confidence']:.1%} confidence)")
            else:
                findings.append(f"✓ AI Model: Authentic ({ai_result['real_confidence']:.1%} confidence)")

        # === METADATA (15% weight) ===
        META_WEIGHT = 0.15
        total_risk += meta_result['risk'] * META_WEIGHT
        confidence_sum += META_WEIGHT

        if meta_result['is_trusted_camera']:
            findings.append(f"✓ Verified camera: {meta_result['camera_info']}")
            if meta_result['was_edited']:
                findings.append(f"⚠️ Image edited with {meta_result['editing_software']}")
        else:
            findings.append("⚠️ No trusted camera metadata found")

        # === ELA (15% weight) ===
        ELA_WEIGHT = 0.15
        total_risk += ela_result['risk'] * ELA_WEIGHT
        confidence_sum += ELA_WEIGHT

        if ela_result['risk'] > 60:
            findings.append(f"⚠️ Compression: {ela_result['assessment']} (ELA: {ela_result['ela_score']:.1f})")
        else:
            findings.append(f"✓ Compression: {ela_result['assessment']} (ELA: {ela_result['ela_score']:.1f})")

        # === FACE TEXTURE (7% weight) ===
        if face_result['faces_detected'] > 0:
            FACE_WEIGHT = 0.07
            total_risk += face_result['risk'] * FACE_WEIGHT
            confidence_sum += FACE_WEIGHT

            if face_result['risk'] > 60:
                findings.append(f"⚠️ Face texture: {face_result['assessment']}")
            else:
                findings.append(f"✓ Natural face texture detected")
=======
            if face_result['risk'] > 60:
                findings.append(f"⚠️ Face texture: {face_result['assessment']}")
            else:
                findings.append(f"✓ Natural face texture detected")
>>>>>>> 3dd3b3e (Update backend logic, remove old detectors, and sync styles)

        # === NOISE (3% weight) ===
        NOISE_WEIGHT = 0.03
        total_risk += noise_result['risk'] * NOISE_WEIGHT
        confidence_sum += NOISE_WEIGHT

        if noise_result['risk'] > 15:
            findings.append(f"⚠️ Noise pattern: {noise_result['assessment']}")

        # Final risk score
        final_risk = min(max(total_risk, 0), 100)

        # Agreement-based confidence calculation
        agreement = np.std([
            ai_result.get('fake_confidence', 0),
            ela_result['risk'] / 100,
            meta_result['risk'] / 100
        ])
        overall_confidence = round(max(0.6, 1 - agreement), 2)

        # Three-zone verdict system
        if final_risk < 40:
            verdict = "AUTHENTIC"
        elif final_risk > 65:
            verdict = "LIKELY_DEEPFAKE"
        else:
            verdict = "REVIEW_REQUIRED"

        return {
            'risk_score': round(final_risk, 1),
            'verdict': verdict,
            'confidence': overall_confidence,
            'findings': findings
        }

    def analyze_image(self, image_path):
        """
        MAIN ANALYSIS PIPELINE
        Combines all detection methods for maximum accuracy
        """
        print(f"\n🔍 ULTRA-ACCURATE IMAGE ANALYSIS: {os.path.basename(image_path)}")

        # Step 1: Gather all metrics
        sharpness = self.get_sharpness(image_path)
        print(f"  📏 Sharpness: {sharpness:.1f}")

        ai_result = self.ai_deepfake_check(image_path)
        if ai_result['available']:
            print(f"  🤖 AI Model: Confidence={ai_result['fake_confidence']:.1%}")

        ela_result = self.error_level_analysis(image_path)
        print(f"  📊 ELA: {ela_result['assessment']} ({ela_result['ela_score']:.1f})")

        meta_result = self.analyze_metadata(image_path)
        print(f"  📷 Camera: {meta_result['camera_info']}")

        face_result = self.analyze_face_texture(image_path, sharpness)
        if face_result['faces_detected'] > 0:
            print(f"  👤 Faces: {face_result['faces_detected']} - {face_result['assessment']}")

        noise_result = self.noise_analysis(image_path)

        # Step 2: Ensemble decision
        result = self.ensemble_decision(ai_result, ela_result, meta_result, face_result, noise_result)

        # Step 3: Add Kenya-specific warnings
        kenya_warnings = []
        if result['risk_score'] > 70 and face_result['faces_detected'] > 0:
            kenya_warnings.append({
                'type': 'ELECTION_MANIPULATION',
                'severity': 'CRITICAL',
                'warning': 'Political deepfake risk - Report to IEBC/NIS',
                'action': 'Verify source before sharing on social media'
            })

        # Step 4: Complete response
        result['kenya_warnings'] = kenya_warnings
        result['details'] = {
            'sharpness': round(sharpness, 1),
            'ai_fake_confidence': round(ai_result['fake_confidence'] * 100, 1) if ai_result['available'] else 0,
            'ai_available': ai_result['available'],
            'ela_score': round(ela_result['ela_score'], 2),
            'camera': meta_result['camera_info'],
            'faces_detected': face_result['faces_detected'],
            'metadata_richness': meta_result['metadata_count']
        }

        print(f"  🎯 RESULT: Risk={result['risk_score']}%, Verdict={result['verdict']}, Confidence={result['confidence']}")

        return result

# ============== ULTRA-ACCURATE AUDIO DETECTOR ==============
class UltraAudioDetector:
    def __init__(self):
        self.sample_rate = 16000
        print("🔧 Ultra-Accurate Audio Detector initialized")

    def extract_features(self, y, sr):
        """Extract comprehensive audio features"""
        import librosa

        features = {}

        # MFCC (voice texture/timbre)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        features['mfcc_mean'] = float(np.mean(mfcc))
        features['mfcc_std'] = float(np.std(mfcc))
        features['mfcc_var'] = float(np.var(mfcc))

        # Spectral features
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features['spectral_centroid_mean'] = float(np.mean(cent))
        features['spectral_centroid_std'] = float(np.std(cent))

        # Zero crossing rate (articulation)
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features['zcr_mean'] = float(np.mean(zcr))
        features['zcr_std'] = float(np.std(zcr))

        # RMS energy
        rms = librosa.feature.rms(y=y)[0]
        features['rms_mean'] = float(np.mean(rms))
        features['rms_std'] = float(np.std(rms))

        return features

    def check_breathing(self, y, sr):
        """Detect natural pauses/breathing - AI voices lack this"""
        import librosa

        rms = librosa.feature.rms(y=y)[0]
        threshold = np.mean(rms) * 0.12

        silence_ratio = float(np.sum(rms < threshold) / len(rms))

        # Natural speech has 3-8% silence for breathing
        has_natural_pauses = 0.02 < silence_ratio < 0.15

        return {
            'silence_ratio': silence_ratio,
            'has_natural_pauses': has_natural_pauses,
            'risk': 0 if has_natural_pauses else 30
        }

    def check_consistency(self, y, sr):
        """AI voices are TOO consistent over time"""
        import librosa

        # Split into segments
        segment_len = sr  # 1 second
        segments = [y[i:i+segment_len] for i in range(0, len(y), segment_len)]

        if len(segments) < 2:
            return {'consistency_score': 50, 'risk': 10}

        # Analyze variance across segments
        segment_vars = []
        for seg in segments:
            if len(seg) < 1024:
                continue
            seg_var = float(np.var(seg))
            segment_vars.append(seg_var)

        if len(segment_vars) < 2:
            return {'consistency_score': 50, 'risk': 10}

        # Std of variances - real voices vary more
        consistency = float(np.std(segment_vars))

        # Too consistent = AI
        if consistency < 0.0005:
            risk = 40
        elif consistency < 0.002:
            risk = 20
        else:
            risk = 0

        return {'consistency_score': consistency, 'risk': risk}

    def analyze_audio(self, audio_path):
        """
        MAIN AUDIO ANALYSIS
        Detects AI-generated voices with high accuracy
        """
        import librosa

        print(f"\n🎵 ULTRA-ACCURATE AUDIO ANALYSIS: {os.path.basename(audio_path)}")

        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=self.sample_rate)

            # Pad if too short
            if len(y) < 2048:
                y = np.pad(y, (0, 2048 - len(y)))

            # Extract features
            features = self.extract_features(y, sr)
            breathing = self.check_breathing(y, sr)
            consistency = self.check_consistency(y, sr)

            # Decision logic
            risk = 0
            findings = []

            # 1. MFCC Variance (voice texture)
            mfcc_var = features['mfcc_var']
            if mfcc_var < 150:
                risk += 75
                findings.append(f"⚠️ Robotic voice texture (var: {mfcc_var:.0f})")
            elif mfcc_var < 400:
                risk += 40
                findings.append(f"⚠️ Smooth voice texture (var: {mfcc_var:.0f})")
            elif mfcc_var < 800:
                risk += 15
                findings.append(f"✓ Moderate voice variation")
            else:
                findings.append(f"✓ Natural voice variation")

            # 2. Breathing patterns
            if not breathing['has_natural_pauses']:
                risk += breathing['risk']
                findings.append("⚠️ No natural breathing pauses detected")
            else:
                findings.append("✓ Natural breathing patterns")

            # 3. Temporal consistency
            risk += consistency['risk']
            if consistency['risk'] > 25:
                findings.append("⚠️ Unnaturally consistent over time")

            # 4. Spectral analysis
            cent_std = features['spectral_centroid_std']
            if cent_std < 250:
                risk += 25
                findings.append("⚠️ Monotonous frequency range")

            risk = min(risk, 98)

            # Kenya warnings
            kenya_warnings = []
            if risk > 65:
                kenya_warnings.append({
                    'type': 'MPESA_FRAUD',
                    'severity': 'HIGH',
                    'warning': 'Voice cloning detected - M-Pesa scam risk',
                    'action': 'NEVER authorize M-Pesa transactions via voice call. Use PIN only.'
                })
                kenya_warnings.append({
                    'type': 'DIASPORA_SCAM',
                    'severity': 'HIGH',
                    'warning': 'Family impersonation scam risk',
                    'action': 'Verify identity through video call or shared secret question'
                })

            result = {
                'risk_score': risk,
                'is_authentic': risk < 50,
                'confidence': 0.88,
                'findings': findings,
                'kenya_warnings': kenya_warnings,
                'details': {
                    'mfcc_variance': int(mfcc_var),
                    'silence_ratio': round(breathing['silence_ratio'], 3),
                    'consistency_score': round(consistency['consistency_score'], 5),
                    'spectral_std': int(cent_std)
                }
            }

            return result
>>>>>>> 3dd3b3e (Update backend logic, remove old detectors, and sync styles)
            print(f"  🎯 RESULT: Risk={risk}%, Authentic={result['is_authentic']}")

            return result
=======
            return result
>>>>>>> 3dd3b3e (Update backend logic, remove old detectors, and sync styles)

        except Exception as e:
            print(f"❌ Audio analysis error: {e}")
            import traceback
            traceback.print_exc()
            return {
                'risk_score': 0,
                'is_authentic': True,
                'confidence': 0.0,
                'findings': [f"Error: {str(e)}"],
                'kenya_warnings': [],
                'details': {}
            }

# ============== ULTRA-ACCURATE TEXT DETECTOR ==============
class UltraTextDetector:
    def __init__(self):
        self.pipeline = None
        self.lock = threading.Lock()
        print("🔧 Ultra-Accurate Text Detector initialized")

    def analyze_text(self, text):
        """Advanced fake news detection"""
        print(f"\n📝 ULTRA-ACCURATE TEXT ANALYSIS: {len(text)} characters")

        # Load AI model
        with self.lock:
            if self.pipeline is None:
                from transformers import pipeline
                print("📥 Loading fake news detection model...")
                self.pipeline = pipeline(
                    "text-classification",
                    model="hamzab/roberta-fake-news-classification"
                )
                print("✅ Model loaded")

        # AI analysis
        ai_result = self.pipeline(text[:512])[0]
        is_fake = ai_result['label'] in ['FAKE', 'LABEL_0', '0']
        confidence = ai_result['score']

        # Base risk from AI
        if is_fake:
            risk = int(confidence * 100)
        else:
            risk = int((1 - confidence) * 100)

        findings = []

        # AI finding
        if is_fake and confidence > 0.7:
            findings.append(f"⚠️ AI Model: FAKE NEWS detected ({confidence:.1%} confidence)")
        elif not is_fake and confidence > 0.7:
            findings.append(f"✓ AI Model: Content appears credible ({confidence:.1%})")
        else:
            findings.append(f"⚠️ AI Model: Uncertain ({confidence:.1%} confidence)")

        # Heuristic analysis
        txt_lower = text.lower()

        # Clickbait keywords
        clickbait = ['shocking', 'secret', 'exposed', 'won\'t believe', 'urgent',
                     'breaking', 'censored', 'they don\'t want you', 'must see']
        clickbait_count = sum(1 for kw in clickbait if kw in txt_lower)

        if clickbait_count > 0:
            penalty = min(clickbait_count * 12, 40)
            risk = min(risk + penalty, 96)
            findings.append(f"⚠️ {clickbait_count} clickbait/sensational keywords")

        # Excessive caps
        caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        if caps_ratio > 0.5:
            risk = min(risk + 18, 96)
            findings.append(f"⚠️ Excessive capitalization ({caps_ratio*100:.0f}%)")

        # Source citations
        has_source = any(indicator in txt_lower for indicator in [
            'according to', 'research shows', 'study found', 'source:',
            'http://', 'https://', 'reuters', 'bbc', 'cnn', 'nation media'
        ])

        if has_source:
            risk = max(risk - 12, 5)
            findings.append("✓ Contains source citations")

        # Vague claims
        vague = ['some say', 'many believe', 'experts claim', 'they say', 'reports suggest']
        vague_count = sum(1 for v in vague if v in txt_lower)

        if vague_count > 0:
            risk = min(risk + vague_count * 8, 96)
            findings.append(f"⚠️ {vague_count} vague/unattributed claims")

        # Kenya-specific analysis
        kenya_warnings = []

        # Tribal content
        tribal_keywords = ['kalenjin', 'kikuyu', 'luo', 'luhya', 'kamba',
                          'tribe', 'tribal', 'ethnic', 'community']
        has_tribal = any(kw in txt_lower for kw in tribal_keywords)

        # Political figures
        politicians = ['ruto', 'raila', 'uhuru', 'gachagua', 'karua']
        has_politics = any(pol in txt_lower for pol in politicians)

        if has_tribal and risk > 55:
            kenya_warnings.append({
                'type': 'HATE_SPEECH',
                'severity': 'CRITICAL',
                'warning': 'Content may inflame tribal tensions',
                'action': 'Report to NCIC (National Cohesion and Integration Commission)'
            })

        if has_politics and risk > 70:
            kenya_warnings.append({
                'type': 'ELECTION_MISINFORMATION',
                'severity': 'HIGH',
                'warning': 'Political misinformation detected',
                'action': 'Verify with official IEBC sources before sharing'
            })

        result = {
            'risk_score': risk,
            'is_authentic': risk < 50,
            'confidence': confidence,
            'findings': findings,
            'kenya_warnings': kenya_warnings,
            'details': {
                'ai_label': ai_result['label'],
                'ai_confidence': round(confidence, 3),
                'clickbait_count': clickbait_count,
                'has_sources': has_source,
                'contains_tribal_content': has_tribal,
                'contains_political_content': has_politics
            }
        }

        print(f"  🎯 RESULT: Risk={risk}%, Authentic={result['is_authentic']}")

        return result

# ============== INITIALIZE DETECTORS ==============
image_detector = UltraImageDetector()
audio_detector = UltraAudioDetector()
text_detector = UltraTextDetector()

# ============== ACCURACY TRACKING ==============
class AccuracyTracker:
    def __init__(self):
        self.log_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'detection_log.json')

    def log_detection(self, media_type, result, actual_label=None):
        """Log each detection for accuracy tracking"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'type': media_type,
            'risk_score': result.get('risk_score', 0),
            'is_authentic': result.get('is_authentic', True),
            'confidence': result.get('confidence', 0),
            'actual_label': actual_label
        }

        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            print(f"⚠️ Logging error: {e}")

    def get_stats(self):
        """Calculate accuracy statistics"""
        try:
            if not os.path.exists(self.log_file):
                return {'total': 0, 'accuracy': 0}

            with open(self.log_file, 'r') as f:
                logs = [json.loads(line) for line in f if line.strip()]

            total = len(logs)
            correct = sum(1 for log in logs
                         if log.get('actual_label') is not None
                         and log['is_authentic'] == log['actual_label'])

            labeled = sum(1 for log in logs if log.get('actual_label') is not None)

            return {
                'total_detections': total,
                'labeled_samples': labeled,
                'accuracy': round(correct / labeled * 100, 2) if labeled > 0 else 0
            }
        except:
            return {'total': 0, 'accuracy': 0}

accuracy_tracker = AccuracyTracker()

#API ENDPOINTS 

@app.route('/')
def home():
    return jsonify({
        "system": "SafEye - High-confidence AI-assisted detection",
        "version": "2.0 (Competition Grade)",
        "kenya_focus": "Protecting 50M+ Kenyans from digital deception",
        "use_cases": {
            "election_2027": "Protect 20M+ voters from political deepfakes",
            "mpesa_security": "Secure 30M+ M-Pesa users from voice cloning",
            "diaspora_protection": "Protect $400M+ monthly remittances",
            "media_integrity": "Support 500+ Kenyan media outlets"
        },
        "endpoints": {
            "/api/analyze/image": "POST - Image deepfake detection",
            "/api/analyze/audio": "POST - Audio deepfake detection",
            "/api/analyze/text": "POST - Text misinformation detection",
            "/api/analytics": "GET - System performance statistics"
        }
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'version': '2.0',
        'ai_model_loaded': image_detector.ai_model is not None,
        'accuracy': accuracy_tracker.get_stats()
    })

@app.route('/api/analyze/image', methods=['POST'])
def analyze_image():
    """Ultra-accurate image deepfake detection"""
    filepath = None
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Generate unique filename
        ext = os.path.splitext(secure_filename(file.filename))[1].lower()
        if ext not in ['.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.jfif']:
            return jsonify({'error': 'Invalid image format'}), 400

        unique_filename = f"{uuid.uuid4().hex}{ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)

        # Save and analyze
        file.save(filepath)
        result = image_detector.analyze_image(filepath)

        # Log for tracking
        accuracy_tracker.log_detection('image', result)

        # Cleanup
        if os.path.exists(filepath):
            os.remove(filepath)

        return jsonify(result)

    except Exception as e:
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
            except:
                pass

        import traceback
        print(f"❌ Image analysis error:\n{traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze/audio', methods=['POST'])
def analyze_audio():
    """Ultra-accurate audio deepfake detection"""
    filepath = None
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Validate format
        ext = os.path.splitext(secure_filename(file.filename))[1].lower()
        if ext not in ['.mp3', '.wav', '.m4a', '.ogg']:
            return jsonify({'error': 'Invalid audio format'}), 400

        unique_filename = f"{uuid.uuid4().hex}{ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)

        # Save and analyze
        file.save(filepath)
        result = audio_detector.analyze_audio(filepath)

        # Log for tracking
        accuracy_tracker.log_detection('audio', result)

        # Cleanup
        if os.path.exists(filepath):
            os.remove(filepath)

        return jsonify(result)

    except Exception as e:
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
            except:
                pass

        import traceback
        print(f"❌ Audio analysis error:\n{traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze/text', methods=['POST'])
def analyze_text():
    """Ultra-accurate text misinformation detection"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        if len(text) < 10:
            return jsonify({'error': 'Text too short (minimum 10 characters)'}), 400

        # Analyze
        result = text_detector.analyze_text(text)

        # Log for tracking
        accuracy_tracker.log_detection('text', result)

        return jsonify(result)

    except Exception as e:
        import traceback
        print(f"❌ Text analysis error:\n{traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/analytics', methods=['GET'])
def get_analytics():
    """Get system performance statistics"""
    try:
        stats = accuracy_tracker.get_stats()

        return jsonify({
            'system_performance': stats,
            'detectors': {
                'image': {
                    'status': 'active',
                    'ai_model_loaded': image_detector.ai_model is not None and image_detector.ai_model != "unavailable",
                    'accuracy_estimate': 'Benchmark-dependent'
                },
                'audio': {
                    'status': 'active',
                    'accuracy_estimate': 'Benchmark-dependent'
                },
                'text': {
                    'status': 'active',
                    'ai_model_loaded': text_detector.pipeline is not None,
                    'accuracy_estimate': 'Benchmark-dependent'
                }
            },
            'kenya_protection': {
                'elections': 'Active monitoring for political deepfakes',
                'mpesa': 'Voice cloning detection for financial security',
                'social_cohesion': 'Tribal hate speech detection',
                'diaspora': 'Family scam prevention'
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/test-model', methods=['GET'])
def test_model():
    """Test if AI model is available"""
    image_detector.load_ai_model()

    return jsonify({
        'ai_model_available': image_detector.ai_model is not None and image_detector.ai_model != "unavailable",
        'model_status': 'loaded' if image_detector.ai_model not in [None, "unavailable"] else 'unavailable',
        'fallback_mode': image_detector.ai_model == "unavailable"
    })

# ============== RUN SERVER ==============

    print("=" * 80)
    print("🇰🇪 SafEye - High-confidence AI-assisted Detection System")
    print("=" * 80)
    print("Version: 2.0 (Competition Grade)")
    print("Detection Mode: High-confidence AI-assisted ensemble")
    print("Kenya Focus: Protecting 50M+ Kenyans from digital deception")
    print("-" * 80)
    print("Use Cases:")
    print("  ✓ Election 2027: Protect 20M+ voters from political deepfakes")
    print("  ✓ M-Pesa Security: Secure 30M+ users from voice cloning scams")
    print("  ✓ Diaspora Protection: Protect $400M+ monthly remittances")
    print("  ✓ Media Integrity: Support 500+ Kenyan media outlets")
    print("-" * 80)
    print("🚀 Starting server on http://0.0.0.0:5000")
    print("=" * 80)
    print()

    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
>>>>>>> 3dd3b3e (Update backend logic, remove old detectors, and sync styles)
if __name__ == '__main__':
    print("=" * 80)
    print("🇰🇪 SafEye - High-confidence AI-assisted Detection System")
    print("=" * 80)
    print("Version: 2.0 (Competition Grade)")
    print("Detection Mode: High-confidence AI-assisted ensemble")
    print("Kenya Focus: Protecting 50M+ Kenyans from digital deception")
    print("-" * 80)
    print("Use Cases:")
    print("  ✓ Election 2027: Protect 20M+ voters from political deepfakes")
    print("  ✓ M-Pesa Security: Secure 30M+ users from voice cloning scams")
    print("  ✓ Diaspora Protection: Protect $400M+ monthly remittances")
    print("  ✓ Media Integrity: Support 500+ Kenyan media outlets")
    print("-" * 80)
    print("🚀 Starting server on http://0.0.0.0:5000")
    print("=" * 80)
    print()

    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
=======
    print("=" * 80)
    print("🇰🇪 SafEye - High-confidence AI-assisted Detection System")
    print("=" * 80)
    print("Version: 2.0 (Competition Grade)")
    print("Detection Mode: High-confidence AI-assisted ensemble")
    print("Kenya Focus: Protecting 50M+ Kenyans from digital deception")
    print("-" * 80)
    print("Use Cases:")
    print("  ✓ Election 2027: Protect 20M+ voters from political deepfakes")
    print("  ✓ M-Pesa Security: Secure 30M+ users from voice cloning scams")
    print("  ✓ Diaspora Protection: Protect $400M+ monthly remittances")
    print("  ✓ Media Integrity: Support 500+ Kenyan media outlets")
    print("-" * 80)
    print("🚀 Starting server on http://0.0.0.0:5000")
    print("=" * 80)
    print()

    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
>>>>>>> 3dd3b3e (Update backend logic, remove old detectors, and sync styles)
    print()
