"""
SafEye — Test Configuration & Shared Fixtures
"""

import os
import sys
import tempfile
import wave
import pytest
import numpy as np
from PIL import Image

# Ensure project root is on the path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Set test environment before importing app
os.environ['FLASK_ENV'] = 'testing'
os.environ['RATELIMIT_ENABLED'] = 'false'


# ═══════════════════════════════════════════════════════════
#  Fixtures: Flask App & Client
# ═══════════════════════════════════════════════════════════

@pytest.fixture(scope='session')
def flask_app():
    """Create the Flask app in testing mode."""
    from app import app, DB_PATH
    app.config['TESTING'] = True
    app.config['RATELIMIT_ENABLED'] = False
    yield app
    # Clean up test database after all tests
    if os.path.exists(DB_PATH) and 'test_' in os.path.basename(DB_PATH):
        os.remove(DB_PATH)


@pytest.fixture
def client(flask_app):
    """Flask test client for making HTTP requests."""
    with flask_app.test_client() as c:
        yield c


@pytest.fixture
def authenticated_client(flask_app, client):
    """Test client with a logged-in session."""
    with client.session_transaction() as sess:
        sess['user'] = {
            'name': 'Test User',
            'email': 'test@safeye.ke',
            'picture': None,
        }
    return client


# ═══════════════════════════════════════════════════════════
#  Fixtures: Detectors (expensive, session-scoped)
# ═══════════════════════════════════════════════════════════

@pytest.fixture(scope='session')
def image_detector():
    """Session-scoped image detector (model loading is expensive)."""
    from app import UltraImageDetector
    return UltraImageDetector()


@pytest.fixture(scope='session')
def audio_detector():
    """Session-scoped audio detector."""
    from app import UltraAudioDetector
    return UltraAudioDetector()


@pytest.fixture(scope='session')
def text_detector():
    """Session-scoped text detector."""
    from app import UltraTextDetector
    return UltraTextDetector()


# ═══════════════════════════════════════════════════════════
#  Fixtures: Test Files
# ═══════════════════════════════════════════════════════════

@pytest.fixture
def test_image():
    """Create a temporary test JPEG, yield its path, clean up after."""
    img = Image.new('RGB', (200, 200), color='red')
    fd, path = tempfile.mkstemp(suffix='.jpg')
    os.close(fd)
    img.save(path)
    yield path
    if os.path.exists(path):
        os.remove(path)


@pytest.fixture
def test_image_white():
    """Plain white image (should be low-risk)."""
    img = Image.new('RGB', (200, 200), color='white')
    fd, path = tempfile.mkstemp(suffix='.jpg')
    os.close(fd)
    img.save(path)
    yield path
    if os.path.exists(path):
        os.remove(path)


@pytest.fixture
def test_wav():
    """Create a 1-second 440Hz sine wave WAV file."""
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    signal = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    fd, path = tempfile.mkstemp(suffix='.wav')
    os.close(fd)
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(signal.tobytes())
    yield path
    if os.path.exists(path):
        os.remove(path)


@pytest.fixture
def test_image_bytes():
    """Return (BytesIO, filename) for upload testing."""
    import io
    img = Image.new('RGB', (100, 100), color='blue')
    buf = io.BytesIO()
    img.save(buf, 'JPEG')
    buf.seek(0)
    return buf, 'test.jpg'


@pytest.fixture
def test_wav_bytes():
    """Return (BytesIO, filename) for upload testing."""
    import io
    sr = 16000
    t = np.linspace(0, 0.5, sr // 2, endpoint=False)
    signal = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(signal.tobytes())
    buf.seek(0)
    return buf, 'test.wav'
