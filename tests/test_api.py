"""
Integration tests for SafEye API endpoints.
Tests the Flask routes end-to-end using the test client.
"""

import io
import json
import pytest


# ═══════════════════════════════════════════════════════════
#  HEALTH CHECK
# ═══════════════════════════════════════════════════════════

class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get('/api/health')
        assert resp.status_code == 200

    def test_health_has_required_fields(self, client):
        data = client.get('/api/health').get_json()
        assert data['status'] == 'healthy'
        assert 'version' in data
        assert 'models' in data
        assert 'platform' in data

    def test_health_reports_modules(self, client):
        data = client.get('/api/health').get_json()
        modules = data['modules']
        assert modules['auth'] is True
        assert modules['election_shield'] is True
        assert modules['whatsapp_checker'] is True


# ═══════════════════════════════════════════════════════════
#  AUTH API
# ═══════════════════════════════════════════════════════════

class TestAuthAPI:
    def test_me_returns_null_when_not_logged_in(self, client):
        resp = client.get('/api/me')
        assert resp.status_code == 200
        assert resp.get_json()['user'] is None

    def test_me_returns_user_when_authenticated(self, authenticated_client):
        resp = authenticated_client.get('/api/me')
        data = resp.get_json()
        assert data['user'] is not None
        assert data['user']['email'] == 'test@safeye.ke'

    def test_login_rejects_missing_fields(self, client):
        resp = client.post('/api/login',
                          data=json.dumps({}),
                          content_type='application/json')
        assert resp.status_code == 400


# ═══════════════════════════════════════════════════════════
#  ANALYSIS ENDPOINTS  — INPUT VALIDATION
# ═══════════════════════════════════════════════════════════

class TestAnalysisValidation:
    def test_image_rejects_no_file(self, client):
        resp = client.post('/api/analyze/image')
        assert resp.status_code == 400
        assert 'error' in resp.get_json()

    def test_audio_rejects_no_file(self, client):
        resp = client.post('/api/analyze/audio')
        assert resp.status_code == 400

    def test_text_rejects_empty_body(self, client):
        resp = client.post('/api/analyze/text',
                          data=json.dumps({'text': ''}),
                          content_type='application/json')
        assert resp.status_code == 400

    def test_text_rejects_no_json(self, client):
        resp = client.post('/api/analyze/text', data='not json')
        assert resp.status_code == 400

    def test_forward_rejects_short_text(self, client):
        resp = client.post('/api/analyze/forward',
                          data=json.dumps({'text': 'hi'}),
                          content_type='application/json')
        assert resp.status_code == 400

    def test_document_rejects_no_file(self, client):
        resp = client.post('/api/analyze/document')
        assert resp.status_code == 400

    def test_image_rejects_invalid_extension(self, client):
        data = {'file': (io.BytesIO(b'not a real file'), 'test.exe')}
        resp = client.post('/api/analyze/image',
                          data=data,
                          content_type='multipart/form-data')
        assert resp.status_code == 400
        assert 'not allowed' in resp.get_json()['error'].lower()


# ═══════════════════════════════════════════════════════════
#  ANALYSIS ENDPOINTS  — FUNCTIONAL
# ═══════════════════════════════════════════════════════════

class TestImageAnalysis:
    def test_image_analysis_returns_result(self, client, test_image_bytes):
        buf, name = test_image_bytes
        resp = client.post('/api/analyze/image',
                          data={'file': (buf, name)},
                          content_type='multipart/form-data')
        assert resp.status_code == 200
        data = resp.get_json()
        assert 'risk_score' in data
        assert 'verdict' in data
        assert data['verdict'] in ('LIKELY_DEEPFAKE', 'AUTHENTIC', 'REVIEW_REQUIRED')
        assert 0 <= data['risk_score'] <= 100

    def test_image_result_has_findings(self, client, test_image_bytes):
        buf, name = test_image_bytes
        data = client.post('/api/analyze/image',
                          data={'file': (buf, name)},
                          content_type='multipart/form-data').get_json()
        assert isinstance(data['findings'], list)
        assert isinstance(data.get('kenya_warnings', []), list)


class TestTextAnalysis:
    def test_text_analysis_returns_result(self, client):
        resp = client.post('/api/analyze/text',
                          data=json.dumps({'text': 'This is a normal news article about current events in Kenya.'}),
                          content_type='application/json')
        assert resp.status_code == 200
        data = resp.get_json()
        assert 'risk_score' in data
        assert 'verdict' in data
        assert 0 <= data['risk_score'] <= 100

    def test_clickbait_increases_risk(self, client):
        resp = client.post('/api/analyze/text',
                          data=json.dumps({'text': 'SHOCKING secret exposed! You will not believe this secret!'}),
                          content_type='application/json')
        data = resp.get_json()
        assert data['risk_score'] >= 20


class TestForwardAnalysis:
    def test_forward_analysis_returns_result(self, client):
        resp = client.post('/api/analyze/forward',
                          data=json.dumps({'text': 'URGENT! Forward this to all your contacts! The government has announced free money for everyone! Send to 10 people to claim!'}),
                          content_type='application/json')
        assert resp.status_code == 200
        data = resp.get_json()
        assert 'risk_score' in data
        assert 'forward_analysis' in data


# ═══════════════════════════════════════════════════════════
#  PROFILE ENDPOINTS (require auth)
# ═══════════════════════════════════════════════════════════

class TestProfileAPI:
    def test_profile_requires_auth(self, client):
        resp = client.get('/api/profile')
        assert resp.status_code == 401

    def test_history_requires_auth(self, client):
        resp = client.get('/api/history')
        assert resp.status_code == 401

    def test_users_requires_auth(self, client):
        resp = client.get('/api/users')
        assert resp.status_code == 401


# ═══════════════════════════════════════════════════════════
#  SECURITY HEADERS
# ═══════════════════════════════════════════════════════════

class TestSecurityHeaders:
    def test_security_headers_present(self, client):
        resp = client.get('/api/health')
        assert resp.headers.get('X-Content-Type-Options') == 'nosniff'
        assert resp.headers.get('X-Frame-Options') == 'DENY'
        assert resp.headers.get('X-XSS-Protection') == '1; mode=block'

    def test_no_server_header(self, client):
        resp = client.get('/api/health')
        # Server header should be removed by middleware
        assert 'Server' not in resp.headers or 'Flask' not in resp.headers.get('Server', '')
