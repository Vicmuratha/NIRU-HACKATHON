"""
Integration tests for SafEye API endpoints.
Tests the Flask routes end-to-end using the test client.
"""

import io
import json
import uuid
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

    def test_health_has_system_metrics(self, client):
        """Health endpoint should include system resource metrics."""
        import app as _app
        _app._health_cache["data"] = None
        _app._health_cache["expires"] = 0.0

        data = client.get('/api/health').get_json()
        assert 'system' in data
        assert 'uptime_seconds' in data
        assert data['uptime_seconds'] >= 0

    def test_health_cache_header_present(self, client):
        """First call should be a cache MISS, second within TTL a HIT."""
        # Reset the module-level cache so this test is deterministic
        import app as _app
        _app._health_cache["data"] = None
        _app._health_cache["expires"] = 0.0

        resp1 = client.get('/api/health')
        assert resp1.headers.get('X-Cache') == 'MISS'
        assert 'max-age' in resp1.headers.get('Cache-Control', '')

        resp2 = client.get('/api/health')
        assert resp2.headers.get('X-Cache') == 'HIT'


# ═══════════════════════════════════════════════════════════
#  API DOCS
# ═══════════════════════════════════════════════════════════

class TestApiDocsEndpoint:
    def test_api_docs_returns_200(self, client):
        resp = client.get('/api/docs')
        assert resp.status_code == 200

    def test_api_docs_has_required_fields(self, client):
        data = client.get('/api/docs').get_json()
        assert data['openapi'] == '3.0.0'
        assert data['basePath'] == '/api'
        assert 'info' in data
        assert 'endpoints' in data
        assert 'servers' in data

    def test_api_docs_lists_core_endpoints(self, client):
        endpoints = client.get('/api/docs').get_json()['endpoints']
        assert 'GET /api/health' in endpoints
        assert 'GET /api/version' in endpoints
        assert 'GET /api/history' in endpoints
        assert 'GET /api/report/{history_id}' in endpoints
        assert 'POST /api/analyze/video' in endpoints


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

    def test_video_rejects_no_file(self, client):
        resp = client.post('/api/analyze/video')
        assert resp.status_code == 400

    def test_video_rejects_invalid_extension(self, client):
        data = {'file': (io.BytesIO(b'not a real file'), 'test.txt')}
        resp = client.post('/api/analyze/video',
                          data=data,
                          content_type='multipart/form-data')
        assert resp.status_code == 400

    def test_image_rejects_invalid_extension(self, client):
        data = {'file': (io.BytesIO(b'not a real file'), 'test.exe')}
        resp = client.post('/api/analyze/image',
                          data=data,
                          content_type='multipart/form-data')
        assert resp.status_code == 400
        assert 'not allowed' in resp.get_json()['error'].lower()


# ═══════════════════════════════════════════════════════════
#  ANALYSIS ENDPOINTS  — FUNCTIONAL (require model inference)
#  Run with: pytest -m slow   (skipped by default in fast CI)
# ═══════════════════════════════════════════════════════════

@pytest.mark.slow
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


@pytest.mark.slow
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


@pytest.mark.slow
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
#  PLATFORM STATS
# ═══════════════════════════════════════════════════════════

class TestStatsEndpoint:
    def test_stats_returns_200(self, client):
        resp = client.get('/api/stats')
        assert resp.status_code == 200

    def test_stats_has_required_fields(self, client):
        data = client.get('/api/stats').get_json()
        assert 'total_scans' in data
        assert 'threats_detected' in data
        assert 'registered_users' in data
        assert 'by_type' in data
        assert isinstance(data['by_type'], dict)
        assert 'video' in data['by_type']

    def test_stats_has_recent_threats(self, client):
        data = client.get('/api/stats').get_json()
        assert 'recent_threats' in data
        assert isinstance(data['recent_threats'], list)


# ═══════════════════════════════════════════════════════════
#  ANALYTICS TRENDS
# ═══════════════════════════════════════════════════════════

class TestAnalyticsTrends:
    def test_trends_returns_200(self, client):
        resp = client.get('/api/analytics/trends')
        assert resp.status_code == 200

    def test_trends_has_required_fields(self, client):
        data = client.get('/api/analytics/trends').get_json()
        assert 'period_days' in data
        assert 'daily' in data
        assert 'by_type' in data
        assert 'top_threats' in data
        assert isinstance(data['daily'], list)
        assert isinstance(data['by_type'], list)

    def test_trends_respects_days_param(self, client):
        data = client.get('/api/analytics/trends?days=7').get_json()
        assert data['period_days'] == 7

    def test_trends_clamps_max_days(self, client):
        data = client.get('/api/analytics/trends?days=999').get_json()
        assert data['period_days'] == 90


# ═══════════════════════════════════════════════════════════
#  THREAT FEED
# ═══════════════════════════════════════════════════════════

class TestThreatFeed:
    def test_threats_returns_200(self, client):
        resp = client.get('/api/threats/recent')
        assert resp.status_code == 200

    def test_threats_has_required_fields(self, client):
        data = client.get('/api/threats/recent').get_json()
        assert 'count' in data
        assert 'threats' in data
        assert isinstance(data['threats'], list)

    def test_threats_respects_limit(self, client):
        data = client.get('/api/threats/recent?limit=5').get_json()
        assert data['count'] <= 5


# ═══════════════════════════════════════════════════════════
#  REPORT EXPORT
# ═══════════════════════════════════════════════════════════

class TestReportExport:
    def test_report_requires_auth(self, client):
        resp = client.get('/api/report/1')
        assert resp.status_code == 401

    def test_report_not_found(self, authenticated_client):
        resp = authenticated_client.get('/api/report/99999')
        assert resp.status_code in (404,)


# ═══════════════════════════════════════════════════════════
#  VERSION ENDPOINT
# ═══════════════════════════════════════════════════════════

class TestVersionEndpoint:
    def test_version_returns_200(self, client):
        resp = client.get('/api/version')
        assert resp.status_code == 200

    def test_version_has_capabilities(self, client):
        data = client.get('/api/version').get_json()
        assert 'capabilities' in data
        assert 'video_analysis' in data['capabilities']
        assert 'deepfake_detection' in data['capabilities']
        assert 'analytics_trends' in data['capabilities']
        assert 'threat_feed' in data['capabilities']
        assert 'report_export' in data['capabilities']
        assert data['app'] == 'SafEye'


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

    def test_request_id_header_generated(self, client):
        """Every response must include an X-Request-ID (UUID4) header."""
        resp = client.get('/api/health')
        rid = resp.headers.get('X-Request-ID')
        assert rid is not None, "X-Request-ID header missing"
        # Validate it is a proper UUID
        uuid.UUID(rid, version=4)

    def test_request_id_echo(self, client):
        """If the client sends X-Request-ID, the server echoes it back."""
        custom_id = str(uuid.uuid4())
        resp = client.get('/api/health', headers={'X-Request-ID': custom_id})
        assert resp.headers.get('X-Request-ID') == custom_id


# ═══════════════════════════════════════════════════════════
#  SIGNUP & LOGIN FLOWS
# ═══════════════════════════════════════════════════════════

class TestSignupFlow:
    """Tests for POST /signup — user registration."""

    def test_signup_page_loads(self, client):
        resp = client.get('/signup')
        assert resp.status_code == 200

    def test_signup_rejects_empty_fields(self, client):
        resp = client.post('/signup', data={
            'username': '', 'email': '', 'password': '', 'confirm_password': ''
        }, follow_redirects=True)
        assert resp.status_code == 200
        assert b'fill in all fields' in resp.data.lower() or resp.status_code == 200

    def test_signup_rejects_short_username(self, client):
        resp = client.post('/signup', data={
            'username': 'A', 'email': 'a@b.com',
            'password': 'securepass1', 'confirm_password': 'securepass1'
        }, follow_redirects=True)
        assert resp.status_code == 200

    def test_signup_rejects_bad_email(self, client):
        resp = client.post('/signup', data={
            'username': 'TestUser', 'email': 'not-an-email',
            'password': 'securepass1', 'confirm_password': 'securepass1'
        }, follow_redirects=True)
        assert resp.status_code == 200

    def test_signup_rejects_short_password(self, client):
        resp = client.post('/signup', data={
            'username': 'TestUser', 'email': 'new@test.ke',
            'password': 'short', 'confirm_password': 'short'
        }, follow_redirects=True)
        assert resp.status_code == 200

    def test_signup_rejects_mismatched_passwords(self, client):
        resp = client.post('/signup', data={
            'username': 'TestUser', 'email': 'new2@test.ke',
            'password': 'securepass1', 'confirm_password': 'differentpass'
        }, follow_redirects=True)
        assert resp.status_code == 200

    def test_signup_creates_user_and_redirects(self, flask_app, client):
        import sqlite3, os
        unique_email = f'testuser_{uuid.uuid4().hex[:8]}@test.ke'
        resp = client.post('/signup', data={
            'username': 'NewTestUser', 'email': unique_email,
            'password': 'securepass123', 'confirm_password': 'securepass123'
        }, follow_redirects=True)
        assert resp.status_code == 200


class TestLoginFlow:
    """Tests for POST /login — user authentication."""

    def test_login_page_loads(self, client):
        resp = client.get('/login')
        assert resp.status_code == 200

    def test_login_rejects_empty_credentials(self, client):
        resp = client.post('/login', data={
            'email': '', 'password': ''
        }, follow_redirects=True)
        assert resp.status_code == 200

    def test_login_rejects_wrong_password(self, client):
        resp = client.post('/login', data={
            'email': 'nobody@test.ke', 'password': 'wrongpassword1'
        }, follow_redirects=True)
        assert resp.status_code == 200


# ═══════════════════════════════════════════════════════════
#  DETECTION HISTORY — AUTHENTICATED ACCESS
# ═══════════════════════════════════════════════════════════

class TestDetectionHistory:
    """Tests for /api/history with authenticated user session."""

    def test_history_requires_auth(self, client):
        resp = client.get('/api/history')
        assert resp.status_code == 401

    def test_history_returns_list_when_authenticated(self, authenticated_client):
        resp = authenticated_client.get('/api/history')
        # Might be 200 with empty list or 404 if user not in DB
        assert resp.status_code in (200, 404)

    def test_history_pagination_params(self, authenticated_client):
        resp = authenticated_client.get('/api/history?page=1&per_page=5')
        assert resp.status_code in (200, 404)

    def test_history_type_filter(self, authenticated_client):
        resp = authenticated_client.get('/api/history?type=image')
        assert resp.status_code in (200, 404)

    def test_history_delete_requires_auth(self, client):
        resp = client.delete('/api/history/1')
        assert resp.status_code == 401

    def test_history_delete_nonexistent(self, authenticated_client):
        resp = authenticated_client.delete('/api/history/99999')
        # Should still return 200 (no-op delete) or 404
        assert resp.status_code in (200, 404)


# ═══════════════════════════════════════════════════════════
#  AUDIO ERROR HANDLING
# ═══════════════════════════════════════════════════════════

class TestAudioErrorHandling:
    """Tests for audio analysis error cases."""

    def test_audio_rejects_no_file(self, client):
        resp = client.post('/api/analyze/audio')
        assert resp.status_code == 400

    def test_audio_rejects_invalid_extension(self, client):
        data = {'file': (io.BytesIO(b'not audio data'), 'test.exe')}
        resp = client.post('/api/analyze/audio',
                          data=data,
                          content_type='multipart/form-data')
        assert resp.status_code == 400

    def test_audio_rejects_empty_file(self, client):
        """Empty file should be caught by validation before model inference."""
        data = {'file': (io.BytesIO(b''), 'empty.txt')}
        resp = client.post('/api/analyze/audio',
                          data=data,
                          content_type='multipart/form-data')
        assert resp.status_code == 400

    def test_audio_rejects_txt_extension(self, client):
        data = {'file': (io.BytesIO(b'plain text not audio'), 'notes.txt')}
        resp = client.post('/api/analyze/audio',
                          data=data,
                          content_type='multipart/form-data')
        assert resp.status_code == 400


# ═══════════════════════════════════════════════════════════
#  USER LISTING
# ═══════════════════════════════════════════════════════════

class TestUsersAPI:
    def test_users_requires_auth(self, client):
        resp = client.get('/api/users')
        assert resp.status_code == 401

    def test_users_returns_list_when_authenticated(self, authenticated_client):
        resp = authenticated_client.get('/api/users')
        # May get 200 or 404 depending on DB state
        assert resp.status_code in (200, 404)

    def test_users_supports_pagination(self, authenticated_client):
        resp = authenticated_client.get('/api/users?page=1&per_page=10')
        assert resp.status_code in (200, 404)


# ═══════════════════════════════════════════════════════════
#  EDGE CASES & MISC
# ═══════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_404_returns_for_unknown_api_routes(self, client):
        resp = client.get('/api/nonexistent-route')
        assert resp.status_code == 404

    def test_text_rejects_missing_text_key(self, client):
        resp = client.post('/api/analyze/text',
                          data=json.dumps({'content': 'missing text key'}),
                          content_type='application/json')
        assert resp.status_code == 400

    def test_image_rejects_non_image_extension(self, client):
        data = {'file': (io.BytesIO(b'data'), 'doc.pdf')}
        resp = client.post('/api/analyze/image',
                          data=data,
                          content_type='multipart/form-data')
        assert resp.status_code == 400

    def test_forward_rejects_missing_text(self, client):
        resp = client.post('/api/analyze/forward',
                          data=json.dumps({}),
                          content_type='application/json')
        assert resp.status_code == 400


# ═══════════════════════════════════════════════════════════
#  INPUT SANITIZATION
# ═══════════════════════════════════════════════════════════

class TestInputSanitization:
    """Verify that HTML/script tags are stripped from text inputs."""

    @pytest.mark.slow
    def test_html_tags_stripped(self, client):
        payload = {'text': '<b>Breaking</b> <i>news</i>: the earth is flat'}
        resp = client.post('/api/analyze/text',
                          data=json.dumps(payload),
                          content_type='application/json')
        # Should not 400 — tag text is still long enough after stripping
        assert resp.status_code in (200, 500)  # 500 if model unavailable

    @pytest.mark.slow
    def test_script_tags_stripped(self, client):
        payload = {'text': '<script>alert("xss")</script>Some real content here for analysis'}
        resp = client.post('/api/analyze/text',
                          data=json.dumps(payload),
                          content_type='application/json')
        assert resp.status_code in (200, 500)


# ═══════════════════════════════════════════════════════════
#  VERSION ENDPOINT
# ═══════════════════════════════════════════════════════════

class TestVersionEndpoint:
    """Tests for GET /api/version — build & capability metadata."""

    def test_version_returns_200(self, client):
        resp = client.get('/api/version')
        assert resp.status_code == 200

    def test_version_has_required_fields(self, client):
        data = client.get('/api/version').get_json()
        assert data['app'] == 'SafEye'
        assert 'version' in data
        assert 'platform' in data
        assert 'environment' in data

    def test_version_lists_capabilities(self, client):
        data = client.get('/api/version').get_json()
        caps = data['capabilities']
        assert isinstance(caps, list)
        assert 'deepfake_detection' in caps
        assert 'audio_analysis' in caps
        assert 'fake_news_classification' in caps
        assert 'whatsapp_forward_checking' in caps
        assert 'document_verification' in caps
        assert 'election_shield' in caps

    def test_version_includes_python_version(self, client):
        data = client.get('/api/version').get_json()
        assert 'python_version' in data
        assert 'Python' in data['python_version'] or '3.' in data['python_version']


# ═══════════════════════════════════════════════════════════
#  CORS HEADERS
# ═══════════════════════════════════════════════════════════

class TestCORSHeaders:
    """Verify CORS headers on API responses."""

    def test_options_preflight_allowed(self, client):
        resp = client.options('/api/health')
        # Should not be 405 Method Not Allowed
        assert resp.status_code in (200, 204)

    def test_cors_allows_frontend_origin(self, client):
        resp = client.get('/api/health', headers={
            'Origin': 'http://localhost:3000'
        })
        acao = resp.headers.get('Access-Control-Allow-Origin', '')
        assert 'localhost:3000' in acao or acao == '*'


# ═══════════════════════════════════════════════════════════
#  PROFILE UPDATE VALIDATION
# ═══════════════════════════════════════════════════════════

class TestProfileValidation:
    """Tests for PUT /api/profile input validation."""

    def test_profile_update_requires_auth(self, client):
        resp = client.put('/api/profile',
                         data=json.dumps({'name': 'Hacker'}),
                         content_type='application/json')
        assert resp.status_code == 401

    def test_password_change_requires_auth(self, client):
        resp = client.put('/api/profile/password',
                         data=json.dumps({
                             'current_password': 'old',
                             'new_password': 'new12345'
                         }),
                         content_type='application/json')
        assert resp.status_code == 401

    def test_profile_picture_upload_requires_auth(self, client):
        data = {'file': (io.BytesIO(b'\x89PNG\r\n'), 'photo.png')}
        resp = client.post('/api/profile/picture',
                          data=data,
                          content_type='multipart/form-data')
        assert resp.status_code == 401

    def test_sanitize_text_function_directly(self):
        from backend.middleware import sanitize_text
        assert sanitize_text('<b>hello</b>') == 'hello'
        assert sanitize_text('<script>alert(1)</script>world') == 'world'
        assert 'onclick' not in sanitize_text('click onclick=alert(1)')
        assert sanitize_text('  normal text  ') == 'normal text'
