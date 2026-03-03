# Security Policy — SafEye

## Supported Versions

| Version | Supported |
|---------|-----------|
| 3.2.x   | ✅ Active  |
| < 3.2   | ❌ EOL     |

## Reporting a Vulnerability

If you discover a security vulnerability in SafEye, please report it responsibly:

1. **Do NOT** open a public GitHub issue.
2. Email the team at **security@safeye.ke** (or contact via GitHub private vulnerability reporting).
3. Include:
   - A description of the vulnerability
   - Steps to reproduce
   - Potential impact
4. We will acknowledge within **48 hours** and aim to patch critical issues within **7 days**.

## Security Architecture

### Authentication & Sessions
- Passwords hashed with `werkzeug.security` (PBKDF2-SHA256, 260,000 iterations)
- JWT tokens for API authentication (`flask-jwt-extended`) with 24-hour expiry
- Session cookies: `HttpOnly`, `SameSite=Lax`, `Secure` in production
- CSRF protection via `Flask-WTF` on all form endpoints
- OAuth 2.0 via Authlib (Google, GitHub)

### Input Validation
- File uploads validated by extension whitelist and MIME type
- Maximum upload size enforced (50 MB default)
- All filenames sanitised with `werkzeug.utils.secure_filename`
- JSON request bodies validated before processing
- SQL injection mitigated by parameterised queries throughout

### Rate Limiting
- In-memory rate limiter with configurable thresholds
- Default: 200 requests/hour per IP
- Analysis endpoints: 30 requests/minute per IP
- Automatic cleanup of expired rate-limit entries

### Security Headers
All responses include:
- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `X-XSS-Protection: 1; mode=block`
- `Referrer-Policy: strict-origin-when-cross-origin`
- `Permissions-Policy: camera=(), microphone=(), geolocation=()`
- `Content-Security-Policy` (restrictive default-src)
- `Strict-Transport-Security` (when served over HTTPS)
- Server version header stripped from all responses

### Infrastructure
- Docker: non-root user (`safeye`), minimal base image (`python:3.11-slim`)
- Gunicorn: worker recycling (max 1,000 requests), proper signal handling via `tini`
- SQLite: WAL mode, foreign keys enforced, connection pooling
- No secrets in source code — all sensitive values via environment variables

### Dependency Security
- CPU-only PyTorch to minimise attack surface
- Multi-stage Docker build to exclude build tools from runtime image
- `.dockerignore` excludes secrets, tests, and development files

## Applicable Kenyan Law

SafEye is built to operate within Kenya's legal framework:
- **Computer Misuse and Cybercrimes Act, 2018** — criminalises publication of false information
- **NCIC Act** — prohibits ethnic incitement and hate speech
- **Elections Act** — regulates campaign material and media integrity
- **Data Protection Act, 2019** — governs processing of personal data

## Responsible Disclosure Timeline

| Day | Action |
|-----|--------|
| 0   | Vulnerability reported |
| 1-2 | Acknowledgement sent |
| 3-5 | Triage and severity assessment |
| 7   | Patch for critical/high severity |
| 14  | Patch for medium severity |
| 30  | Patch for low severity |
| 90  | Public disclosure (coordinated) |
