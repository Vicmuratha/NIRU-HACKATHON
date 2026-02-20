# Contributing to SafEye

Thank you for your interest in contributing to SafEye — Kenya's AI-powered deepfake and misinformation detection platform.

## Getting Started

### Prerequisites

- Python 3.11+
- Node.js 18+ (for the React frontend)
- Git

### Local Setup

```bash
# Clone the repo
git clone https://github.com/Vicmuratha/NIRU-HACKATHON.git
cd NIRU-HACKATHON

# Create a virtual environment
python -m venv venv
source venv/bin/activate   # Linux/macOS
# venv\Scripts\activate    # Windows

# Install Python dependencies
pip install -r requirements.txt

# Install frontend dependencies
npm install

# Start the backend
python app.py

# In another terminal, build the frontend
npm run build
```

### Environment Variables

Copy `.env.example` to `.env` and configure:

| Variable | Description | Default |
|----------|-------------|---------|
| `JWT_SECRET_KEY` | Secret for JWT token signing | `safeye-hackathon-secret-2026` |
| `FLASK_SECRET_KEY` | Flask session secret | `super_secret_key` |
| `DOWNLOAD_MODELS_ON_STARTUP` | Auto-download AI models | `false` |

## Development Workflow

### Branch Naming

Use descriptive prefixes:

| Prefix | Use For | Example |
|--------|---------|---------|
| `feature/` | New functionality | `feature/video-support` |
| `fix/` | Bug fixes | `fix/confidence-scoring-bug` |
| `refactor/` | Code restructuring | `refactor/remove-duplicate-code` |
| `test/` | Test additions/changes | `test/audio-edge-cases` |
| `docs/` | Documentation only | `docs/api-reference` |
| `ci/` | CI/CD pipeline changes | `ci/github-actions` |

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>: <short description>

<optional body explaining what and why>
```

Types: `feat`, `fix`, `refactor`, `test`, `docs`, `ci`, `chore`

### Pull Requests

1. Create a branch from `main`
2. Make focused, small commits
3. Push your branch and open a PR via GitHub
4. Write a clear description: what changed, why, and what it resolves
5. Reference issues where applicable (e.g., `Closes #5`)

## Project Structure

```
├── app.py                  # Unified Flask backend (v3.1)
├── backend/
│   ├── app.py              # Detection API server (port 7860)
│   ├── middleware.py        # Security headers, rate limiting, validation
│   ├── election_shield.py   # Kenya election context analysis
│   ├── whatsapp_checker.py  # WhatsApp forward pattern detection
│   ├── kenya_documents.py   # Kenyan document forgery detection
│   ├── audio_context.py     # Audio Kenya-specific context
│   └── fake_screenshot.py   # News screenshot manipulation detection
├── models/                  # AI model weights (git-ignored, download on first run)
├── src/                     # React + TypeScript frontend
├── tests/                   # Pytest test suite
└── docs/                    # Documentation
```

## Running Tests

```bash
# Run fast tests only (no model loading)
pytest tests/ -m "not slow" -v

# Run all tests including model inference (slow)
pytest tests/ -v

# Run a specific test file
pytest tests/test_api.py -v
```

## Code Style

- Python: Follow PEP 8 (enforced by flake8 in CI)
- Max line length: 120 characters
- TypeScript: Standard ESLint/Prettier configuration

## Reporting Issues

When filing an issue, include:

1. **Description** — What happened vs. what you expected
2. **Steps to reproduce** — Minimal steps to trigger the bug
3. **Environment** — Python version, OS, browser (if frontend)
4. **Logs** — Relevant error output or screenshots

## Kenya-Specific Context

SafEye is designed for the Kenyan information ecosystem. When contributing detection logic, keep in mind:

- **WhatsApp** is the primary misinformation vector (67% of Kenyans get news via WhatsApp)
- **Election integrity** features should reference IEBC, NCIC, and DCI reporting channels
- **Swahili content** support is important for broad coverage
- **Document forgery** detection covers Kenyan IDs, KRA PINs, NHIF/NSSF cards, etc.
- **News screenshot manipulation** of outlets like Nation, Standard Digital, and Citizen TV is common

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.
