# Contributing to DeepGuard

Thank you for your interest in contributing to DeepGuard! This document explains how to contribute effectively.

---

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for everyone. Be kind, assume good faith, and focus on constructive feedback.

---

## Ways to Contribute

- 🐛 **Report bugs** via GitHub Issues
- 💡 **Suggest features** via GitHub Discussions
- 📖 **Improve documentation**
- 🧪 **Write or improve tests**
- 🔧 **Fix bugs or implement features**
- 🌍 **Translate the dashboard** to other languages

---

## Getting Started

### 1. Fork & Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR-USERNAME/deepguard.git
cd deepguard
git remote add upstream https://github.com/your-org/deepguard.git
```

### 2. Set Up Development Environment

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS

pip install -r requirements.txt
pip install pre-commit
pre-commit install
pre-commit install --hook-type commit-msg

cp .env.example .env
```

### 3. Create a Feature Branch

```bash
# Sync with upstream first
git fetch upstream
git checkout main
git merge upstream/main

# Create your branch
git checkout -b feat/my-amazing-feature
# or
git checkout -b fix/bug-description
```

---

## Development Workflow

### Making Changes

1. Write your code following the code style guidelines below
2. Add or update tests for your changes
3. Ensure all tests pass: `pytest tests/ -v`
4. Update documentation if needed

### Commit Conventions

We use **Conventional Commits** format:

```
<type>(scope): <short description>

[optional body]

[optional footer]
```

**Types:**

| Type | Usage |
|---|---|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `test` | Adding or fixing tests |
| `refactor` | Code refactoring (no feature/fix) |
| `perf` | Performance improvement |
| `ci` | CI/CD configuration changes |
| `chore` | Maintenance tasks |

**Examples:**
```
feat(api): add batch detection endpoint
fix(service): handle empty face list gracefully
docs(api): update authentication section
test(detection): add ONNX inference route test
perf(inference): reduce ONNX session initialization overhead
```

### Pull Request Process

1. **Push your branch**: `git push origin feat/my-amazing-feature`
2. **Open a PR** on GitHub against the `main` branch
3. **Fill in the PR template** (linked automatically)
4. **Ensure CI passes**: All GitHub Actions must be green
5. **Request review** from at least one maintainer
6. **Address review feedback** by pushing additional commits
7. **Squash and merge** when approved

---

## Code Style

### Python

We use **black** for formatting and **ruff** for linting:

```bash
# Format code
black .
ruff check . --fix

# Check without modifying
black --check .
ruff check .
```

**Key style rules:**
- Line length: 100 characters
- Type hints required for all function parameters and return values
- Docstrings required for all public classes and functions (Google style)
- No bare `except:` clauses — always specify exception type
- Use `async`/`await` for all I/O operations

**Docstring format:**
```python
def detect_image(self, file_bytes: bytes, filename: str) -> DetectionResultDB:
    """Detect deepfake in an image from raw bytes.

    Args:
        file_bytes: Raw image file content.
        filename: Original filename for record keeping.

    Returns:
        DetectionResultDB with label, confidence, and XAI data.

    Raises:
        DeepGuardBaseException: If image cannot be decoded.
    """
```

### JavaScript (Frontend)

- ES6+ modules with `import`/`export`
- 2-space indentation
- Single quotes for strings
- `async`/`await` preferred over `.then()` chains
- JSDoc comments for exported functions

### YAML

- 2-space indentation
- No trailing whitespace
- Comments for non-obvious values

---

## Testing Requirements

**All pull requests must:**

1. Maintain or increase code coverage (threshold: 55%)
2. Pass all 187 existing tests
3. Include new tests for any new functionality
4. Include regression tests for any bug fixes

### Writing Tests

```python
# tests/unit/test_my_feature.py
import pytest
from unittest.mock import patch, MagicMock

class TestMyFeature:
    """Tests for my new feature."""

    @pytest.mark.asyncio
    async def test_happy_path(self, db_session) -> None:
        """Test normal successful operation."""
        # Arrange
        service = MyService(db_session)
        
        # Act
        result = await service.do_something("input")
        
        # Assert
        assert result.status == "success"
        assert result.value == "expected"

    @pytest.mark.asyncio
    async def test_error_case(self, db_session) -> None:
        """Test graceful error handling."""
        service = MyService(db_session)
        
        result = await service.do_something("")  # Invalid input
        
        assert result.status == "failed"
        assert result.error_message is not None
```

---

## Issue Reporting

### Bug Reports

Please include:
1. **Description**: Clear description of the bug
2. **Steps to reproduce**: Numbered list of steps
3. **Expected behavior**: What you expected to happen
4. **Actual behavior**: What actually happened
5. **Environment**: OS, Python version, PyTorch version
6. **Logs**: Relevant error messages or stack traces

### Feature Requests

Please include:
1. **Problem statement**: What problem are you trying to solve?
2. **Proposed solution**: How you'd like it to work
3. **Alternatives considered**: Other approaches you've thought of
4. **Additional context**: Screenshots, mockups, references

---

## Review Criteria

Pull requests are reviewed for:

| Criterion | Description |
|---|---|
| **Correctness** | Does the code work as intended? |
| **Tests** | Are there adequate tests? Do they pass? |
| **Code quality** | Is the code readable and maintainable? |
| **Documentation** | Are public APIs documented? Is README updated? |
| **Performance** | Does this introduce any regressions? |
| **Security** | Are there any security concerns? |
| **Compatibility** | Does it break any existing behavior? |

---

## License

By contributing to DeepGuard, you agree that your contributions will be licensed under the Apache License 2.0. See [LICENSE](../LICENSE) for details.

---

## Getting Help

- 📖 Read the [Developer Manual](DEVELOPER_MANUAL.md)
- 💬 Open a GitHub Discussion for questions
- 🐛 Open a GitHub Issue for bugs
- 📧 Email: deepguard@example.com

Thank you for contributing! 🎉
