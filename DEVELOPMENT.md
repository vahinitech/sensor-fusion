# Development Guide

## Code Quality Standards

This project enforces high code quality standards through automated checks.

### Black Code Formatting

All Python code must be formatted with Black:

```bash
# Check formatting
black --check src/ examples/ *.py

# Auto-format code
black src/ examples/ *.py
```

**Configuration**: Line length is 120 characters (see `pyproject.toml`)

### Pylint Code Analysis

Code must achieve a Pylint score of **9.5 or higher**:

```bash
# Run Pylint
pylint src/ --fail-under=9.5

# Run with detailed report
pylint src/ --fail-under=9.5 --output-format=colorized --score=yes
```

**Configuration**: See `.pylintrc` for disabled rules and settings

### Running All Quality Checks

Use the provided script to run all checks at once:

```bash
./run_quality_checks.sh
```

This will run:

1. Black formatting check
2. Pylint analysis (score >= 9.5)
3. Complete test suite with coverage

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_core.py -v

# Run specific test
pytest tests/test_core.py::TestSensorParser::test_parse_valid_line -v
```

### Writing Tests

- Place tests in `tests/` directory
- Follow naming convention: `test_*.py`
- Use descriptive test names
- Add docstrings to test methods
- Aim for >80% code coverage

Example:

```python
class TestMyFeature:
    """Test cases for my feature"""
    
    def test_basic_functionality(self):
        """Test that basic functionality works"""
        result = my_feature.process()
        assert result == expected
```

## CI/CD Pipeline

### GitHub Actions Workflows

Three workflows run automatically on push/PR:

#### 1. Code Quality (`code-quality.yml`)

- Runs Black formatting check
- Runs Pylint (fails if score < 9.5)

#### 2. Tests (`tests.yml`)

- Runs on Python 3.9, 3.10, 3.11
- Executes full test suite
- Uploads coverage to Codecov

#### 3. CI Pipeline (`ci.yml`)

- Combined workflow with all checks
- Build and import verification
- Syntax checking

### Pre-commit Checks

Before committing code, ensure:

```bash
# 1. Format code
black src/ examples/ *.py

# 2. Run quality checks
./run_quality_checks.sh

# 3. Commit only if all checks pass
```

## Development Workflow

### 1. Setup Development Environment

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Make Changes

- Write clean, well-documented code
- Follow PEP 8 style guide
- Add tests for new features
- Update documentation

### 3. Test Locally

```bash
# Run quality checks
./run_quality_checks.sh

# If any check fails, fix issues and re-run
```

### 4. Commit and Push

```bash
git add .
git commit -m "Description of changes"
git push
```

### 5. Monitor CI/CD

- Check GitHub Actions for results
- Fix any issues flagged by CI
- Wait for all checks to pass before merging

## Common Issues

### Black Formatting Failures

**Issue**: Code doesn't match Black style

**Solution**:

```bash
black src/ examples/ *.py
```

### Pylint Score Too Low

**Issue**: Pylint score < 9.5

**Solution**:

1. Review Pylint output for issues
2. Fix high-priority issues (errors, warnings)
3. Refactor code if needed
4. Some rules are disabled in `.pylintrc` - check if you need to adjust

### Test Failures

**Issue**: Tests fail locally or in CI

**Solution**:

1. Run tests with verbose output: `pytest tests/ -v --tb=long`
2. Fix the failing test or code
3. Re-run tests to verify fix

### Import Errors

**Issue**: Module imports fail

**Solution**:

1. Ensure you're in the correct directory
2. Add `sys.path.insert(0, 'src')` if needed
3. Check for circular imports
4. Verify `__init__.py` files exist

## Best Practices

### Code Style

- Use meaningful variable names
- Keep functions small and focused
- Add docstrings to modules, classes, and functions
- Use type hints where appropriate
- Comment complex logic

### Testing

- Write tests before or alongside features (TDD)
- Test edge cases and error conditions
- Use fixtures for common setup
- Mock external dependencies
- Aim for high coverage (>80%)

### Git Commits

- Write clear, descriptive commit messages
- Keep commits focused and atomic
- Reference issue numbers when applicable
- Use conventional commit format:
  - `feat:` for new features
  - `fix:` for bug fixes
  - `test:` for test changes
  - `docs:` for documentation
  - `refactor:` for code refactoring

## Resources

- [Black Documentation](https://black.readthedocs.io/)
- [Pylint Documentation](https://pylint.pycqa.org/)
- [Pytest Documentation](https://docs.pytest.org/)
- [PEP 8 Style Guide](https://pep8.org/)
