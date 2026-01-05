# Test Suite

Comprehensive test suite for the Sensor Fusion Dashboard project.

## Running Tests

### Run all tests

```bash
pytest tests/ -v
```

### Run specific test file

```bash
pytest tests/test_core.py -v
```

### Run with coverage

```bash
pytest tests/ --cov=src --cov-report=html
```

### Run specific test class or function

```bash
pytest tests/test_core.py::TestSensorParser -v
pytest tests/test_core.py::TestSensorParser::test_parse_valid_line -v
```

## Test Structure

- `test_core.py` - Tests for core sensor functionality (parser, buffers, battery)
- `test_ai.py` - Tests for AI/ML character recognition components
- `test_actions.py` - Tests for action detection functionality

## Coverage

View the HTML coverage report after running tests:

```bash
pytest tests/ --cov=src --cov-report=html
open htmlcov/index.html
```

## CI/CD

Tests run automatically on every push and pull request via GitHub Actions:

- **Code Quality**: Black formatting + Pylint (score >= 9.5)
- **Tests**: Python 3.9, 3.10, 3.11
- **Coverage**: Reports uploaded to Codecov

## Writing New Tests

Follow these conventions:

1. Create test files with `test_` prefix
2. Create test classes with `Test` prefix
3. Create test functions with `test_` prefix
4. Use descriptive test names
5. Add docstrings to test methods
6. Use pytest fixtures for common setup

Example:

```python
class TestNewFeature:
    """Test cases for new feature"""
    
    def test_basic_functionality(self):
        """Test basic functionality works as expected"""
        result = new_feature.process()
        assert result == expected_value
```
