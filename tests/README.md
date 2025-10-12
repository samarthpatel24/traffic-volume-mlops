# Test files for the Traffic Volume Prediction MLOps Project

This directory contains test files to validate the functionality of the machine learning pipeline.

## Test Files

### `test_models.py`
- **TestDataPreprocessing**: Tests data cleaning, feature engineering, and encoding functions
- **TestModelValidation**: Tests configuration loading and data validation
- **TestModelPerformance**: Tests model performance against defined thresholds

## Running Tests

Run all tests:
```bash
python -m pytest tests/ -v
```

Run specific test file:
```bash
python -m pytest tests/test_models.py -v
```

Run with coverage:
```bash
python -m pytest tests/ --cov=src --cov-report=html
```

## Test Coverage

The tests cover:
- Data preprocessing pipeline
- Feature engineering functions
- Configuration validation
- Model performance thresholds
- Data validation requirements

## CI/CD Integration

These tests are automatically run in GitHub Actions on:
- Pull requests
- Pushes to main/master branches
- Manual workflow triggers

## Adding New Tests

When adding new functionality:
1. Create corresponding test functions
2. Follow naming convention: `test_<function_name>`
3. Include both positive and negative test cases
4. Update this README with new test descriptions