# Contributing to ParcelExtract

We welcome contributions to ParcelExtract! This document provides guidelines for contributing to the project.

## 🤝 How to Contribute

### Reporting Bugs

Before creating a bug report:
1. Check if the issue already exists in [GitHub Issues](https://github.com/yourusername/parcelextract/issues)
2. Update to the latest version to see if the issue persists
3. Try to reproduce the issue with minimal test data

When creating a bug report, include:
- **Operating system** and Python version
- **ParcelExtract version** (`parcelextract --version`)
- **Complete error message** and stack trace
- **Minimal example** that reproduces the issue
- **Input data specifications** (file format, size, etc.)

### Requesting Features

For feature requests:
1. Check existing issues and discussions
2. Describe the use case and motivation
3. Provide examples of desired behavior
4. Consider implementation complexity and scope

### Contributing Code

#### Setting Up Development Environment

1. **Fork and clone the repository:**
   ```bash
   git clone https://github.com/yourusername/parcelextract.git
   cd parcelextract
   ```

2. **Install development dependencies:**
   ```bash
   # Using uv (recommended)
   uv sync --dev
   
   # Using pip
   pip install -e ".[dev]"
   ```

3. **Verify setup:**
   ```bash
   uv run pytest
   uv run ruff check src/
   uv run mypy src/parcelextract/
   ```

#### Development Workflow

ParcelExtract follows **Test-Driven Development (TDD)**:

1. **Write failing tests first**
2. **Implement minimal code to pass tests**
3. **Refactor and improve**
4. **Verify all tests pass**

Example TDD workflow:
```bash
# 1. Create failing test
echo "def test_new_feature():
    from parcelextract.core.extractor import ParcelExtractor
    extractor = ParcelExtractor('atlas.nii.gz')
    result = extractor.new_method()
    assert result == expected_value" >> tests/test_extractor.py

# 2. Run test (should fail)
uv run pytest tests/test_extractor.py::test_new_feature -v

# 3. Implement feature
# Edit src/parcelextract/core/extractor.py

# 4. Run test (should pass)
uv run pytest tests/test_extractor.py::test_new_feature -v

# 5. Run full test suite
uv run pytest
```

#### Code Style Guidelines

**Language:** Python 3.12+

**Formatting:** We use `ruff` for code formatting and linting:
```bash
# Format code
uv run ruff format src/ tests/

# Check for issues
uv run ruff check src/ tests/

# Fix automatically fixable issues
uv run ruff check --fix src/ tests/
```

**Type Checking:** Use type hints and `mypy`:
```bash
uv run mypy src/parcelextract/
```

**Key Rules:**
- Maximum line length: 100 characters
- Use type hints for all public functions
- Follow PEP 8 naming conventions
- Keep functions focused and modular
- Prefer composition over inheritance

#### Documentation Standards

**Docstrings:** Use NumPy-style docstrings:
```python
def extract_timeseries(data: np.ndarray, atlas: np.ndarray) -> np.ndarray:
    """
    Extract time-series signals from 4D data using atlas.
    
    Parameters
    ----------
    data : numpy.ndarray
        4D neuroimaging data (x, y, z, time)
    atlas : numpy.ndarray
        3D atlas with integer labels
        
    Returns
    -------
    numpy.ndarray
        2D array (n_parcels, n_timepoints) with extracted signals
        
    Raises
    ------
    ValueError
        If data and atlas have incompatible shapes
        
    Examples
    --------
    >>> data = np.random.randn(10, 10, 10, 100)
    >>> atlas = np.random.randint(0, 5, (10, 10, 10))
    >>> signals = extract_timeseries(data, atlas)
    >>> signals.shape
    (4, 100)
    """
```

**Comments:** Use comments sparingly, prefer self-documenting code:
```python
# Good: self-documenting
parcels_with_data = [p for p in parcels if p.voxel_count > 0]

# Bad: obvious comment
parcels_with_data = []  # Create empty list
for p in parcels:  # Loop through parcels
    if p.voxel_count > 0:  # Check if parcel has data
        parcels_with_data.append(p)  # Add to list
```

#### Testing Guidelines

**Test Structure:**
- One test file per module: `test_<module>.py`
- Use descriptive test names: `test_extract_mean_handles_nan_values`
- Group related tests in classes: `TestExtractionStrategies`

**Test Coverage:**
- Aim for >90% code coverage
- Test edge cases and error conditions
- Use fixtures for common test data

**Example Test:**
```python
import pytest
import numpy as np
from parcelextract.core.strategies import MeanExtractionStrategy

class TestMeanExtractionStrategy:
    """Test mean extraction strategy."""
    
    def test_extract_basic_functionality(self):
        """Test basic mean extraction."""
        # Arrange
        strategy = MeanExtractionStrategy()
        data_4d = np.random.randn(2, 2, 2, 10)
        mask_3d = np.ones((2, 2, 2), dtype=bool)
        
        # Act
        result = strategy.extract(data_4d, mask_3d)
        
        # Assert
        assert result.shape == (10,)
        assert not np.any(np.isnan(result))
    
    def test_extract_with_nan_values(self):
        """Test mean extraction handles NaN values correctly."""
        # Arrange
        strategy = MeanExtractionStrategy()
        data_4d = np.random.randn(2, 2, 2, 10)
        data_4d[0, 0, 0, :] = np.nan  # Introduce NaN values
        mask_3d = np.ones((2, 2, 2), dtype=bool)
        
        # Act
        result = strategy.extract(data_4d, mask_3d)
        
        # Assert
        assert result.shape == (10,)
        assert not np.all(np.isnan(result))  # Should handle NaN gracefully
```

**Fixtures for Test Data:**
```python
# conftest.py
@pytest.fixture
def synthetic_4d_data():
    """Generate synthetic 4D neuroimaging data."""
    return np.random.randn(20, 20, 20, 100).astype(np.float32)

@pytest.fixture  
def synthetic_atlas():
    """Generate synthetic atlas with 10 parcels."""
    return np.random.randint(0, 11, (20, 20, 20)).astype(np.int16)
```

#### Pull Request Process

1. **Create feature branch:**
   ```bash
   git checkout -b feature/add-new-extraction-strategy
   ```

2. **Make changes following TDD:**
   - Write failing tests
   - Implement feature
   - Ensure tests pass
   - Update documentation

3. **Verify code quality:**
   ```bash
   # Run full test suite
   uv run pytest --cov=parcelextract
   
   # Check formatting and linting
   uv run ruff format src/ tests/
   uv run ruff check src/ tests/
   
   # Type checking
   uv run mypy src/parcelextract/
   ```

4. **Commit with clear messages:**
   ```bash
   git add -A
   git commit -m "feat: add weighted median extraction strategy

   - Implement WeightedMedianExtractionStrategy class
   - Add comprehensive tests with edge cases
   - Update documentation and examples
   - Maintain 90%+ test coverage
   
   Closes #123"
   ```

5. **Push and create pull request:**
   ```bash
   git push origin feature/add-new-extraction-strategy
   ```

6. **Pull request checklist:**
   - [ ] Tests pass locally
   - [ ] Code coverage maintained (>90%)
   - [ ] Documentation updated
   - [ ] Type hints added
   - [ ] Clear commit messages
   - [ ] No merge conflicts

## 🏗️ Development Guidelines

### Project Architecture

ParcelExtract follows a modular architecture:

```
src/parcelextract/
├── core/           # Core extraction logic
├── io/             # Input/output operations  
├── atlases/        # Atlas management
├── cli/            # Command-line interface
└── utils/          # Utility functions
```

**Design Principles:**
- **Single Responsibility:** Each module has one clear purpose
- **Dependency Inversion:** Depend on abstractions, not implementations
- **Strategy Pattern:** Multiple extraction algorithms
- **Factory Pattern:** Atlas loading based on input type

### Adding New Features

#### New Extraction Strategy

1. **Create strategy class in `core/strategies.py`:**
   ```python
   class MyNewStrategy(ExtractionStrategy):
       """My new extraction strategy."""
       
       def extract(self, data_4d: np.ndarray, mask_3d: np.ndarray) -> np.ndarray:
           # Implementation
           pass
   ```

2. **Add tests in `tests/test_strategies.py`:**
   ```python
   class TestMyNewStrategy:
       def test_extract_basic(self):
           # Test implementation
           pass
   ```

3. **Register strategy in `core/extractor.py`:**
   ```python
   STRATEGIES = {
       'mean': MeanExtractionStrategy,
       'median': MedianExtractionStrategy,
       'pca': PCAExtractionStrategy,
       'weighted_mean': WeightedMeanExtractionStrategy,
       'my_new': MyNewStrategy,  # Add here
   }
   ```

4. **Update CLI and documentation**

#### New Atlas Source

1. **Create manager in `atlases/`:**
   ```python
   class MyAtlasManager:
       def get_atlas(self, name: str, **kwargs) -> str:
           # Implementation
           pass
   ```

2. **Add integration tests**
3. **Update CLI atlas resolution logic**
4. **Document new atlas source**

### Performance Considerations

- **Memory usage:** Consider chunked processing for large datasets
- **CPU efficiency:** Profile bottlenecks with `cProfile`
- **I/O optimization:** Minimize file reads/writes
- **Caching:** Cache expensive computations when appropriate

### Error Handling

**Use specific exceptions:**
```python
# Good
from parcelextract.core.validators import ValidationError
if not img_4d.ndim == 4:
    raise ValidationError(f"Expected 4D image, got {img_4d.ndim}D")

# Bad  
if not img_4d.ndim == 4:
    raise ValueError("Wrong dimensions")
```

**Provide helpful error messages:**
```python
# Good
raise ValidationError(
    f"Atlas and image have incompatible shapes: "
    f"atlas {atlas.shape} vs image {img.shape[:3]}. "
    f"Consider resampling atlas to match image space."
)

# Bad
raise ValueError("Shape mismatch")
```

## 🔍 Review Process

### Code Review Criteria

**Functionality:**
- [ ] Code works as intended
- [ ] Edge cases handled appropriately
- [ ] Performance is acceptable
- [ ] No breaking changes to existing API

**Code Quality:**
- [ ] Follows style guidelines
- [ ] Well-structured and readable
- [ ] Appropriate use of design patterns
- [ ] No code duplication

**Testing:**
- [ ] Comprehensive test coverage
- [ ] Tests are clear and maintainable
- [ ] Edge cases tested
- [ ] Performance tests if applicable

**Documentation:**
- [ ] Public APIs documented
- [ ] Docstrings complete and accurate
- [ ] README updated if necessary
- [ ] Examples provided

### Reviewer Guidelines

**Be constructive:**
- Focus on code, not the person
- Explain the "why" behind suggestions
- Offer specific alternatives
- Acknowledge good practices

**Be thorough:**
- Test the changes locally
- Consider edge cases and error conditions
- Check for potential performance impacts
- Verify documentation accuracy

## 📈 Release Process

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):
- **MAJOR:** Breaking changes to public API
- **MINOR:** New features, backwards compatible
- **PATCH:** Bug fixes, backwards compatible

### Release Checklist

1. [ ] All tests pass
2. [ ] Documentation updated
3. [ ] CHANGELOG.md updated  
4. [ ] Version bumped in `pyproject.toml`
5. [ ] Git tag created
6. [ ] GitHub release created
7. [ ] Package published (if applicable)

## 🆘 Getting Help

**For contributors:**
- Join development discussions in GitHub Issues
- Ask questions in GitHub Discussions
- Review existing code for patterns and conventions

**For maintainers:**
- Review PRs promptly
- Provide constructive feedback
- Help contributors improve their submissions
- Keep documentation up to date

Thank you for contributing to ParcelExtract! 🙏