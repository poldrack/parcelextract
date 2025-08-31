# CLAUDE.md - ParcelExtract Project Guide

## Project Overview

**ParcelExtract** is a Python module for extracting time-series signals from 4D neuroimaging data (fMRI) based on brain parcellation schemes. This tool provides researchers with a standardized, BIDS-compliant method for extracting regional brain signals for connectivity and network analyses.

**Version:** 1.0.0  
**Python:** 3.12+  
**Package Manager:** uv (>=0.8.14)

## Development Strategy

- Use a test-driven development strategy, developing tests prior to generating solutions to the tests.
- Run the tests and ensure that they fail prior to generating any solutions.
- Write code that passes the tests.
- IMPORTANT: Do not modify the tests simply so that the code passes. Only modify the tests if you identify a specific error in the test.

## Notes for Development

- Think about the problem before generating code.
- Always add a smoke test for the main() function.
- Prefer reliance on widely used packages (such as numpy, pandas, and scikit-learn); avoid unknown packages from Github.
- Do not include *any* code in `__init__.py` files.
- Use pytest for testing.
- Write code that is clean and modular. Prefer shorter functions/methods over longer ones.
- Use functions rather than classes for tests. Use pytest fixtures to share resources between tests.

## Session Guidelines

- Always read PLANNING.md at the start of every new conversation
- Check TASKS.md and SCRATCHPAD.md before starting your work
- Mark completed tasks immediately within TASKS.md
- Add newly discovered tasks to TASKS.md
- Use SCRATCHPAD.md as a scratchpad to outline plans

## Core Architecture

### Module Structure
```
src/parcelextract/
├── __init__.py              # Keep empty
├── core/
│   ├── __init__.py          # Keep empty
│   ├── extractor.py         # Main extraction logic
│   ├── strategies.py        # Extraction strategies
│   └── validators.py        # Input validation
├── io/
│   ├── __init__.py          # Keep empty
│   ├── readers.py           # File reading utilities
│   ├── writers.py           # Output generation
│   └── bids.py             # BIDS naming/parsing
├── atlases/
│   ├── __init__.py          # Keep empty
│   ├── manager.py          # Atlas management
│   └── templateflow.py     # TemplateFlow integration
├── cli/
│   ├── __init__.py          # Keep empty
│   └── main.py             # Command-line interface
├── utils/
│   ├── __init__.py          # Keep empty
│   ├── logging.py          # Logging configuration
│   └── helpers.py          # Utility functions
└── tests/
    ├── __init__.py          # Keep empty
    ├── test_extractor.py
    ├── test_strategies.py
    ├── test_io.py
    ├── test_atlases.py
    └── fixtures/           # Test data
```

### Key Dependencies
```toml
# Use uv for dependency management
[dependencies]
nibabel = ">=3.2.0"      # Neuroimaging file I/O
nilearn = ">=0.10.0"     # Neuroimaging data manipulation
templateflow = ">=0.8.0" # Atlas management
numpy = ">=1.20.0"       # Numerical operations
pandas = ">=1.3.0"       # Data structuring
scipy = ">=1.7.0"        # Scientific computing

[dev-dependencies]
pytest = ">=7.0.0"
pytest-cov = ">=4.0.0"
```

## Core Functional Requirements

### Input Processing
- Accept 4D Nifti files (.nii, .nii.gz)
- Support both file paths and nibabel image objects
- Validate input dimensions (must be 4D)
- Accept parcellation atlas via:
  - TemplateFlow atlas names
  - Custom Nifti parcellation files
  - Predefined atlas identifiers

### Signal Extraction Strategies
1. **Mean**: Average signal across voxels in parcel
2. **Median**: Median signal across voxels
3. **PCA**: First principal component
4. **Weighted Mean**: Weighted by probability maps

### Output Requirements
- Tab-delimited (.tsv) files for time-series
- JSON sidecar files with metadata
- BIDS-compliant naming:
  ```
  sub-<label>_[ses-<label>]_task-<label>_[run-<index>]_space-<label>_atlas-<label>_timeseries.tsv
  ```

## Core API Design

### Main Classes

```python
class ParcelExtractor:
    """Main extraction class"""
    def __init__(self, atlas, strategy='mean', **kwargs):
        pass
    
    def fit(self, img_4d, confounds=None):
        pass
    
    def transform(self, img_4d, confounds=None):
        pass
    
    def fit_transform(self, img_4d, confounds=None):
        pass

class ExtractionStrategy:
    """Base class for extraction strategies"""
    def extract(self, data_4d, mask_3d):
        pass

class AtlasManager:
    """Atlas loading and management"""
    def load_atlas(self, atlas_spec):
        pass
    
    def get_labels(self):
        pass
    
    def get_metadata(self):
        pass
```

### CLI Interface
```bash
parcelextract \
    --input /path/to/sub-01_task-rest_bold.nii.gz \
    --atlas schaefer2018 \
    --dimensionality 200 \
    --strategy mean \
    --output-dir /path/to/derivatives \
    --confounds /path/to/confounds.tsv
```

## Testing Requirements

### Test Coverage Goals
- Minimum 90% code coverage
- 100% coverage for critical paths
- Edge case testing for all validators

### Test Categories
1. **Unit Tests**: Individual functions/methods
2. **Integration Tests**: End-to-end workflows
3. **Regression Tests**: Known edge cases
4. **Performance Tests**: Speed and memory usage

### Test Data Strategy
- Use synthetic 4D Nifti files for unit tests
- Create minimal test atlases with known properties
- Generate BIDS-compliant test structures

## Implementation Phases

### Phase 1: Core Functionality (Priority 1)
- [ ] Basic project structure setup
- [ ] Input validation (validators.py)
- [ ] Mean extraction strategy
- [ ] Simple file I/O (readers.py, writers.py)
- [ ] Unit test framework setup

### Phase 2: Atlas Integration (Priority 2)
- [ ] Atlas manager implementation
- [ ] TemplateFlow integration
- [ ] Atlas metadata handling
- [ ] Multiple atlas format support

### Phase 3: Advanced Features (Priority 3)
- [ ] Additional extraction strategies (median, PCA, weighted)
- [ ] Confound regression support
- [ ] Batch processing capabilities
- [ ] Progress indicators

### Phase 4: BIDS Compliance (Priority 4)
- [ ] BIDS naming conventions (bids.py)
- [ ] JSON sidecar generation
- [ ] Entity preservation from input files
- [ ] Derivatives naming support

### Phase 5: CLI & Documentation (Priority 5)
- [ ] Command-line interface (cli/main.py)
- [ ] Argument parsing and validation
- [ ] Logging configuration
- [ ] User documentation
- [ ] Example notebooks

## Code Quality Standards

### Style Guidelines
- Follow PEP 8
- Use type hints for all public functions
- Maximum line length: 100 characters
- Docstrings for all public APIs

### Error Handling
- Validate all inputs early
- Provide clear, actionable error messages
- Use custom exceptions for domain-specific errors
- Log errors appropriately

### Performance Considerations
- Process typical 4D image (200 timepoints, 3mm) in <30 seconds
- Use memory-efficient loading for large datasets
- Support parallel processing where applicable
- Chunk processing for very large files

## Common Pitfalls to Avoid

1. **Atlas Misalignment**: Always check atlas-image alignment
2. **NaN Handling**: Properly handle missing data in parcels
3. **BIDS Naming**: Preserve all entities from input files
4. **Type Consistency**: Ensure consistent dtype throughout pipeline

## Validation Checklist

Before considering any component complete:
- [ ] Unit tests written and passing
- [ ] Function/method has type hints
- [ ] Docstring is complete and accurate
- [ ] Error cases are handled
- [ ] Logging is appropriate
- [ ] Performance is acceptable

## Quick Reference

### File Reading
```python
import nibabel as nib
img = nib.load('path/to/file.nii.gz')
data = img.get_fdata()  # Returns 4D array
```

### BIDS Parsing
```python
# Use simple regex or string parsing
# Avoid heavy dependencies for v1.0
import re
pattern = r'sub-(?P<subject>[a-zA-Z0-9]+)'
```

### Testing Pattern
```python
import pytest
import numpy as np

def test_extraction_mean():
    # Arrange
    data_4d = np.random.randn(10, 10, 10, 50)
    mask_3d = np.ones((10, 10, 10), dtype=bool)
    
    # Act
    result = extract_mean(data_4d, mask_3d)
    
    # Assert
    assert result.shape == (50,)
    assert not np.any(np.isnan(result))
```

## Common Commands

```bash
# Setup project with uv
uv init --package .
uv add nibabel nilearn numpy pandas scipy templateflow

# Run tests
uv run pytest
uv run pytest --cov=parcelextract

# Run specific test file
uv run pytest tests/test_extractor.py

# Run CLI
uv run python -m parcelextract.cli.main --help
```

## Important Notes

1. **Start with TDD**: Write failing tests first, then implement
2. **Keep it Simple**: Don't over-engineer for v1.0
3. **Document as You Go**: Update docstrings immediately
4. **Test Edge Cases**: Empty parcels, all-NaN regions, single voxel parcels
5. **Version Everything**: Use semantic versioning from the start

## Resources

- [BIDS Specification](https://bids-specification.readthedocs.io/)
- [Nibabel Documentation](https://nipy.org/nibabel/)
- [Nilearn Documentation](https://nilearn.github.io/)
- [TemplateFlow](https://www.templateflow.org/)

---

Remember: The goal is a reliable, well-tested tool that researchers can trust with their data. Quality over features for v1.0.
