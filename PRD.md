# Project Requirement Document: ParcelExtract

## 1. Executive Summary

**Project Name:** ParcelExtract  
**Version:** 1.0.0  
**Date:** August 2025  
**Author:** Development Team  

ParcelExtract is a Python module designed to extract time-series signals from 4-dimensional neuroimaging data (fMRI) based on brain parcellation schemes. The tool will provide researchers with a standardized, BIDS-compliant method for extracting regional brain signals for subsequent connectivity and network analyses.

## 2. Project Overview

### 2.1 Purpose
The primary purpose of ParcelExtract is to streamline the extraction of parcellated brain signals from 4D Nifti images, providing researchers with a reliable, standardized tool that integrates seamlessly with existing neuroimaging workflows.

### 2.2 Scope
- Extract signals from 4D Nifti brain images using various parcellation atlases
- Support multiple extraction strategies (mean, median, PCA, etc.)
- Generate BIDS-compliant output files with comprehensive metadata
- Provide both command-line interface (CLI) and Python API access
- Support batch processing of multiple subjects/sessions

### 2.3 Target Users
- Neuroimaging researchers
- Data scientists working with brain connectivity
- Clinical researchers analyzing fMRI data
- Neuroinformatics pipeline developers

## 3. Functional Requirements

### 3.1 Core Functionality

#### 3.1.1 Input Processing
- **FR-001:** Accept 4D Nifti files (.nii, .nii.gz) as primary input
- **FR-002:** Support both file paths and nibabel image objects as input
- **FR-003:** Validate input dimensions (must be 4D)
- **FR-004:** Accept parcellation atlas specification via:
  - TemplateFlow atlas names
  - Custom Nifti parcellation files
  - Predefined atlas identifiers

#### 3.1.2 Signal Extraction
- **FR-005:** Extract time-series from each parcel/region
- **FR-006:** Support multiple extraction strategies:
  - Mean signal across voxels
  - Median signal across voxels
  - Principal component (first eigenvariate)
  - Weighted mean (by probability maps)
- **FR-007:** Handle masked regions and NaN values appropriately
- **FR-008:** Support confound regression during extraction (optional)

#### 3.1.3 Output Generation
- **FR-009:** Save extracted signals as tab-delimited text files (.tsv)
- **FR-010:** Generate JSON sidecar files with metadata
- **FR-011:** Follow BIDS derivatives naming conventions
- **FR-012:** Support custom output directories and naming schemes

### 3.2 Metadata Management

#### 3.2.1 JSON Sidecar Contents
- **FR-013:** Record extraction parameters and methods
- **FR-014:** Include parcellation atlas information:
  - Atlas name and version
  - Number of parcels
  - Resolution
  - Reference publication/DOI
- **FR-015:** Document preprocessing status
- **FR-016:** Include temporal information:
  - TR (repetition time)
  - Number of time points
  - Extraction timestamp

### 3.3 Atlas Integration

#### 3.3.1 TemplateFlow Support
- **FR-017:** Query available atlases from TemplateFlow
- **FR-018:** Automatic atlas downloading and caching
- **FR-019:** Support multiple atlas resolutions
- **FR-020:** Handle probabilistic and deterministic atlases

### 3.4 BIDS Compliance

#### 3.4.1 File Naming
- **FR-021:** Generate BIDS-compliant output filenames:
  - `sub-<label>_[ses-<label>]_task-<label>_[run-<index>]_space-<label>_atlas-<label>_timeseries.tsv`
- **FR-022:** Preserve BIDS entities from input files
- **FR-023:** Add appropriate derivative suffixes

## 4. Non-Functional Requirements

### 4.1 Performance
- **NFR-001:** Process a typical 4D image (200 timepoints, 3mm resolution) in under 30 seconds
- **NFR-002:** Support parallel processing for batch operations
- **NFR-003:** Memory-efficient loading of large datasets

### 4.2 Reliability
- **NFR-004:** Comprehensive error handling and logging
- **NFR-005:** Graceful degradation when optional features unavailable
- **NFR-006:** Validation of all inputs and outputs

### 4.3 Usability
- **NFR-007:** Clear, informative error messages
- **NFR-008:** Comprehensive documentation with examples
- **NFR-009:** Progress indicators for long-running operations

### 4.4 Maintainability
- **NFR-010:** Modular architecture with clear separation of concerns
- **NFR-011:** Comprehensive test coverage (>90%)
- **NFR-012:** PEP 8 compliant code style
- **NFR-013:** Type hints for all public functions

### 4.5 Compatibility
- **NFR-014:** Support Python 3.12+
- **NFR-015:** Cross-platform compatibility (Linux, macOS)
- **NFR-016:** Integration with common neuroimaging tools

## 5. Technical Architecture

### 5.1 Module Structure
```
parcelextract/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── extractor.py       # Main extraction logic
│   ├── strategies.py      # Extraction strategies
│   └── validators.py      # Input validation
├── io/
│   ├── __init__.py
│   ├── readers.py         # File reading utilities
│   ├── writers.py         # Output generation
│   └── bids.py           # BIDS naming/parsing
├── atlases/
│   ├── __init__.py
│   ├── manager.py        # Atlas management
│   └── templateflow.py   # TemplateFlow integration
├── cli/
│   ├── __init__.py
│   └── main.py          # Command-line interface
├── utils/
│   ├── __init__.py
│   ├── logging.py       # Logging configuration
│   └── helpers.py       # Utility functions
└── tests/
    ├── __init__.py
    ├── test_extractor.py
    ├── test_strategies.py
    ├── test_io.py
    ├── test_atlases.py
    └── fixtures/         # Test data
```

### 5.2 Key Dependencies
- **uv** (>=0.8.14): package management
- **nibabel** (>=3.2.0): Neuroimaging file I/O
- **nilearn** (>=0.10.0): Neuroimaging data manipulation
- **templateflow** (>=0.8.0): Atlas management
- **numpy** (>=1.20.0): Numerical operations
- **pandas** (>=1.3.0): Data structuring
- **scipy** (>=1.7.0): Scientific computing

### 5.3 API Design

#### 5.3.1 Core Classes
```python
class ParcelExtractor:
    """Main extraction class"""
    def __init__(self, atlas, strategy='mean', **kwargs)
    def fit(self, img_4d, confounds=None)
    def transform(self, img_4d, confounds=None)
    def fit_transform(self, img_4d, confounds=None)
    
class ExtractionStrategy:
    """Base class for extraction strategies"""
    def extract(self, data_4d, mask_3d)
    
class AtlasManager:
    """Atlas loading and management"""
    def load_atlas(self, atlas_spec)
    def get_labels(self)
    def get_metadata(self)
```

#### 5.3.2 CLI Interface
```bash
parcelextract \
    --input /path/to/sub-01_task-rest_bold.nii.gz \
    --atlas schaefer2018 \
    --dimensionality 200 \
    --strategy mean \
    --output-dir /path/to/derivatives \
    --confounds /path/to/confounds.tsv
```

## 6. Testing Requirements

### 6.1 Test Strategy
- **Unit tests** for all core functions
- **Integration tests** for end-to-end workflows
- **Regression tests** for known edge cases
- **Performance tests** for optimization validation

### 6.2 Test Coverage
- Minimum 90% code coverage
- 100% coverage for critical paths
- Edge case testing for all input validators

### 6.3 Test Data
- Synthetic 4D Nifti files for unit tests
- Sample atlases with known properties
- BIDS-compliant test dataset for integration tests

### 6.4 Continuous Integration
- Automated testing on push/PR
- Multi-platform testing (Linux, macOS, Windows)
- Multiple Python version testing (3.8, 3.9, 3.10, 3.11, 3.12)

## 7. Documentation Requirements

### 7.1 User Documentation
- **Installation guide** with dependency management
- **Quick start tutorial** with example workflow
- **API reference** with all public functions
- **CLI documentation** with all options
- **Example notebooks** demonstrating use cases

### 7.2 Developer Documentation
- **Architecture overview** with design decisions
- **Contributing guidelines** with code standards
- **Testing guide** for running and writing tests
- **Release process** documentation

## 8. Development Methodology

### 8.1 Test-Driven Development (TDD)
- Write tests before implementation
- Red-Green-Refactor cycle
- Continuous integration from day one

### 8.2 Version Control
- Git-based workflow
- Feature branches for development
- Pull request reviews required
- Semantic versioning (MAJOR.MINOR.PATCH)

### 8.3 Code Quality
- Pre-commit hooks for formatting
- Type checking with mypy
- Linting with pylint/flake8
- Documentation coverage checks

## 9. Deliverables

### 9.1 Phase 1: Core Functionality (Weeks 1-4)
- Basic extraction with mean strategy
- Simple file I/O
- Unit test framework

### 9.2 Phase 2: Atlas Integration (Weeks 5-6)
- TemplateFlow integration
- Multiple atlas support
- Atlas metadata handling

### 9.3 Phase 3: Advanced Features (Weeks 7-8)
- Multiple extraction strategies
- Confound regression
- Batch processing

### 9.4 Phase 4: BIDS Compliance (Weeks 9-10)
- BIDS naming conventions
- JSON sidecar generation
- PyBIDS integration

### 9.5 Phase 5: Documentation & Release (Weeks 11-12)
- Complete documentation
- Example notebooks
- Package distribution setup
- Initial release (v1.0.0)

## 10. Success Criteria

### 10.1 Functional Criteria
- Successfully extracts signals from standard test datasets
- Produces BIDS-compliant outputs
- Integrates with TemplateFlow atlases

### 10.2 Quality Criteria
- >90% test coverage
- All tests passing
- Documentation complete
- Code review approved

### 10.3 Performance Criteria
- Meets performance benchmarks
- Memory usage within acceptable limits
- Scales linearly with data size

## 11. Risk Assessment

### 11.1 Technical Risks
- **Risk:** Atlas format inconsistencies
  - **Mitigation:** Comprehensive validation and conversion utilities
  
- **Risk:** Memory constraints with large datasets
  - **Mitigation:** Chunked processing and memory mapping

- **Risk:** BIDS specification changes
  - **Mitigation:** Abstraction layer for BIDS handling

### 11.2 Project Risks
- **Risk:** Scope creep with additional features
  - **Mitigation:** Strict adherence to PRD, defer to v2.0
  
- **Risk:** Dependency version conflicts
  - **Mitigation:** Comprehensive dependency testing, version pinning

## 12. Future Enhancements (v2.0+)

- Surface-based parcellation support
- Dynamic parcellation generation
- Real-time processing capabilities
- GUI interface
- Cloud-based processing support
- Additional extraction strategies (ICA, clustering)
- Multi-modal integration (structural + functional)

## 13. Approval and Sign-off

This PRD represents the complete specification for ParcelExtract v1.0.0. Implementation should follow the requirements and timeline outlined above, with regular reviews at each phase milestone.

**Prepared by:** Development Team  
**Review Date:** [To be scheduled]  
**Approval Date:** [Pending]
