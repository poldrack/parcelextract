# TASKS.md - ParcelExtract Project Tasks

## ✅ COMPLETED: Project Setup & Configuration
- [x] Initialize project with uv (`uv init --package .`)
- [x] Update pyproject.toml with project metadata and dependencies
- [x] Set up directory structure as specified in PRD
- [x] Configure .gitignore for Python project (basic setup)
- [ ] Set up pre-commit hooks for code quality
- [x] Create initial README.md with project overview
- [x] Create PLANNING.md for strategic notes
- [x] Create SCRATCHPAD.md for development notes

## ✅ COMPLETED: Milestone 1 - Core Functionality (Priority 1)

### ✅ Test Framework Setup - COMPLETE
- [x] Set up pytest configuration (pytest.ini)
- [x] Create test fixtures directory structure
- [x] Generate synthetic 4D Nifti test data
- [x] Create minimal test atlases with known properties
- [x] Set up pytest fixtures for shared test resources
- [x] Configure pytest-cov for coverage reporting

### ✅ Input Validation Module (`core/validators.py`) - COMPLETE
- [x] Write tests for 4D Nifti validation
- [x] Write tests for file path validation
- [x] Write tests for nibabel image object validation
- [x] Write tests for dimension checking
- [x] Implement validate_input_image() function
- [x] Implement validate_atlas_spec() function
- [x] Implement validate_output_dir() function
- [x] Add custom exception classes for validation errors
- **Status**: 22 passing tests, comprehensive edge case coverage

### ✅ File I/O Module (`io/readers.py`) - COMPLETE
- [x] Write tests for Nifti file loading
- [x] Write tests for handling .nii and .nii.gz files
- [x] Write tests for error handling on invalid files
- [x] Implement load_nifti() function
- [x] Implement get_image_metadata() function
- [x] Implement validate_4d_image() function
- [x] Add memory-efficient loading for large files
- **Status**: 24 passing tests, handles compressed/uncompressed formats

### ✅ Extraction Strategies (`core/strategies.py`) - COMPLETE
- [x] Write tests for mean extraction strategy
- [x] Write tests for handling NaN values
- [x] Write tests for empty parcels edge case
- [x] Write tests for single voxel parcels
- [x] Create ExtractionStrategy base class
- [x] Implement MeanExtractionStrategy class
- [x] Implement MedianExtractionStrategy class
- [x] Implement PCAExtractionStrategy class
- [x] Implement WeightedMeanExtractionStrategy class
- [x] Add proper dtype consistency handling
- **Status**: 23 passing tests, all 4 strategies implemented with full edge case handling

### ✅ Core Extractor (`core/extractor.py`) - COMPLETE 
- [x] Write tests for ParcelExtractor initialization (TDD approach)
- [x] Write tests for atlas and strategy parameters (TDD approach) 
- [x] Write tests for fit_transform() method (TDD approach)
- [x] Write tests for actual signal extraction (TDD approach)
- [x] Write tests for different strategy selection (TDD approach)
- [x] Write tests for invalid strategy error handling (TDD approach)
- [x] Write tests for PCA strategy functionality (TDD approach)
- [x] Write tests for input validation in fit_transform (TDD approach)
- [x] Implement ParcelExtractor class with proper TDD methodology
- [x] Implement basic signal extraction pipeline
- [x] Add support for different strategy selection (mean, median, pca, weighted_mean)
- [x] Implement proper input validation using existing validators
- [x] Implement error handling and recovery for invalid inputs/strategies
- [ ] Add logging for extraction process
- **Status**: 9 passing tests, full functionality with all strategies and validation

### Basic Output Writer (`io/writers.py`)
- [ ] Write tests for TSV file generation
- [ ] Write tests for output directory creation
- [ ] Write tests for file naming
- [ ] Implement write_timeseries_tsv() function
- [ ] Implement create_output_dir() function
- [ ] Add basic metadata to output files

## Milestone 2: Atlas Integration (Priority 2)

### Atlas Manager (`atlases/manager.py`)
- [ ] Write tests for atlas loading from file
- [ ] Write tests for atlas validation
- [ ] Write tests for label extraction
- [ ] Write tests for metadata handling
- [ ] Implement AtlasManager class
- [ ] Implement load_atlas() method
- [ ] Implement get_labels() method
- [ ] Implement get_metadata() method
- [ ] Add atlas format detection
- [ ] Support deterministic atlases
- [ ] Support probabilistic atlases

### TemplateFlow Integration (`atlases/templateflow.py`)
- [ ] Write tests for TemplateFlow queries
- [ ] Write tests for atlas downloading
- [ ] Write tests for caching mechanism
- [ ] Write tests for multiple resolutions
- [ ] Implement TemplateFlowManager class
- [ ] Implement query_available_atlases() method
- [ ] Implement download_atlas() method
- [ ] Implement cache management
- [ ] Add resolution selection logic
- [ ] Handle connection errors gracefully

### Atlas-Image Alignment
- [ ] Write tests for alignment checking
- [ ] Write tests for resampling operations
- [ ] Implement check_alignment() function
- [ ] Implement resample_atlas_to_image() function
- [ ] Add warning system for misalignment

## Milestone 3: Advanced Features (Priority 3)

### Additional Extraction Strategies
- [ ] Write tests for median extraction
- [ ] Write tests for PCA extraction
- [ ] Write tests for weighted mean extraction
- [ ] Implement MedianExtractionStrategy class
- [ ] Implement PCAExtractionStrategy class
- [ ] Implement WeightedMeanExtractionStrategy class
- [ ] Add strategy selection mechanism
- [ ] Optimize performance for each strategy

### Confound Regression
- [ ] Write tests for confound loading
- [ ] Write tests for confound validation
- [ ] Write tests for regression implementation
- [ ] Implement load_confounds() function
- [ ] Implement validate_confounds() function
- [ ] Implement apply_confound_regression() method
- [ ] Add to extraction pipeline
- [ ] Handle missing confound values

### Batch Processing
- [ ] Write tests for batch input handling
- [ ] Write tests for progress tracking
- [ ] Implement batch_process() function
- [ ] Implement progress indicators
- [ ] Add batch error handling
- [ ] Create batch summary reports

### Performance Optimization
- [ ] Profile current implementation
- [ ] Optimize memory usage
- [ ] Benchmark against requirements (<30s for typical image)

## Milestone 4: BIDS Compliance (Priority 4)

### BIDS Parsing (`io/bids.py`)
- [ ] Write tests for BIDS entity parsing
- [ ] Write tests for filename generation
- [ ] Write tests for entity preservation
- [ ] Implement parse_bids_filename() function
- [ ] Implement generate_bids_filename() function
- [ ] Implement preserve_entities() function
- [ ] Add derivatives naming support
- [ ] Handle special BIDS cases

### JSON Sidecar Generation
- [ ] Write tests for metadata collection
- [ ] Write tests for JSON structure
- [ ] Write tests for sidecar naming
- [ ] Implement collect_metadata() function
- [ ] Implement generate_json_sidecar() function
- [ ] Add extraction parameters to metadata
- [ ] Add atlas information to metadata
- [ ] Include temporal information (TR, timepoints)
- [ ] Add preprocessing status fields

### BIDS Derivatives Structure
- [ ] Write tests for derivatives directory structure
- [ ] Write tests for dataset_description.json
- [ ] Implement create_derivatives_structure() function
- [ ] Generate dataset_description.json
- [ ] Ensure BIDS-compliant organization

## Milestone 5: CLI & Documentation (Priority 5)

### Command-Line Interface (`cli/main.py`)
- [ ] Write smoke test for main() function
- [ ] Write tests for argument parsing
- [ ] Write tests for CLI validation
- [ ] Write tests for error handling
- [ ] Implement argument parser
- [ ] Implement main() entry point
- [ ] Add all CLI options from PRD
- [ ] Implement verbose/quiet modes
- [ ] Add --version flag
- [ ] Create helpful error messages

### Logging Configuration (`utils/logging.py`)
- [ ] Write tests for logging setup
- [ ] Write tests for log levels
- [ ] Implement configure_logging() function
- [ ] Add file logging option
- [ ] Implement log rotation
- [ ] Add structured logging format

### Utility Functions (`utils/helpers.py`)
- [ ] Write tests for helper functions
- [ ] Implement common utility functions
- [ ] Add type conversion utilities
- [ ] Add path manipulation helpers

### User Documentation
- [ ] Write installation guide
- [ ] Create quick start tutorial
- [ ] Write API reference documentation
- [ ] Document all CLI options
- [ ] Create example notebooks
- [ ] Add troubleshooting guide

### Developer Documentation
- [ ] Document architecture decisions
- [ ] Write contributing guidelines
- [ ] Create testing guide
- [ ] Document release process
- [ ] Add code style guide

### Example Notebooks
- [ ] Create basic usage notebook
- [ ] Create batch processing example
- [ ] Create atlas comparison notebook
- [ ] Create visualization examples

## Milestone 6: Quality Assurance & Release

### Testing Completion
- [ ] Achieve >90% test coverage
- [ ] Add edge case tests
- [ ] Add regression test suite
- [ ] Perform integration testing
- [ ] Add performance benchmarks
- [ ] Test on multiple platforms

### Code Quality
- [ ] Add type hints to all public functions
- [ ] Complete all docstrings
- [ ] Run and fix pylint/flake8 issues
- [ ] Run mypy type checking
- [ ] Ensure PEP 8 compliance
- [ ] Remove all TODO comments

### Package Distribution
- [ ] Configure package metadata
- [ ] Set up GitHub Actions CI/CD
- [ ] Create release workflow
- [ ] Prepare CHANGELOG.md
- [ ] Test package installation
- [ ] Create distribution packages

### Final Release (v1.0.0)
- [ ] Final code review
- [ ] Update version numbers
- [ ] Create release notes
- [ ] Tag release in git
- [ ] Publish to PyPI (if applicable)
- [ ] Announce release

## Continuous Tasks (Throughout Development)

### Code Quality Maintenance
- [ ] Regular code reviews
- [ ] Update tests for new features
- [ ] Maintain documentation accuracy
- [ ] Monitor test coverage
- [ ] Address technical debt

### Project Management
- [ ] Update TASKS.md with completion status
- [ ] Add new tasks as discovered
- [ ] Use SCRATCHPAD.md for planning
- [ ] Regular progress reviews
- [ ] Maintain clean git history

## Notes

- **Priority**: Tasks should be completed following the Test-Driven Development (TDD) approach
- **Testing First**: Always write failing tests before implementing functionality
- **No Code in `__init__.py`**: Keep all `__init__.py` files empty as specified
- **Dependencies**: Use `uv` for all dependency management
- **Documentation**: Update docstrings immediately after implementation
- **Coverage Goal**: Maintain >90% test coverage throughout development

## Task Status Legend
- [ ] Not started
- [x] Completed
- [~] In progress
- [!] Blocked

---

Last Updated: [Date]
Total Tasks: ~185
Completed: 0
In Progress: 0
Blocked: 0
