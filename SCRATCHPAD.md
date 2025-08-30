# Development scratchpad

- Use this file to keep notes on ongoing development work.
- When the work is completed, clean it out from this file, so that the contents only reflect ongoing work.

## COMPLETED PHASE 1 (Core Foundation) - FINISHED ✅

✅ **Project Setup Complete**
- Full module structure created with proper `__init__.py` files (kept empty per spec)
- Dependencies configured: nibabel, nilearn, numpy, pandas, scipy, scikit-learn, templateflow
- pytest setup with fixtures for synthetic 4D/3D data and test atlases
- 88%+ test coverage across core modules

✅ **Input Validation Module** (`src/parcelextract/core/validators.py`)
- Comprehensive validation for 4D neuroimaging data, atlas specifications, output directories
- Custom `ValidationError` exception class
- Handles file paths, nibabel objects, dimension checking, atlas validation
- 22 passing tests with edge cases (empty masks, NaN values, invalid files)

✅ **File I/O Module** (`src/parcelextract/io/readers.py`) 
- `load_nifti()` function with proper error handling for .nii/.nii.gz files
- `get_image_metadata()` extracts dimensions, voxel sizes, TR, units
- `validate_4d_image()` ensures proper 4D structure with timepoints
- 24 passing tests including compressed/uncompressed formats, metadata extraction

✅ **Extraction Strategies Module** (`src/parcelextract/core/strategies.py`)
- Abstract base class `ExtractionStrategy` using proper ABC pattern
- `MeanExtractionStrategy` with NaN handling via `np.nanmean()`
- `MedianExtractionStrategy` with `np.nanmedian()`
- `PCAExtractionStrategy` using scikit-learn, handles edge cases (single voxel, constant data)
- `WeightedMeanExtractionStrategy` for probabilistic atlases
- 23 passing tests including edge cases (empty masks, NaN values, insufficient timepoints)

## ✅ TDD DEMONSTRATION COMPLETED - Core Extractor

**Proper TDD Methodology Demonstrated** with `ParcelExtractor` class:

### Red-Green-Refactor Cycles Completed:
1. **Cycle 1**: Basic instantiation test → minimal empty class
2. **Cycle 2**: Atlas parameter test → store atlas as instance variable  
3. **Cycle 3**: Strategy parameter test → store strategy as instance variable
4. **Cycle 4**: `fit_transform()` method test → return correct shape dummy array
5. **Cycle 5**: Actual signal extraction test → implement real extraction using existing strategies

### Key TDD Principles Demonstrated:
- ✅ **RED**: Write failing test first, verify failure
- ✅ **GREEN**: Write minimal code to make test pass, verify all tests pass
- ✅ **REFACTOR**: (minimal needed at this stage)
- ✅ **Incremental**: Built functionality step-by-step, never more than needed
- ✅ **Regression Safety**: All previous tests continued to pass at each step

### Current Status:
- **74 total tests passing** (22 validators + 24 I/O + 23 strategies + 5 extractor)
- **`ParcelExtractor` basic functionality working** with mean extraction strategy
- **Real signal extraction implemented** using existing strategy pattern
- **Clean API design** driven by test requirements

## READY FOR NEXT PHASE
- Core extractor needs strategy selection logic (currently hardcoded to mean)
- Could add atlas management or output writing modules
- All foundation components complete and well-tested
