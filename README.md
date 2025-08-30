# ParcelExtract

**Extract time-series signals from 4D neuroimaging data using brain parcellation schemes.**

ParcelExtract is a Python package and command-line tool for extracting regional time-series signals from 4D neuroimaging data (e.g., fMRI) using brain atlases. It supports multiple extraction strategies and provides BIDS-compliant outputs for seamless integration into neuroimaging analysis pipelines.

[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-110%20passing-green.svg)]()
[![Coverage](https://img.shields.io/badge/coverage-89%25-brightgreen.svg)]()

## 🚀 Features

- **Multiple Extraction Strategies**: Mean, median, PCA, and weighted mean signal extraction
- **TemplateFlow Integration**: Automatic downloading of standard brain atlases
- **BIDS-Compliant Outputs**: TSV time-series files with JSON sidecar metadata
- **Flexible Atlas Support**: Use TemplateFlow atlases or custom parcellation files
- **Command-Line Interface**: Easy-to-use CLI for batch processing and scripting
- **Python API**: Integrate directly into your analysis workflows
- **Comprehensive Testing**: 110 tests with 89% code coverage

## 📦 Installation

### Using uv (Recommended)

```bash
# Install from source
git clone <repository-url>
cd parcelextract
uv sync
```

### Using pip

```bash
# Install from source
git clone <repository-url>
cd parcelextract
pip install -e .
```

### Dependencies

ParcelExtract requires Python 3.12+ and the following packages:
- nibabel ≥3.2.0 (neuroimaging file I/O)
- nilearn ≥0.10.0 (neuroimaging data manipulation)  
- numpy ≥1.20.0 (numerical operations)
- pandas ≥1.3.0 (data structuring)
- scipy ≥1.7.0 (scientific computing)
- templateflow ≥0.8.0 (brain atlas management)

## 🔧 Quick Start

### Command-Line Usage

Extract time-series using a TemplateFlow atlas:

```bash
parcelextract \
    --input sub-01_task-rest_bold.nii.gz \
    --atlas Schaefer2018 \
    --desc 400Parcels17Networks \
    --output-dir results/ \
    --strategy mean \
    --verbose
```

Extract using a custom atlas file:

```bash
parcelextract \
    --input sub-01_task-rest_bold.nii.gz \
    --atlas /path/to/custom_atlas.nii.gz \
    --output-dir results/ \
    --strategy median
```

### Python API Usage

```python
from parcelextract.core.extractor import ParcelExtractor

# Initialize extractor with atlas and strategy
extractor = ParcelExtractor(
    atlas='/path/to/atlas.nii.gz',
    strategy='mean'
)

# Extract time-series from 4D image
timeseries = extractor.fit_transform('/path/to/bold.nii.gz')

# timeseries is a 2D array: (n_parcels, n_timepoints)
print(f"Extracted {timeseries.shape[0]} parcels, {timeseries.shape[1]} timepoints")
```

With TemplateFlow atlas:

```python
from parcelextract.atlases.templateflow import TemplateFlowManager
from parcelextract.core.extractor import ParcelExtractor

# Download atlas from TemplateFlow
tf_manager = TemplateFlowManager()
atlas_path = tf_manager.get_atlas(
    'Schaefer2018', 
    space='MNI152NLin2009cAsym',
    desc='400Parcels17Networks'
)

# Use with extractor
extractor = ParcelExtractor(atlas=atlas_path, strategy='pca')
timeseries = extractor.fit_transform('sub-01_task-rest_bold.nii.gz')
```

## 📖 Usage Guide

### Command-Line Interface

The `parcelextract` command provides a complete extraction pipeline:

```bash
parcelextract [OPTIONS]
```

#### Required Arguments

- `--input PATH`: Path to input 4D NIfTI file (.nii or .nii.gz)
- `--atlas ATLAS`: Atlas specification (TemplateFlow name or file path)
- `--output-dir PATH`: Output directory for results

#### Optional Arguments

- `--strategy {mean,median,pca,weighted_mean}`: Signal extraction strategy (default: mean)
- `--space SPACE`: Template space for TemplateFlow atlases (default: MNI152NLin2009cAsym)
- `--desc DESC`: Atlas description/variant (e.g., 400Parcels17Networks)
- `--verbose`: Enable verbose output
- `--help`: Show help message
- `--version`: Show version information

#### Examples

**Basic extraction with TemplateFlow atlas:**
```bash
parcelextract \
    --input sub-01_task-rest_bold.nii.gz \
    --atlas Schaefer2018 \
    --output-dir derivatives/parcelextract/
```

**Specify atlas variant:**
```bash
parcelextract \
    --input sub-01_task-rest_bold.nii.gz \
    --atlas Schaefer2018 \
    --desc 800Parcels7Networks \
    --output-dir derivatives/parcelextract/ \
    --strategy median
```

**Use different template space:**
```bash
parcelextract \
    --input sub-01_task-rest_bold.nii.gz \
    --atlas AAL \
    --space MNI152NLin6Asym \
    --output-dir derivatives/parcelextract/
```

**Custom atlas file:**
```bash
parcelextract \
    --input sub-01_task-rest_bold.nii.gz \
    --atlas /path/to/my_custom_atlas.nii.gz \
    --output-dir derivatives/parcelextract/ \
    --strategy pca
```

### Supported Atlases

ParcelExtract supports atlases from TemplateFlow and custom atlas files:

#### TemplateFlow Atlases
- **Schaefer2018**: Multi-resolution cortical parcellations
- **AAL**: Automated Anatomical Labeling atlas
- **HarvardOxford**: Harvard-Oxford cortical atlas

#### Custom Atlases
- Any 3D NIfTI file (.nii or .nii.gz) with integer labels
- Labels should start from 1 (background = 0 is ignored)
- Must be in the same space as your input data

### Extraction Strategies

1. **Mean** (default): Average signal across all voxels in each parcel
2. **Median**: Median signal across voxels (robust to outliers)
3. **PCA**: First principal component of voxel signals
4. **Weighted Mean**: Probability-weighted average (for probabilistic atlases)

### Output Format

ParcelExtract generates BIDS-compliant output files:

#### Time-series File (TSV)
```
sub-01_task-rest_atlas-Schaefer2018_desc-400Parcels17Networks_timeseries.tsv
```

Content:
```
parcel_0    parcel_1    parcel_2    ...
-0.142      0.256       -0.089      ...
0.031       -0.124      0.198       ...
...         ...         ...         ...
```

#### Metadata File (JSON)
```
sub-01_task-rest_atlas-Schaefer2018_desc-400Parcels17Networks_timeseries.json
```

Content:
```json
{
    "extraction_strategy": "mean",
    "atlas": "Schaefer2018",
    "n_parcels": 400,
    "n_timepoints": 200,
    "input_file": "/path/to/sub-01_task-rest_bold.nii.gz"
}
```

## 🐍 Python API Reference

### ParcelExtractor Class

The main class for time-series extraction:

```python
from parcelextract.core.extractor import ParcelExtractor

extractor = ParcelExtractor(atlas, strategy='mean')
```

#### Parameters
- `atlas` (str or Path): Path to atlas NIfTI file
- `strategy` (str): Extraction strategy ('mean', 'median', 'pca', 'weighted_mean')

#### Methods

**`fit_transform(img_4d)`**
Extract time-series from 4D image.

- **Parameters**: `img_4d` (str, Path, or nibabel image): 4D neuroimaging data
- **Returns**: `numpy.ndarray` (n_parcels × n_timepoints): Extracted time-series

```python
# From file path
timeseries = extractor.fit_transform('data.nii.gz')

# From nibabel image object
import nibabel as nib
img = nib.load('data.nii.gz')
timeseries = extractor.fit_transform(img)
```

### TemplateFlow Integration

Access TemplateFlow atlases programmatically:

```python
from parcelextract.atlases.templateflow import TemplateFlowManager

tf_manager = TemplateFlowManager()

# Download atlas
atlas_path = tf_manager.get_atlas(
    atlas_name='Schaefer2018',
    space='MNI152NLin2009cAsym',
    desc='400Parcels17Networks'
)

# Use with extractor
extractor = ParcelExtractor(atlas=atlas_path)
```

### I/O Utilities

Save results programmatically:

```python
from parcelextract.io.writers import write_timeseries_tsv, write_json_sidecar

# Save time-series to TSV
write_timeseries_tsv(timeseries, 'output_timeseries.tsv')

# Save metadata to JSON
metadata = {
    'extraction_strategy': 'mean',
    'atlas': 'Schaefer2018',
    'n_parcels': timeseries.shape[0],
    'n_timepoints': timeseries.shape[1]
}
write_json_sidecar(metadata, 'output_timeseries.json')
```

## 🔍 Advanced Usage

### Batch Processing

Process multiple subjects using shell scripting:

```bash
#!/bin/bash

# Process all subjects in BIDS dataset
for subject in sub-*; do
    for session in ${subject}/ses-*; do
        for run in ${session}/func/*task-rest*_bold.nii.gz; do
            parcelextract \
                --input "${run}" \
                --atlas Schaefer2018 \
                --desc 400Parcels17Networks \
                --output-dir derivatives/parcelextract/"${subject}"/
        done
    done
done
```

### Integration with Python Workflows

```python
from pathlib import Path
from parcelextract.core.extractor import ParcelExtractor

def process_subject(subject_dir, atlas_path, output_dir):
    """Process all functional runs for a subject."""
    extractor = ParcelExtractor(atlas=atlas_path, strategy='mean')
    
    # Find all BOLD files
    bold_files = subject_dir.glob('**/*_bold.nii.gz')
    
    results = {}
    for bold_file in bold_files:
        print(f"Processing {bold_file.name}...")
        
        # Extract time-series
        timeseries = extractor.fit_transform(bold_file)
        
        # Generate output path
        output_stem = bold_file.stem.replace('.nii', '')
        output_file = output_dir / f"{output_stem}_atlas-custom_timeseries.tsv"
        
        # Save results
        write_timeseries_tsv(timeseries, output_file)
        results[bold_file.name] = timeseries
    
    return results

# Usage
results = process_subject(
    subject_dir=Path('sub-01'),
    atlas_path='custom_atlas.nii.gz',
    output_dir=Path('derivatives/parcelextract/sub-01')
)
```

## 🛠️ Development

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=parcelextract

# Run specific test file
uv run pytest tests/test_extractor.py
```

### Code Quality

```bash
# Format code
uv run ruff format parcelextract

# Lint code
uv run ruff check parcelextract

# Type checking
uv run mypy parcelextract
```

### Project Structure

```
parcelextract/
├── src/parcelextract/          # Main package
│   ├── core/                   # Core extraction logic
│   │   ├── extractor.py       # Main ParcelExtractor class
│   │   ├── strategies.py      # Extraction strategies
│   │   └── validators.py      # Input validation
│   ├── io/                    # Input/output operations
│   │   ├── readers.py         # File reading utilities
│   │   └── writers.py         # Output generation
│   ├── atlases/               # Atlas management
│   │   ├── manager.py         # Atlas loading
│   │   └── templateflow.py    # TemplateFlow integration
│   └── cli/                   # Command-line interface
│       └── main.py            # CLI entry point
├── tests/                     # Test suite
├── docs/                      # Documentation
└── pyproject.toml             # Project configuration
```

## ❓ FAQ

**Q: What input formats are supported?**
A: ParcelExtract supports 4D NIfTI files (.nii and .nii.gz) as input. The data should be preprocessed and in standard space if using TemplateFlow atlases.

**Q: Can I use my own custom atlas?**
A: Yes! Any 3D NIfTI file with integer labels can be used as an atlas. Labels should start from 1 (background = 0 is ignored).

**Q: Which extraction strategy should I use?**
A: 'mean' is the most common choice. Use 'median' for robustness to outliers, 'pca' for dimensionality reduction, or 'weighted_mean' for probabilistic atlases.

**Q: How do I handle missing or corrupted parcels?**
A: ParcelExtract automatically handles empty parcels by returning NaN values. Check your extraction results for NaN values and investigate the corresponding parcels in your atlas.

**Q: Can I extract signals from only specific parcels?**
A: Currently, ParcelExtract extracts signals from all parcels in the atlas. You can post-process the results to select specific parcels of interest.

**Q: Is ParcelExtract BIDS-compliant?**
A: ParcelExtract generates BIDS-inspired output filenames and metadata. While not fully BIDS-compliant, the outputs follow BIDS naming conventions for derivatives.

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for details on:
- Reporting bugs
- Requesting features  
- Submitting pull requests
- Code style and testing requirements

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📚 Citation

If you use ParcelExtract in your research, please cite:

```bibtex
@software{parcelextract2025,
  title={ParcelExtract: Time-series extraction from neuroimaging data},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/parcelextract}
}
```

## 🆘 Support

- **Documentation**: [Link to full documentation]
- **Issues**: Report bugs and feature requests on GitHub Issues
- **Discussions**: Ask questions on GitHub Discussions

---

**ParcelExtract**: Making neuroimaging time-series extraction simple, standardized, and reproducible.