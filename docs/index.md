---
layout: default
title: Home
---

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

## 📖 Documentation

- [Installation Guide](installation)
- [Usage Guide](#usage-guide)
- [API Reference](#python-api-reference)
- [Contributing](contributing)
- [Development Guide](development)
- [Changelog](changelog)

## Supported Atlases

ParcelExtract supports atlases from TemplateFlow and custom atlas files:

### TemplateFlow Atlases
- **Schaefer2018**: Multi-resolution cortical parcellations
- **AAL**: Automated Anatomical Labeling atlas
- **HarvardOxford**: Harvard-Oxford cortical atlas

### Custom Atlases
- Any 3D NIfTI file (.nii or .nii.gz) with integer labels
- Labels should start from 1 (background = 0 is ignored)
- Must be in the same space as your input data

## Extraction Strategies

1. **Mean** (default): Average signal across all voxels in each parcel
2. **Median**: Median signal across voxels (robust to outliers)
3. **PCA**: First principal component of voxel signals
4. **Weighted Mean**: Probability-weighted average (for probabilistic atlases)

## Output Format

ParcelExtract generates BIDS-compliant output files:

### Time-series File (TSV)
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

### Metadata File (JSON)
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

## 🤝 Contributing

We welcome contributions! Please see our [contributing guidelines](contributing) for details on:
- Reporting bugs
- Requesting features  
- Submitting pull requests
- Code style and testing requirements

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/yourusername/parcelextract/blob/main/LICENSE) file for details.

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

- **Documentation**: [Full documentation](/)
- **Issues**: [Report bugs and feature requests](https://github.com/yourusername/parcelextract/issues)
- **Discussions**: [Ask questions](https://github.com/yourusername/parcelextract/discussions)

---

**ParcelExtract**: Making neuroimaging time-series extraction simple, standardized, and reproducible.