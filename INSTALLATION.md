# Installation Guide

This guide provides detailed installation instructions for ParcelExtract.

## Requirements

- **Python**: 3.12 or higher
- **Operating System**: Linux, macOS, or Windows
- **Memory**: Minimum 4GB RAM (8GB+ recommended for large datasets)
- **Storage**: ~1GB for dependencies, additional space for TemplateFlow atlases

## Installation Methods

### Method 1: Using uv (Recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package manager that handles dependencies efficiently.

#### 1. Install uv

```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Alternative: using pip
pip install uv
```

#### 2. Clone and Install ParcelExtract

```bash
# Clone the repository
git clone https://github.com/yourusername/parcelextract.git
cd parcelextract

# Install with development dependencies
uv sync --dev

# Or install for production use only
uv sync
```

#### 3. Verify Installation

```bash
# Test the CLI
uv run parcelextract --help

# Test the Python API
uv run python -c "from parcelextract.core.extractor import ParcelExtractor; print('✓ ParcelExtract installed successfully')"
```

### Method 2: Using pip

#### 1. Create Virtual Environment

```bash
# Create virtual environment
python -m venv parcelextract-env

# Activate (Linux/macOS)
source parcelextract-env/bin/activate

# Activate (Windows)
parcelextract-env\Scripts\activate
```

#### 2. Install ParcelExtract

```bash
# Clone and install
git clone https://github.com/yourusername/parcelextract.git
cd parcelextract
pip install -e .

# Or install development dependencies
pip install -e ".[dev]"
```

#### 3. Verify Installation

```bash
# Test the CLI
parcelextract --help

# Test the Python API  
python -c "from parcelextract.core.extractor import ParcelExtractor; print('✓ ParcelExtract installed successfully')"
```

### Method 3: Docker (Coming Soon)

A Docker container will be available for easy deployment:

```bash
# Pull the image
docker pull parcelextract/parcelextract:latest

# Run with mounted data
docker run -v /path/to/data:/data parcelextract/parcelextract:latest \
    parcelextract --input /data/input.nii.gz --atlas Schaefer2018 --output-dir /data/results
```

## Dependency Details

ParcelExtract requires the following core dependencies:

```toml
[dependencies]
nibabel = ">=3.2.0"      # Neuroimaging file I/O
nilearn = ">=0.10.0"     # Neuroimaging data manipulation  
numpy = ">=1.20.0"       # Numerical operations
pandas = ">=1.3.0"       # Data structuring
scipy = ">=1.7.0"        # Scientific computing
templateflow = ">=0.8.0" # Brain atlas management
```

Development dependencies (optional):
```toml
[dev-dependencies]
pytest = ">=7.0.0"       # Testing framework
pytest-cov = ">=4.0.0"   # Coverage reporting
ruff = ">=0.6.0"         # Linting and formatting
mypy = ">=1.0.0"         # Type checking
```

## TemplateFlow Setup

ParcelExtract uses TemplateFlow for downloading standard brain atlases. TemplateFlow will be automatically configured on first use.

### Manual TemplateFlow Configuration

```bash
# Set TemplateFlow home directory (optional)
export TEMPLATEFLOW_HOME=/path/to/templateflow

# Pre-download common atlases (optional)
python -c "
import templateflow.api as tflow
tflow.get(template='MNI152NLin2009cAsym', atlas='Schaefer2018', desc='400Parcels17Networks', resolution=2)
tflow.get(template='MNI152NLin2009cAsym', atlas='AAL', resolution=2)
"
```

### TemplateFlow Storage Requirements

- Schaefer2018 (400 parcels): ~50MB
- Schaefer2018 (800 parcels): ~100MB  
- AAL atlas: ~10MB
- Harvard-Oxford atlas: ~20MB

## Troubleshooting

### Common Installation Issues

#### Issue: "Command not found: uv"
```bash
# Solution: Add uv to PATH
echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

#### Issue: "Permission denied" during installation
```bash
# Solution: Install in user directory
pip install --user -e .

# Or use virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -e .
```

#### Issue: "Package conflicts" with existing packages
```bash
# Solution: Create clean environment with uv
uv venv parcelextract-clean
source parcelextract-clean/bin/activate  # or parcelextract-clean\Scripts\activate on Windows
uv sync
```

#### Issue: TemplateFlow download errors
```bash
# Solution: Check internet connection and try manual download
python -c "
import templateflow.api as tflow
print('TemplateFlow version:', tflow.__version__)
# Test basic download
tflow.get(template='MNI152NLin2009cAsym', resolution=2, desc='brain')
"
```

#### Issue: "Module not found" errors
```bash
# Solution: Verify installation
pip list | grep -E "(nibabel|nilearn|templateflow|numpy|pandas|scipy)"

# If missing, reinstall
pip install --force-reinstall -e .
```

### Platform-Specific Notes

#### macOS
- May need to install Xcode command line tools: `xcode-select --install`
- For Apple Silicon (M1/M2), ensure you're using Python built for ARM64

#### Windows
- Use PowerShell or Command Prompt as Administrator for system-wide installation
- Git Bash is recommended for running bash scripts
- Long path support may need to be enabled for TemplateFlow

#### Linux
- Most distributions work out of the box
- On older systems, may need to compile some dependencies from source
- Check that Python development headers are installed: `python3-dev` (Debian/Ubuntu) or `python3-devel` (RHEL/CentOS)

## Performance Optimization

### For Large Datasets

```bash
# Increase memory limits if needed
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Use faster linear algebra libraries
pip install numpy[mkl]  # Intel MKL (if available)
# or
pip install openblas    # OpenBLAS alternative
```

### For Batch Processing

Consider using a workflow manager like Snakemake or Nextflow:

```python
# Snakemake example rule
rule parcelextract:
    input:
        bold="data/{subject}_task-{task}_bold.nii.gz"
    output:
        tsv="results/{subject}_task-{task}_atlas-{atlas}_timeseries.tsv",
        json="results/{subject}_task-{task}_atlas-{atlas}_timeseries.json"
    shell:
        "parcelextract --input {input.bold} --atlas {wildcards.atlas} --output-dir results/"
```

## Verification Tests

Run these tests to ensure everything is working correctly:

### Basic Functionality Test

```bash
# Create test data (requires nibabel)
python -c "
import numpy as np
import nibabel as nib
from pathlib import Path

# Create synthetic 4D data
data_4d = np.random.randn(20, 20, 20, 100).astype(np.float32)
img_4d = nib.Nifti1Image(data_4d, affine=np.eye(4))
nib.save(img_4d, 'test_bold.nii.gz')

# Create synthetic atlas
atlas_data = np.random.randint(0, 10, (20, 20, 20)).astype(np.int16)
atlas_img = nib.Nifti1Image(atlas_data, affine=np.eye(4))
nib.save(atlas_img, 'test_atlas.nii.gz')

print('Test data created: test_bold.nii.gz, test_atlas.nii.gz')
"

# Test extraction
parcelextract --input test_bold.nii.gz --atlas test_atlas.nii.gz --output-dir test_results/ --verbose

# Cleanup
rm test_bold.nii.gz test_atlas.nii.gz
rm -rf test_results/
```

### TemplateFlow Integration Test

```bash
# Test TemplateFlow download
python -c "
from parcelextract.atlases.templateflow import TemplateFlowManager

# Test TemplateFlow connection
tf_manager = TemplateFlowManager()
atlas_path = tf_manager.get_atlas('Schaefer2018', 'MNI152NLin2009cAsym')
print(f'✓ TemplateFlow atlas downloaded: {atlas_path}')
"
```

### Performance Test

```bash
# Run test suite to verify installation
uv run pytest tests/ -v

# Check coverage
uv run pytest --cov=parcelextract tests/
```

## Getting Help

If you encounter issues not covered here:

1. Check the [FAQ](README.md#-faq) section
2. Search existing [GitHub Issues](https://github.com/yourusername/parcelextract/issues)
3. Create a new issue with:
   - Your operating system and Python version
   - Complete error message
   - Steps to reproduce the problem
   - Output of `pip list` or `uv tree`

## Next Steps

Once installed, see the [README.md](README.md) for usage examples and the full API reference.

For developers, see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on contributing to the project.