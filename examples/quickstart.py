#!/usr/bin/env python3
"""
ParcelExtract Quick Start Examples

This script demonstrates basic usage of ParcelExtract for extracting
time-series signals from neuroimaging data.

Requirements:
- ParcelExtract installed (see INSTALLATION.md)
- Test data (generated in this script)

Usage:
    python examples/quickstart.py
"""

import numpy as np
import nibabel as nib
from pathlib import Path
import tempfile
import shutil

from parcelextract.core.extractor import ParcelExtractor
from parcelextract.io.writers import write_timeseries_tsv, write_json_sidecar
from parcelextract.atlases.templateflow import TemplateFlowManager


def create_test_data(output_dir: Path):
    """Create synthetic test data for demonstration."""
    print("Creating synthetic test data...")
    
    # Create 4D BOLD data (64x64x32 voxels, 200 timepoints)
    bold_data = np.random.randn(64, 64, 32, 200).astype(np.float32)
    
    # Add some realistic structure
    # Add slow drifts
    time_points = np.arange(200)
    drift = 0.1 * np.sin(2 * np.pi * time_points / 100)
    bold_data += drift[np.newaxis, np.newaxis, np.newaxis, :]
    
    # Add spatial correlation
    for t in range(200):
        # Smooth each timepoint slightly
        from scipy.ndimage import gaussian_filter
        bold_data[:, :, :, t] = gaussian_filter(bold_data[:, :, :, t], sigma=0.5)
    
    # Save BOLD data
    bold_img = nib.Nifti1Image(bold_data, affine=np.eye(4))
    bold_file = output_dir / "sub-01_task-rest_bold.nii.gz"
    nib.save(bold_img, bold_file)
    
    # Create atlas with 10 parcels
    atlas_data = np.zeros((64, 64, 32), dtype=np.int16)
    
    # Define parcel regions (simplified)
    parcels = [
        ((10, 30), (10, 30), (5, 15)),   # Parcel 1: left frontal
        ((35, 55), (10, 30), (5, 15)),   # Parcel 2: right frontal
        ((10, 30), (35, 55), (5, 15)),   # Parcel 3: left parietal
        ((35, 55), (35, 55), (5, 15)),   # Parcel 4: right parietal
        ((20, 45), (20, 45), (16, 26)),  # Parcel 5: central
    ]
    
    for i, ((x1, x2), (y1, y2), (z1, z2)) in enumerate(parcels, 1):
        atlas_data[x1:x2, y1:y2, z1:z2] = i
    
    # Save atlas
    atlas_img = nib.Nifti1Image(atlas_data, affine=np.eye(4))
    atlas_file = output_dir / "test_atlas.nii.gz"
    nib.save(atlas_img, atlas_file)
    
    print(f"✓ Created test data:")
    print(f"  - BOLD: {bold_file} ({bold_data.shape})")
    print(f"  - Atlas: {atlas_file} ({atlas_data.shape}, {len(parcels)} parcels)")
    
    return bold_file, atlas_file


def example_basic_extraction():
    """Basic extraction example with synthetic data."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Time-Series Extraction")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        
        # Create test data
        bold_file, atlas_file = create_test_data(tmp_path)
        
        # Initialize extractor
        print("\nInitializing ParcelExtractor...")
        extractor = ParcelExtractor(
            atlas=str(atlas_file),
            strategy='mean'
        )
        
        # Extract time-series
        print("Extracting time-series...")
        timeseries = extractor.fit_transform(str(bold_file))
        
        print(f"✓ Extraction complete!")
        print(f"  - Shape: {timeseries.shape}")
        print(f"  - {timeseries.shape[0]} parcels")
        print(f"  - {timeseries.shape[1]} timepoints")
        print(f"  - Data range: {timeseries.min():.3f} to {timeseries.max():.3f}")
        
        # Show sample data
        print(f"\nFirst 5 timepoints for first 3 parcels:")
        for parcel in range(min(3, timeseries.shape[0])):
            values = timeseries[parcel, :5]
            print(f"  Parcel {parcel}: {' '.join([f'{v:6.3f}' for v in values])} ...")


def example_different_strategies():
    """Compare different extraction strategies."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Comparing Extraction Strategies")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        
        # Create test data
        bold_file, atlas_file = create_test_data(tmp_path)
        
        strategies = ['mean', 'median', 'pca', 'weighted_mean']
        
        print(f"\nComparing {len(strategies)} extraction strategies...")
        results = {}
        
        for strategy in strategies:
            print(f"  {strategy}...", end=" ")
            
            extractor = ParcelExtractor(
                atlas=str(atlas_file),
                strategy=strategy
            )
            
            timeseries = extractor.fit_transform(str(bold_file))
            results[strategy] = timeseries
            
            print(f"✓ Shape: {timeseries.shape}")
        
        # Compare results
        print(f"\nStrategy comparison for Parcel 0:")
        print("Strategy    First 5 timepoints")
        print("-" * 40)
        
        for strategy in strategies:
            values = results[strategy][0, :5]  # First parcel, first 5 timepoints
            values_str = ' '.join([f'{v:6.3f}' for v in values])
            print(f"{strategy:10s}  {values_str}")
        
        # Statistics
        print(f"\nSignal statistics (all parcels):")
        print("Strategy    Mean ± Std    Min     Max")
        print("-" * 40)
        
        for strategy in strategies:
            data = results[strategy]
            mean_val = np.mean(data)
            std_val = np.std(data)
            min_val = np.min(data)
            max_val = np.max(data)
            print(f"{strategy:10s}  {mean_val:5.3f}±{std_val:5.3f}  {min_val:6.3f}  {max_val:6.3f}")


def example_save_results():
    """Save extraction results to files."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Saving Results to Files")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        output_dir = tmp_path / "results"
        output_dir.mkdir()
        
        # Create test data
        bold_file, atlas_file = create_test_data(tmp_path)
        
        # Extract time-series
        print("\nExtracting time-series with PCA strategy...")
        extractor = ParcelExtractor(
            atlas=str(atlas_file),
            strategy='pca'
        )
        
        timeseries = extractor.fit_transform(str(bold_file))
        
        # Generate output filenames
        input_stem = bold_file.stem.replace('.nii', '')  # Remove .nii from .nii.gz
        output_stem = f"{input_stem}_atlas-testAtlas_timeseries"
        
        tsv_file = output_dir / f"{output_stem}.tsv"
        json_file = output_dir / f"{output_stem}.json"
        
        # Save TSV file
        print(f"Saving time-series to: {tsv_file}")
        write_timeseries_tsv(timeseries, tsv_file)
        
        # Create metadata
        metadata = {
            'extraction_strategy': 'pca',
            'atlas': str(atlas_file),
            'n_parcels': timeseries.shape[0],
            'n_timepoints': timeseries.shape[1],
            'input_file': str(bold_file),
            'description': 'Example extraction using synthetic data'
        }
        
        # Save JSON file
        print(f"Saving metadata to: {json_file}")
        write_json_sidecar(metadata, json_file)
        
        # Verify files
        print(f"\n✓ Files created successfully:")
        print(f"  - TSV file: {tsv_file.stat().st_size} bytes")
        print(f"  - JSON file: {json_file.stat().st_size} bytes")
        
        # Show file contents (first few lines)
        print(f"\nTSV file preview (first 3 lines):")
        with open(tsv_file, 'r') as f:
            for i, line in enumerate(f):
                if i >= 3:
                    break
                print(f"  {line.rstrip()}")
        
        print(f"\nJSON metadata:")
        import json
        with open(json_file, 'r') as f:
            metadata_loaded = json.load(f)
        
        for key, value in metadata_loaded.items():
            print(f"  {key}: {value}")


def example_templateflow():
    """Example using TemplateFlow atlas (requires internet connection)."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Using TemplateFlow Atlas")
    print("="*60)
    
    try:
        print("Testing TemplateFlow connection...")
        
        # Initialize TemplateFlow manager
        tf_manager = TemplateFlowManager()
        
        print("✓ TemplateFlow initialized")
        print("\nAttempting to download Schaefer2018 atlas...")
        print("(This may take a few moments on first run)")
        
        # Download atlas
        atlas_path = tf_manager.get_atlas(
            'Schaefer2018',
            space='MNI152NLin2009cAsym',
            desc='400Parcels17Networks'
        )
        
        print(f"✓ Atlas downloaded to: {atlas_path}")
        
        # Load atlas to check properties
        atlas_img = nib.load(atlas_path)
        atlas_data = atlas_img.get_fdata()
        n_parcels = int(atlas_data.max())
        
        print(f"  - Shape: {atlas_data.shape}")
        print(f"  - Number of parcels: {n_parcels}")
        print(f"  - File size: {Path(atlas_path).stat().st_size / 1024 / 1024:.1f} MB")
        
        print(f"\n✓ TemplateFlow integration working correctly!")
        print(f"  Use this atlas with:")
        print(f"  extractor = ParcelExtractor(atlas='{atlas_path}')")
        
    except ImportError:
        print("⚠️  TemplateFlow not available (not installed)")
        print("   Install with: uv add templateflow")
    except Exception as e:
        print(f"⚠️  TemplateFlow error: {e}")
        print("   This may be due to network connectivity issues")
        print("   TemplateFlow requires internet connection for first download")


def example_cli_usage():
    """Show CLI usage examples."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Command-Line Interface Usage")
    print("="*60)
    
    print("ParcelExtract provides a command-line interface for batch processing:")
    print()
    
    examples = [
        {
            "title": "Basic extraction with custom atlas",
            "command": """parcelextract \\
    --input sub-01_task-rest_bold.nii.gz \\
    --atlas custom_atlas.nii.gz \\
    --output-dir results/ \\
    --verbose"""
        },
        {
            "title": "TemplateFlow atlas with specific variant",
            "command": """parcelextract \\
    --input sub-01_task-rest_bold.nii.gz \\
    --atlas Schaefer2018 \\
    --desc 400Parcels17Networks \\
    --output-dir results/ \\
    --strategy mean"""
        },
        {
            "title": "Different template space",
            "command": """parcelextract \\
    --input sub-01_task-rest_bold.nii.gz \\
    --atlas AAL \\
    --space MNI152NLin6Asym \\
    --output-dir results/ \\
    --strategy median"""
        },
        {
            "title": "PCA extraction strategy",
            "command": """parcelextract \\
    --input sub-01_task-rest_bold.nii.gz \\
    --atlas HarvardOxford \\
    --output-dir results/ \\
    --strategy pca \\
    --verbose"""
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example['title']}:")
        print(f"   {example['command']}")
        print()
    
    print("For help with all options:")
    print("   parcelextract --help")
    
    print("\nOutput files follow BIDS naming conventions:")
    print("   sub-01_task-rest_atlas-Schaefer2018_desc-400Parcels17Networks_timeseries.tsv")
    print("   sub-01_task-rest_atlas-Schaefer2018_desc-400Parcels17Networks_timeseries.json")


def main():
    """Run all examples."""
    print("ParcelExtract Quick Start Examples")
    print("=" * 60)
    print()
    print("This script demonstrates key features of ParcelExtract:")
    print("1. Basic time-series extraction")
    print("2. Different extraction strategies")  
    print("3. Saving results to files")
    print("4. TemplateFlow atlas integration")
    print("5. Command-line usage examples")
    
    # Run examples
    example_basic_extraction()
    example_different_strategies()
    example_save_results()
    example_templateflow()
    example_cli_usage()
    
    print("\n" + "="*60)
    print("Examples completed successfully!")
    print("="*60)
    print("\nNext steps:")
    print("- Try ParcelExtract with your own data")
    print("- Explore different extraction strategies")
    print("- Use TemplateFlow atlases for standard analyses")
    print("- Check the full documentation in README.md")
    print("\nFor questions or issues:")
    print("- GitHub Issues: https://github.com/yourusername/parcelextract/issues")
    print("- Documentation: README.md and CONTRIBUTING.md")


if __name__ == '__main__':
    main()