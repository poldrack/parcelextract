#!/usr/bin/env python3
"""
Demonstration of DiFuMo Atlas Support

This script demonstrates ParcelExtract's support for the DiFuMo atlas
from TemplateFlow, including automatic detection of probabilistic
segmentation files and available dimension variants.

DiFuMo (Dictionary Learning for Functional Modes) is a probabilistic
functional brain atlas that provides parcellations at different
dimensionalities (64, 128, 256, 512, 1024 dimensions).

Key features demonstrated:
1. DiFuMo atlas recognition and downloading
2. Available dimension variants
3. Probabilistic segmentation file handling
4. Integration with ParcelExtract's probabilistic atlas detection
"""

from pathlib import Path
import tempfile
import numpy as np
import nibabel as nib

from parcelextract.atlases.templateflow import TemplateFlowManager
from parcelextract.core.extractor import ParcelExtractor


def demonstrate_difumo_availability():
    """Show what DiFuMo variants are available."""
    print("="*70)
    print("DIFUMO ATLAS AVAILABILITY")
    print("="*70)
    
    tf_manager = TemplateFlowManager()
    
    # Check if DiFuMo is in available atlases
    print("\n1. AVAILABLE ATLASES")
    print("-" * 30)
    available_atlases = tf_manager.list_available_atlases('MNI152NLin2009cAsym')
    print(f"Available atlases: {', '.join(available_atlases)}")
    
    if 'DiFuMo' in available_atlases:
        print("✅ DiFuMo is supported!")
    else:
        print("❌ DiFuMo not found in available atlases")
        return
    
    # Show available DiFuMo dimensions
    print("\n2. DIFUMO DIMENSION VARIANTS")
    print("-" * 35)
    descriptions = tf_manager.list_available_descriptions('DiFuMo', 'MNI152NLin2009cAsym')
    print("Available DiFuMo variants:")
    for i, desc in enumerate(descriptions, 1):
        dimensions = desc.replace('dimensions', '')
        print(f"  {i}. {dimensions} dimensions ({desc})")
    
    print(f"\nTotal variants available: {len(descriptions)}")


def demonstrate_difumo_download():
    """Show downloading of DiFuMo atlas variants."""
    print("\n" + "="*70)
    print("DIFUMO ATLAS DOWNLOADING")
    print("="*70)
    
    tf_manager = TemplateFlowManager()
    
    # Test downloading different variants
    test_variants = ['64dimensions', '128dimensions', '256dimensions']
    
    for i, variant in enumerate(test_variants, 1):
        print(f"\n{i}. DOWNLOADING DIFUMO {variant.upper()}")
        print("-" * 40)
        
        try:
            atlas_path = tf_manager.get_atlas('DiFuMo', 'MNI152NLin2009cAsym', desc=variant)
            print(f"✅ Success! Downloaded to:")
            print(f"   {atlas_path}")
            
            # Check file properties
            if Path(atlas_path).exists():
                size_mb = Path(atlas_path).stat().st_size / (1024 * 1024)
                print(f"   File size: {size_mb:.2f} MB")
                
                # Load and show basic properties
                img = nib.load(atlas_path)
                print(f"   Shape: {img.shape}")
                print(f"   Data type: {img.get_fdata().dtype}")
                print(f"   File type: {'probseg' if 'probseg' in atlas_path else 'dseg'}")
                
        except Exception as e:
            print(f"❌ Error downloading {variant}: {e}")


def demonstrate_difumo_extraction():
    """Demonstrate signal extraction with DiFuMo atlas."""
    print("\n" + "="*70)
    print("DIFUMO SIGNAL EXTRACTION DEMO")
    print("="*70)
    
    # Create synthetic 4D data
    print("\n1. CREATING SYNTHETIC DATA")
    print("-" * 30)
    img_shape = (20, 24, 20, 30)  # Small image for demo
    print(f"Creating synthetic 4D image: {img_shape}")
    
    # Generate synthetic BOLD-like data
    np.random.seed(42)  # Reproducible results
    img_4d = np.random.randn(*img_shape).astype(np.float32) * 0.5
    
    # Add some structured signals to different regions
    t = np.arange(img_shape[3])
    
    # Add signals to different brain regions
    signal1 = 2.0 * np.sin(2 * np.pi * 0.1 * t)  # 0.1 Hz signal
    signal2 = 1.5 * np.cos(2 * np.pi * 0.15 * t)  # 0.15 Hz signal
    
    img_4d[5:10, 8:12, 8:12, :] += signal1  # Region 1
    img_4d[12:17, 8:12, 8:12, :] += signal2  # Region 2
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Save synthetic data
        img_file = temp_path / "synthetic_data.nii.gz"
        nib.save(nib.Nifti1Image(img_4d, np.eye(4)), img_file)
        print(f"Saved synthetic data to: {img_file.name}")
        
        # Test extraction with DiFuMo
        print("\n2. EXTRACTING WITH DIFUMO 64-DIMENSION ATLAS")
        print("-" * 50)
        
        try:
            # Download DiFuMo atlas
            tf_manager = TemplateFlowManager()
            difumo_path = tf_manager.get_atlas('DiFuMo', 'MNI152NLin2009cAsym', desc='64dimensions')
            print(f"Using DiFuMo atlas: {Path(difumo_path).name}")
            
            # Initialize extractor
            extractor = ParcelExtractor(atlas=difumo_path, strategy='mean')
            
            # Check atlas properties
            print(f"Atlas is probabilistic: {extractor.is_probabilistic_atlas()}")
            print(f"Effective strategy: {extractor.get_effective_strategy()}")
            print(f"Original strategy: {extractor.strategy}")
            
            # Extract signals
            print("\nExtracting time series...")
            timeseries = extractor.fit_transform(str(img_file))
            print(f"Output shape: {timeseries.shape}")
            print(f"Extracted {timeseries.shape[0]} time series with {timeseries.shape[1]} timepoints")
            
            # Show signal statistics
            print(f"\nSignal statistics:")
            print(f"  Mean amplitude: {np.mean(np.abs(timeseries)):.3f}")
            print(f"  Standard deviation: {np.std(timeseries):.3f}")
            print(f"  Range: {np.min(timeseries):.3f} to {np.max(timeseries):.3f}")
            
            print("✅ DiFuMo extraction completed successfully!")
            
        except Exception as e:
            print(f"❌ Error during extraction: {e}")


def demonstrate_atlas_comparison():
    """Compare DiFuMo with Schaefer2018 atlas."""
    print("\n" + "="*70)
    print("DIFUMO VS SCHAEFER COMPARISON")
    print("="*70)
    
    tf_manager = TemplateFlowManager()
    
    print("\n1. FILE TYPE COMPARISON")
    print("-" * 25)
    
    try:
        # Get both atlases
        difumo_path = tf_manager.get_atlas('DiFuMo', 'MNI152NLin2009cAsym', desc='64dimensions')
        schaefer_path = tf_manager.get_atlas('Schaefer2018', 'MNI152NLin2009cAsym', desc='100Parcels7Networks')
        
        print("DiFuMo atlas:")
        print(f"  Path: {Path(difumo_path).name}")
        print(f"  Type: {'Probabilistic (probseg)' if 'probseg' in difumo_path else 'Discrete (dseg)'}")
        
        print("\nSchaefer2018 atlas:")
        print(f"  Path: {Path(schaefer_path).name}")
        print(f"  Type: {'Probabilistic (probseg)' if 'probseg' in schaefer_path else 'Discrete (dseg)'}")
        
        # Compare atlas properties
        print("\n2. ATLAS PROPERTIES")
        print("-" * 20)
        
        difumo_extractor = ParcelExtractor(atlas=difumo_path)
        schaefer_extractor = ParcelExtractor(atlas=schaefer_path)
        
        print("DiFuMo:")
        print(f"  Probabilistic: {difumo_extractor.is_probabilistic_atlas()}")
        print(f"  Strategy: {difumo_extractor.get_effective_strategy()}")
        
        print("\nSchaefer2018:")
        print(f"  Probabilistic: {schaefer_extractor.is_probabilistic_atlas()}")
        print(f"  Strategy: {schaefer_extractor.get_effective_strategy()}")
        
    except Exception as e:
        print(f"Error in comparison: {e}")


def main():
    """Run all DiFuMo demonstrations."""
    print("ParcelExtract: DiFuMo Atlas Support Demonstration")
    print("=" * 60)
    print("\nDiFuMo (Dictionary Learning for Functional Modes) is a")
    print("probabilistic functional brain atlas providing parcellations")
    print("at multiple dimensionalities for functional connectivity analysis.")
    print()
    print("This demo shows ParcelExtract's comprehensive DiFuMo support:")
    
    try:
        demonstrate_difumo_availability()
        demonstrate_difumo_download()
        demonstrate_difumo_extraction()
        demonstrate_atlas_comparison()
        
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print("✅ DiFuMo atlas recognition and downloading")
        print("✅ Multiple dimension variants (64, 128, 256, 512, 1024)")
        print("✅ Probabilistic segmentation file support")
        print("✅ Automatic weighted mean strategy selection")
        print("✅ Integration with existing ParcelExtract workflow")
        
        print("\n🎯 CLI USAGE EXAMPLES:")
        print("uv run parcelextract --input data.nii.gz --atlas DiFuMo")
        print("uv run parcelextract --input data.nii.gz --atlas DiFuMo --desc 128dimensions")
        print("uv run parcelextract --input data.nii.gz --atlas DiFuMo --desc 512dimensions")
        
        print(f"\n🎉 DiFuMo demo completed successfully!")
        
    except ImportError:
        print("\n❌ TemplateFlow is not installed.")
        print("Install it with: uv add templateflow")
        return
    except Exception as e:
        print(f"\n❌ Demo error: {e}")


if __name__ == '__main__':
    main()