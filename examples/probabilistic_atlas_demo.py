#!/usr/bin/env python3
"""
Demonstration of Automatic Probabilistic Atlas Detection

This script demonstrates ParcelExtract's ability to automatically detect
whether an atlas contains discrete integer labels or probabilistic/continuous
values, and automatically apply the appropriate extraction strategy.

Key features demonstrated:
1. Automatic detection of atlas type (discrete vs probabilistic)
2. Automatic strategy selection (weighted_mean for probabilistic atlases)
3. Comparison of extraction results between atlas types
4. Preservation of user's original strategy choice for reference
"""

import numpy as np
import nibabel as nib
from pathlib import Path
import tempfile

from parcelextract.core.extractor import ParcelExtractor


def create_test_data():
    """Create both discrete and probabilistic atlases with known signals."""
    # Create 4D BOLD-like data
    img_shape = (20, 20, 20, 50)
    img_4d = np.random.randn(*img_shape).astype(np.float32) * 0.2  # Background noise
    
    # Add structured signals to specific regions
    t = np.arange(50, dtype=np.float32)
    
    # Region 1: Strong oscillatory signal
    signal1 = 2.0 * np.sin(2 * np.pi * 0.1 * t) + 0.5 * np.cos(2 * np.pi * 0.3 * t)
    img_4d[5:10, 5:10, 8:12, :] += signal1
    
    # Region 2: Different oscillatory signal
    signal2 = 1.5 * np.cos(2 * np.pi * 0.15 * t) + 0.8 * np.sin(2 * np.pi * 0.25 * t)
    img_4d[12:17, 5:10, 8:12, :] += signal2
    
    # Region 3: Slower drift signal
    signal3 = 1.0 * np.sin(2 * np.pi * 0.05 * t) + 0.3 * t / 50
    img_4d[5:10, 12:17, 8:12, :] += signal3
    
    return img_4d, signal1, signal2, signal3


def create_discrete_atlas():
    """Create a discrete integer atlas."""
    atlas_shape = (20, 20, 20)
    atlas_data = np.zeros(atlas_shape, dtype=np.int16)
    
    # Define discrete parcels
    atlas_data[5:10, 5:10, 8:12] = 1    # Parcel 1
    atlas_data[12:17, 5:10, 8:12] = 2   # Parcel 2
    atlas_data[5:10, 12:17, 8:12] = 3   # Parcel 3
    
    return atlas_data


def create_probabilistic_atlas():
    """Create a probabilistic atlas with continuous values."""
    atlas_shape = (20, 20, 20)
    atlas_data = np.zeros(atlas_shape, dtype=np.float32)
    
    # Create probabilistic "parcels" with Gaussian-like weights
    centers = [(7, 7, 10), (14, 7, 10), (7, 14, 10)]  # Same centers as discrete parcels
    
    for center_idx, (cx, cy, cz) in enumerate(centers):
        for x in range(20):
            for y in range(20):
                for z in range(20):
                    # Distance from center
                    dist = np.sqrt((x - cx)**2 + (y - cy)**2 + (z - cz)**2)
                    
                    # Gaussian-like probability (different width for each center)
                    width = 2.0 + center_idx * 0.5  # Varying widths
                    prob = np.exp(-dist**2 / (2 * width**2))
                    
                    # Only keep significant probabilities
                    if prob > 0.1:
                        # Combine probabilities (overlapping regions possible)
                        atlas_data[x, y, z] = max(atlas_data[x, y, z], prob)
    
    return atlas_data


def demonstrate_automatic_detection():
    """Demonstrate automatic atlas type detection and strategy selection."""
    print("="*70)
    print("PROBABILISTIC ATLAS DETECTION DEMONSTRATION")
    print("="*70)
    
    # Create test data
    img_4d, signal1, signal2, signal3 = create_test_data()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Save 4D image
        img_file = temp_path / "test_data.nii.gz"
        nib.save(nib.Nifti1Image(img_4d, np.eye(4)), img_file)
        
        # Test 1: Discrete Atlas
        print("\n1. DISCRETE ATLAS TEST")
        print("-" * 30)
        
        discrete_atlas = create_discrete_atlas()
        discrete_file = temp_path / "discrete_atlas.nii.gz"
        nib.save(nib.Nifti1Image(discrete_atlas, np.eye(4)), discrete_file)
        
        # Initialize with mean strategy
        extractor_discrete = ParcelExtractor(atlas=str(discrete_file), strategy='mean')
        
        print(f"Original strategy: {extractor_discrete.strategy}")
        print(f"Is probabilistic: {extractor_discrete.is_probabilistic_atlas()}")
        print(f"Effective strategy: {extractor_discrete.get_effective_strategy()}")
        
        # Extract signals
        timeseries_discrete = extractor_discrete.fit_transform(str(img_file))
        print(f"Output shape: {timeseries_discrete.shape} (3 parcels, 50 timepoints)")
        
        # Test 2: Probabilistic Atlas
        print("\n2. PROBABILISTIC ATLAS TEST")
        print("-" * 35)
        
        prob_atlas = create_probabilistic_atlas()
        prob_file = temp_path / "probabilistic_atlas.nii.gz"
        nib.save(nib.Nifti1Image(prob_atlas, np.eye(4)), prob_file)
        
        # Initialize with same mean strategy
        extractor_prob = ParcelExtractor(atlas=str(prob_file), strategy='mean')
        
        print(f"Original strategy: {extractor_prob.strategy}")
        print(f"Is probabilistic: {extractor_prob.is_probabilistic_atlas()}")
        print(f"Effective strategy: {extractor_prob.get_effective_strategy()}")
        
        # Extract signals
        timeseries_prob = extractor_prob.fit_transform(str(img_file))
        print(f"Output shape: {timeseries_prob.shape} (1 weighted average, 50 timepoints)")
        
        # Test 3: Strategy Override Demonstration
        print("\n3. STRATEGY OVERRIDE DEMONSTRATION")
        print("-" * 40)
        
        test_strategies = ['median', 'pca']
        
        for strategy in test_strategies:
            extractor_test = ParcelExtractor(atlas=str(prob_file), strategy=strategy)
            print(f"\nRequested: {strategy}")
            print(f"  -> Detected probabilistic: {extractor_test.is_probabilistic_atlas()}")
            print(f"  -> Effective strategy: {extractor_test.get_effective_strategy()}")
            print(f"  -> Original preserved: {extractor_test.strategy}")
        
        # Test 4: Signal Quality Comparison
        print("\n4. SIGNAL QUALITY COMPARISON")
        print("-" * 35)
        
        # Compare extracted signals
        print(f"\nDiscrete atlas results:")
        for i in range(3):
            variance = np.var(timeseries_discrete[i, :])
            mean_amplitude = np.mean(np.abs(timeseries_discrete[i, :]))
            print(f"  Parcel {i+1}: variance={variance:.3f}, mean_amplitude={mean_amplitude:.3f}")
        
        print(f"\nProbabilistic atlas result:")
        variance_prob = np.var(timeseries_prob[0, :])
        mean_amplitude_prob = np.mean(np.abs(timeseries_prob[0, :]))
        print(f"  Weighted avg: variance={variance_prob:.3f}, mean_amplitude={mean_amplitude_prob:.3f}")
        
        # Test 5: Atlas Statistics
        print("\n5. ATLAS STATISTICS")
        print("-" * 25)
        
        print("Discrete atlas:")
        unique_discrete = np.unique(discrete_atlas)
        print(f"  Unique values: {unique_discrete}")
        print(f"  Value range: {discrete_atlas.min():.3f} to {discrete_atlas.max():.3f}")
        print(f"  Data type: {discrete_atlas.dtype}")
        
        print("\nProbabilistic atlas:")
        non_zero_prob = prob_atlas[prob_atlas > 0]
        unique_prob = np.unique(non_zero_prob)
        print(f"  Unique non-zero values: {len(unique_prob)} values")
        print(f"  Value range: {prob_atlas.min():.3f} to {prob_atlas.max():.3f}")
        print(f"  Data type: {prob_atlas.dtype}")
        print(f"  Sample values: {unique_prob[:5]}")  # Show first 5 values
        
        return timeseries_discrete, timeseries_prob


def demonstrate_edge_cases():
    """Demonstrate edge cases in atlas detection."""
    print("\n" + "="*70)
    print("EDGE CASES DEMONSTRATION")
    print("="*70)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create simple 4D data for testing
        simple_4d = np.random.randn(8, 8, 8, 20).astype(np.float32)
        img_file = temp_path / "simple_data.nii.gz"
        nib.save(nib.Nifti1Image(simple_4d, np.eye(4)), img_file)
        
        test_cases = [
            {
                'name': 'Integer values as float',
                'data': np.array([[[0, 1.0, 2.0], [0, 1.0, 0], [2.0, 0, 0]]]).astype(np.float32),
                'expected_discrete': True
            },
            {
                'name': 'Near-integer values',
                'data': np.array([[[0, 1.0001, 2.0002], [0, 1.0, 0], [2.0, 0, 0]]]).astype(np.float32),
                'expected_discrete': False  # Very close but not exactly integers
            },
            {
                'name': 'True probabilities (0-1)',
                'data': np.random.rand(3, 3, 3).astype(np.float32) * 0.9,
                'expected_discrete': False
            },
            {
                'name': 'Few unique float values',
                'data': np.array([[[0, 0.5, 1.0], [0, 0.5, 0], [1.0, 0, 0]]]).astype(np.float32),
                'expected_discrete': False  # Not integers, so probabilistic
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n{i}. {test_case['name'].upper()}")
            print("-" * (len(test_case['name']) + 3))
            
            # Create padded atlas to match image dimensions
            atlas_data = np.zeros((8, 8, 8), dtype=np.float32)
            atlas_data[:3, :3, :3] = test_case['data']
            
            atlas_file = temp_path / f"edge_case_{i}.nii.gz"
            nib.save(nib.Nifti1Image(atlas_data, np.eye(4)), atlas_file)
            
            try:
                extractor = ParcelExtractor(atlas=str(atlas_file), strategy='mean')
                is_prob = extractor.is_probabilistic_atlas()
                effective_strategy = extractor.get_effective_strategy()
                
                print(f"  Detected as probabilistic: {is_prob}")
                print(f"  Effective strategy: {effective_strategy}")
                print(f"  Expected discrete: {test_case['expected_discrete']}")
                
                if is_prob == (not test_case['expected_discrete']):
                    print("  ✅ Detection matches expectation")
                else:
                    print("  ⚠️  Detection differs from expectation")
                
                # Test extraction
                timeseries = extractor.fit_transform(str(img_file))
                print(f"  Output shape: {timeseries.shape}")
                
            except Exception as e:
                print(f"  ❌ Error: {e}")


def main():
    """Run all demonstrations."""
    print("ParcelExtract: Automatic Probabilistic Atlas Detection")
    print("=" * 60)
    print("\nThis demonstration shows how ParcelExtract automatically:")
    print("1. Detects whether an atlas contains discrete or probabilistic values")
    print("2. Selects the appropriate extraction strategy")
    print("3. Handles edge cases in atlas detection")
    print("4. Preserves user's original strategy preference for reference")
    
    # Main demonstration
    discrete_results, prob_results = demonstrate_automatic_detection()
    
    # Edge cases
    demonstrate_edge_cases()
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("\nKey Benefits:")
    print("• Automatic detection eliminates user guesswork about atlas type")
    print("• Optimal extraction strategy is selected automatically")
    print("• Probabilistic atlases get weighted extraction without user intervention")
    print("• Discrete atlases continue to work exactly as before")
    print("• Original strategy choice is preserved for reference/debugging")
    
    print("\nDetection Criteria:")
    print("• Discrete: Values are effectively integers (within 1e-6 tolerance)")
    print("• Probabilistic: Non-integer values AND (has 0-1 values OR >10 unique values)")
    print("• Weighted mean strategy is automatically used for probabilistic atlases")
    
    print("\nResult Formats:")
    print("• Discrete atlas: Multiple time-series (one per parcel)")
    print("• Probabilistic atlas: Single weighted average time-series")
    
    print(f"\nDemo completed successfully! 🎉")


if __name__ == '__main__':
    main()