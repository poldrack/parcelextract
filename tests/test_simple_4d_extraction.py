"""Simple test for 4D probabilistic atlas extraction."""

import pytest
import numpy as np
import nibabel as nib
import tempfile
from pathlib import Path

from parcelextract.core.extractor import ParcelExtractor


def test_simple_4d_probabilistic_extraction(temp_dir):
    """Test basic 4D probabilistic atlas extraction with known setup."""
    # Simple test parameters
    spatial_shape = (10, 10, 10)
    n_timepoints = 50
    n_parcels = 3
    
    # Create very simple, well-separated 4D atlas
    atlas_4d = np.zeros((*spatial_shape, n_parcels), dtype=np.float32)
    
    # Create 3 well-separated, compact parcels
    # Parcel 1: corner (1,1,1) - (3,3,3)
    atlas_4d[1:4, 1:4, 1:4, 0] = 1.0
    
    # Parcel 2: corner (6,6,6) - (8,8,8)  
    atlas_4d[6:9, 6:9, 6:9, 1] = 1.0
    
    # Parcel 3: middle region (4,4,4) - (6,6,6)
    atlas_4d[4:7, 4:7, 4:7, 2] = 1.0
    
    # Create simple, distinct signals
    t = np.arange(n_timepoints)
    signal1 = 2.0 * np.sin(2 * np.pi * 0.1 * t)  # 0.1 Hz sine
    signal2 = 1.5 * np.cos(2 * np.pi * 0.15 * t)  # 0.15 Hz cosine
    signal3 = 1.0 * np.sin(2 * np.pi * 0.05 * t)  # 0.05 Hz sine (slower)
    
    signals = [signal1, signal2, signal3]
    
    # Create 4D image with embedded signals and minimal noise
    img_4d = np.random.randn(*spatial_shape, n_timepoints).astype(np.float32) * 0.01  # Very low noise
    
    # Add signals to their respective regions
    for parcel_id in range(n_parcels):
        parcel_weights = atlas_4d[:, :, :, parcel_id]
        signal = signals[parcel_id]
        
        # Add weighted signal
        for x in range(spatial_shape[0]):
            for y in range(spatial_shape[1]):
                for z in range(spatial_shape[2]):
                    weight = parcel_weights[x, y, z]
                    if weight > 0:
                        img_4d[x, y, z, :] += weight * signal
    
    # Save test data
    affine = np.eye(4)
    img_file = temp_dir / "simple_4d_image.nii.gz"
    atlas_file = temp_dir / "simple_4d_atlas.nii.gz"
    
    nib.save(nib.Nifti1Image(img_4d, affine), img_file)
    nib.save(nib.Nifti1Image(atlas_4d, affine), atlas_file)
    
    print(f"Created simple image: {img_4d.shape}")
    print(f"Created simple atlas: {atlas_4d.shape}")
    print(f"Atlas weights: parcel volumes = {[np.sum(atlas_4d[:,:,:,i] > 0) for i in range(n_parcels)]}")
    
    # Extract using ParcelExtractor
    extractor = ParcelExtractor(atlas=str(atlas_file), strategy='mean')
    
    # Verify detection
    assert extractor.is_probabilistic_atlas() == True
    assert extractor.get_effective_strategy() == 'weighted_mean'
    
    # Extract timeseries
    extracted_ts = extractor.fit_transform(str(img_file))
    
    print(f"Extracted shape: {extracted_ts.shape}")
    assert extracted_ts.shape == (n_parcels, n_timepoints)
    
    # Check correlations
    for parcel_id in range(n_parcels):
        extracted_signal = extracted_ts[parcel_id, :]
        expected_signal = signals[parcel_id]
        
        correlation = np.corrcoef(extracted_signal, expected_signal)[0, 1]
        print(f"Parcel {parcel_id + 1}: correlation = {correlation:.3f}")
        
        # With this simple setup, correlation should be very high
        assert correlation > 0.99, f"Parcel {parcel_id} correlation too low: {correlation:.3f}"
    
    print("✅ Simple 4D probabilistic extraction test passed!")


def test_4d_vs_3d_probabilistic_comparison(temp_dir):
    """Compare 4D probabilistic vs 3D probabilistic extraction."""
    spatial_shape = (8, 8, 8)  
    n_timepoints = 30
    
    # Create identical signals for both tests
    t = np.arange(n_timepoints)
    test_signal = np.sin(2 * np.pi * 0.1 * t)
    
    # Test 1: 3D probabilistic atlas (single weighted region)
    atlas_3d = np.zeros(spatial_shape, dtype=np.float32)
    atlas_3d[2:6, 2:6, 2:6] = 0.8  # Uniform weights in center region
    
    img_3d_test = np.random.randn(*spatial_shape, n_timepoints).astype(np.float32) * 0.01
    
    # Add signal to 3D region
    for x in range(2, 6):
        for y in range(2, 6):
            for z in range(2, 6):
                img_3d_test[x, y, z, :] += atlas_3d[x, y, z] * test_signal
    
    # Save 3D test
    atlas_3d_file = temp_dir / "atlas_3d.nii.gz" 
    img_3d_file = temp_dir / "img_3d.nii.gz"
    nib.save(nib.Nifti1Image(atlas_3d, np.eye(4)), atlas_3d_file)
    nib.save(nib.Nifti1Image(img_3d_test, np.eye(4)), img_3d_file)
    
    # Test 2: 4D probabilistic atlas (single parcel in 4th dimension)
    atlas_4d = np.zeros((*spatial_shape, 1), dtype=np.float32)
    atlas_4d[2:6, 2:6, 2:6, 0] = 0.8  # Same region, same weights
    
    img_4d_test = img_3d_test.copy()  # Identical image data
    
    # Save 4D test  
    atlas_4d_file = temp_dir / "atlas_4d.nii.gz"
    img_4d_file = temp_dir / "img_4d.nii.gz"
    nib.save(nib.Nifti1Image(atlas_4d, np.eye(4)), atlas_4d_file)
    nib.save(nib.Nifti1Image(img_4d_test, np.eye(4)), img_4d_file)
    
    # Extract with both atlas types
    extractor_3d = ParcelExtractor(atlas=str(atlas_3d_file), strategy='mean')
    extractor_4d = ParcelExtractor(atlas=str(atlas_4d_file), strategy='mean')
    
    # Both should be detected as probabilistic
    assert extractor_3d.is_probabilistic_atlas() == True
    assert extractor_4d.is_probabilistic_atlas() == True
    
    ts_3d = extractor_3d.fit_transform(str(img_3d_file))
    ts_4d = extractor_4d.fit_transform(str(img_4d_file))
    
    print(f"3D atlas extraction shape: {ts_3d.shape}")
    print(f"4D atlas extraction shape: {ts_4d.shape}")
    
    # 3D should give (1, n_timepoints), 4D should give (1, n_timepoints)
    assert ts_3d.shape == (1, n_timepoints)
    assert ts_4d.shape == (1, n_timepoints)
    
    # Both should recover the same signal (since identical setup)
    corr_3d = np.corrcoef(ts_3d[0, :], test_signal)[0, 1]
    corr_4d = np.corrcoef(ts_4d[0, :], test_signal)[0, 1]
    
    print(f"3D atlas correlation: {corr_3d:.3f}")
    print(f"4D atlas correlation: {corr_4d:.3f}")
    
    # Both should have high correlation
    assert corr_3d > 0.99
    assert corr_4d > 0.99
    
    # Results should be very similar
    correlation_between = np.corrcoef(ts_3d[0, :], ts_4d[0, :])[0, 1]
    print(f"Correlation between 3D and 4D extraction: {correlation_between:.3f}")
    assert correlation_between > 0.99
    
    print("✅ 3D vs 4D probabilistic comparison passed!")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        test_simple_4d_probabilistic_extraction(temp_path)
        test_4d_vs_3d_probabilistic_comparison(temp_path)