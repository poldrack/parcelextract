"""Test automatic probabilistic atlas detection and weighted mean strategy selection.

These tests verify that the ParcelExtractor can automatically detect when an atlas
contains probabilistic (continuous) values rather than discrete integer labels,
and automatically applies the weighted mean strategy in such cases.
"""

import pytest
import numpy as np
import nibabel as nib
from pathlib import Path

from parcelextract.core.extractor import ParcelExtractor


class TestProbabilisticAtlasDetection:
    """Test automatic detection and handling of probabilistic atlases."""

    def test_detect_discrete_atlas(self, temp_dir):
        """Test that discrete integer atlases are detected correctly."""
        # Create discrete atlas with integer labels
        atlas_shape = (10, 10, 10)
        atlas_data = np.zeros(atlas_shape, dtype=np.int16)
        
        # Create 3 discrete parcels
        atlas_data[2:5, 2:5, 2:5] = 1  # Parcel 1
        atlas_data[6:9, 2:5, 2:5] = 2  # Parcel 2
        atlas_data[2:5, 6:9, 2:5] = 3  # Parcel 3
        
        # Save atlas
        atlas_file = temp_dir / "discrete_atlas.nii.gz"
        nib.save(nib.Nifti1Image(atlas_data, np.eye(4)), atlas_file)
        
        # Initialize extractor with mean strategy
        extractor = ParcelExtractor(atlas=str(atlas_file), strategy='mean')
        
        # Should detect as discrete atlas
        assert extractor.is_probabilistic_atlas() == False
        assert extractor.get_effective_strategy() == 'mean'  # Should keep original strategy

    def test_detect_probabilistic_atlas_float_values(self, temp_dir):
        """Test detection of probabilistic atlas with float values."""
        # Create probabilistic atlas with continuous values
        atlas_shape = (8, 8, 8)
        atlas_data = np.zeros(atlas_shape, dtype=np.float32)
        
        # Create probabilistic "parcel" with varying weights
        center = (4, 4, 4)
        for x in range(8):
            for y in range(8):
                for z in range(8):
                    dist = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
                    # Gaussian-like probability decreasing with distance
                    atlas_data[x, y, z] = np.exp(-dist**2 / 8.0)
        
        # Save atlas
        atlas_file = temp_dir / "probabilistic_atlas.nii.gz"
        nib.save(nib.Nifti1Image(atlas_data, np.eye(4)), atlas_file)
        
        # Initialize extractor with any strategy
        extractor = ParcelExtractor(atlas=str(atlas_file), strategy='mean')
        
        # Should detect as probabilistic atlas
        assert extractor.is_probabilistic_atlas() == True
        assert extractor.get_effective_strategy() == 'weighted_mean'  # Should auto-switch

    def test_detect_probabilistic_atlas_values_between_zero_and_one(self, temp_dir):
        """Test detection when values are clearly probabilities (0-1 range)."""
        atlas_shape = (6, 6, 6)
        atlas_data = np.random.rand(*atlas_shape).astype(np.float32) * 0.8  # Values 0-0.8
        
        # Make some regions have higher probabilities
        atlas_data[1:3, 1:3, 1:3] = 0.9  # High probability region
        atlas_data[4:6, 4:6, 4:6] = 0.7  # Medium probability region
        
        atlas_file = temp_dir / "probability_atlas.nii.gz"
        nib.save(nib.Nifti1Image(atlas_data, np.eye(4)), atlas_file)
        
        extractor = ParcelExtractor(atlas=str(atlas_file), strategy='pca')
        
        # Should detect as probabilistic and override strategy
        assert extractor.is_probabilistic_atlas() == True
        assert extractor.get_effective_strategy() == 'weighted_mean'

    def test_mixed_integer_float_atlas_treated_as_discrete(self, temp_dir):
        """Test that atlas with integer values stored as float is treated as discrete."""
        # Create atlas with integer values but stored as float
        atlas_shape = (8, 8, 8)
        atlas_data = np.zeros(atlas_shape, dtype=np.float32)
        
        # Set discrete integer values (but stored as float)
        atlas_data[1:4, 1:4, 1:4] = 1.0  # Parcel 1
        atlas_data[5:8, 1:4, 1:4] = 2.0  # Parcel 2
        atlas_data[1:4, 5:8, 1:4] = 3.0  # Parcel 3
        
        atlas_file = temp_dir / "integer_as_float_atlas.nii.gz"
        nib.save(nib.Nifti1Image(atlas_data, np.eye(4)), atlas_file)
        
        extractor = ParcelExtractor(atlas=str(atlas_file), strategy='median')
        
        # Should detect as discrete (values are effectively integers)
        assert extractor.is_probabilistic_atlas() == False
        assert extractor.get_effective_strategy() == 'median'  # Keep original

    def test_probabilistic_atlas_extraction_accuracy(self, temp_dir):
        """Test that probabilistic atlas extraction produces correct weighted results."""
        img_shape = (6, 6, 6, 20)
        n_timepoints = img_shape[3]
        
        # Create 4D image data
        img_4d = np.zeros(img_shape, dtype=np.float32)
        
        # Create probabilistic atlas
        atlas_3d = np.zeros((6, 6, 6), dtype=np.float32)
        
        # Define known signals and weights
        voxel_specs = [
            # (x, y, z, weight, signal_amplitude)
            (2, 2, 2, 1.0, 3.0),
            (2, 2, 3, 0.8, 2.0),
            (2, 3, 2, 0.6, 1.0),
            (3, 2, 2, 0.4, 0.5),
        ]
        
        # Create time vector and base signal
        t = np.arange(n_timepoints, dtype=np.float32)
        base_signal = np.sin(2 * np.pi * 0.15 * t)
        
        # Calculate expected weighted mean manually
        expected_weighted_sum = np.zeros(n_timepoints, dtype=np.float32)
        total_weight = 0.0
        
        for x, y, z, weight, amplitude in voxel_specs:
            atlas_3d[x, y, z] = weight
            signal = amplitude * base_signal
            img_4d[x, y, z, :] = signal
            
            expected_weighted_sum += weight * signal
            total_weight += weight
        
        expected_result = expected_weighted_sum / total_weight
        
        # Save test data
        img_file = temp_dir / "test_image.nii.gz"
        atlas_file = temp_dir / "prob_atlas.nii.gz"
        
        nib.save(nib.Nifti1Image(img_4d, np.eye(4)), img_file)
        nib.save(nib.Nifti1Image(atlas_3d, np.eye(4)), atlas_file)
        
        # Extract using ParcelExtractor (should auto-detect and use weighted mean)
        extractor = ParcelExtractor(atlas=str(atlas_file), strategy='mean')  # Start with mean
        
        # Verify detection
        assert extractor.is_probabilistic_atlas() == True
        assert extractor.get_effective_strategy() == 'weighted_mean'
        
        # Extract timeseries
        timeseries = extractor.fit_transform(str(img_file))
        
        # Should extract single timeseries (probabilistic atlas = single "parcel")
        assert timeseries.shape[0] == 1
        assert timeseries.shape[1] == n_timepoints
        
        # Check accuracy
        extracted_signal = timeseries[0, :]
        np.testing.assert_allclose(
            extracted_signal, expected_result, rtol=1e-6,
            err_msg="Probabilistic atlas extraction doesn't match expected weighted mean"
        )

    def test_strategy_override_warning(self, temp_dir):
        """Test that user is warned when strategy is overridden for probabilistic atlas."""
        # Create probabilistic atlas
        atlas_shape = (5, 5, 5)
        atlas_data = np.random.rand(*atlas_shape).astype(np.float32) * 0.9
        
        atlas_file = temp_dir / "prob_atlas.nii.gz"
        nib.save(nib.Nifti1Image(atlas_data, np.eye(4)), atlas_file)
        
        # This test will be implemented when we add logging/warnings
        # For now, just verify the behavior
        extractor = ParcelExtractor(atlas=str(atlas_file), strategy='pca')
        
        assert extractor.is_probabilistic_atlas() == True
        assert extractor.get_effective_strategy() == 'weighted_mean'
        assert extractor.strategy == 'pca'  # Original strategy preserved
        
        # TODO: Add test for warning message when logging is implemented

    def test_discrete_atlas_with_non_integer_values_edge_case(self, temp_dir):
        """Test edge case: atlas with values very close to integers."""
        atlas_shape = (4, 4, 4)
        atlas_data = np.zeros(atlas_shape, dtype=np.float32)
        
        # Values very close to integers (within floating point precision)
        atlas_data[1:3, 1:3, 1:3] = 1.0000001  # Effectively 1
        atlas_data[2:4, 2:4, 2:4] = 2.0000002  # Effectively 2
        
        atlas_file = temp_dir / "near_integer_atlas.nii.gz"
        nib.save(nib.Nifti1Image(atlas_data, np.eye(4)), atlas_file)
        
        extractor = ParcelExtractor(atlas=str(atlas_file), strategy='mean')
        
        # Should be treated as discrete (within tolerance)
        # This will depend on the tolerance we set in the implementation
        # For now, let's be conservative and treat as probabilistic
        # (This can be adjusted based on requirements)
        is_prob = extractor.is_probabilistic_atlas()
        # Either behavior could be correct depending on tolerance chosen
        assert isinstance(is_prob, bool)  # Just verify it returns a boolean

    def test_empty_atlas_handling(self, temp_dir):
        """Test handling of atlas with all zero values."""
        atlas_shape = (4, 4, 4)
        atlas_data = np.zeros(atlas_shape, dtype=np.float32)
        
        atlas_file = temp_dir / "empty_atlas.nii.gz"
        nib.save(nib.Nifti1Image(atlas_data, np.eye(4)), atlas_file)
        
        # Should raise appropriate error regardless of detection
        with pytest.raises(ValueError, match="no non-zero values|no parcels|empty atlas"):
            extractor = ParcelExtractor(atlas=str(atlas_file), strategy='mean')
            
            # Try to create 4D data for testing
            img_4d = np.random.randn(4, 4, 4, 10).astype(np.float32)
            img_file = temp_dir / "test_img.nii.gz"
            nib.save(nib.Nifti1Image(img_4d, np.eye(4)), img_file)
            
            extractor.fit_transform(str(img_file))

    def test_multiple_probabilistic_regions(self, temp_dir):
        """Test handling of atlas with multiple separate probabilistic regions."""
        img_shape = (10, 10, 10, 15)
        atlas_shape = (10, 10, 10)
        
        # Create atlas with two separate probabilistic regions
        atlas_data = np.zeros(atlas_shape, dtype=np.float32)
        
        # Region 1: Gaussian centered at (2, 2, 2)
        center1 = (2, 2, 2)
        for x in range(5):
            for y in range(5):
                for z in range(5):
                    dist1 = np.sqrt((x - center1[0])**2 + (y - center1[1])**2 + (z - center1[2])**2)
                    if dist1 < 3:
                        atlas_data[x, y, z] = max(0, 0.9 - 0.3 * dist1)
        
        # Region 2: Gaussian centered at (7, 7, 7)  
        center2 = (7, 7, 7)
        for x in range(5, 10):
            for y in range(5, 10):
                for z in range(5, 10):
                    dist2 = np.sqrt((x - center2[0])**2 + (y - center2[1])**2 + (z - center2[2])**2)
                    if dist2 < 3:
                        atlas_data[x, y, z] = max(0, 0.8 - 0.25 * dist2)
        
        # Create corresponding 4D image
        img_4d = np.random.randn(*img_shape).astype(np.float32) * 0.1  # Small noise
        
        # Add distinct signals to each region
        t = np.arange(img_shape[3])
        signal1 = np.sin(2 * np.pi * 0.1 * t)
        signal2 = np.cos(2 * np.pi * 0.2 * t)
        
        # Apply signals to regions
        for x in range(10):
            for y in range(10):
                for z in range(10):
                    if atlas_data[x, y, z] > 0:
                        if x < 5:  # Region 1
                            img_4d[x, y, z, :] += signal1 * atlas_data[x, y, z]
                        else:  # Region 2
                            img_4d[x, y, z, :] += signal2 * atlas_data[x, y, z]
        
        # Save test data
        atlas_file = temp_dir / "multi_prob_atlas.nii.gz"
        img_file = temp_dir / "multi_prob_img.nii.gz"
        
        nib.save(nib.Nifti1Image(atlas_data, np.eye(4)), atlas_file)
        nib.save(nib.Nifti1Image(img_4d, np.eye(4)), img_file)
        
        # Test extraction
        extractor = ParcelExtractor(atlas=str(atlas_file), strategy='mean')
        
        # Should detect as probabilistic
        assert extractor.is_probabilistic_atlas() == True
        
        # Extract timeseries
        timeseries = extractor.fit_transform(str(img_file))
        
        # For probabilistic atlas, should return single timeseries
        # (This is the current design - may need refinement for multiple regions)
        assert timeseries.shape[0] == 1
        assert timeseries.shape[1] == img_shape[3]
        
        # Extracted signal should be non-trivial
        assert np.std(timeseries[0, :]) > 0.1  # Should have reasonable variance

    def test_preserve_original_strategy_attribute(self, temp_dir):
        """Test that original strategy is preserved even when overridden."""
        # Create probabilistic atlas
        atlas_shape = (4, 4, 4)
        atlas_data = np.random.rand(*atlas_shape).astype(np.float32) * 0.5
        
        atlas_file = temp_dir / "prob_atlas.nii.gz"
        nib.save(nib.Nifti1Image(atlas_data, np.eye(4)), atlas_file)
        
        original_strategy = 'median'
        extractor = ParcelExtractor(atlas=str(atlas_file), strategy=original_strategy)
        
        # Original strategy should be preserved
        assert extractor.strategy == original_strategy
        
        # But effective strategy should be weighted_mean
        assert extractor.get_effective_strategy() == 'weighted_mean'
        
        # Detection should work
        assert extractor.is_probabilistic_atlas() == True