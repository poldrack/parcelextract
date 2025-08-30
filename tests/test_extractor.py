"""Test the main ParcelExtractor class."""

import numpy as np
import pytest

from parcelextract.core.extractor import ParcelExtractor
from parcelextract.core.validators import ValidationError


class TestParcelExtractor:
    """Test ParcelExtractor initialization and basic functionality."""

    def test_can_create_extractor_instance(self):
        """Test that we can create a ParcelExtractor instance."""
        extractor = ParcelExtractor()
        assert extractor is not None

    def test_can_create_extractor_with_atlas(self, test_atlas_nifti):
        """Test that we can create a ParcelExtractor with atlas parameter."""
        extractor = ParcelExtractor(atlas=test_atlas_nifti)
        assert extractor is not None
        assert extractor.atlas == test_atlas_nifti

    def test_can_create_extractor_with_strategy(self):
        """Test that we can create a ParcelExtractor with strategy parameter."""
        extractor = ParcelExtractor(strategy='mean')
        assert extractor is not None
        assert extractor.strategy == 'mean'

    def test_fit_transform_returns_timeseries(self, synthetic_4d_nifti, test_atlas_nifti):
        """Test that fit_transform returns a timeseries array."""
        extractor = ParcelExtractor(atlas=test_atlas_nifti, strategy='mean')
        result = extractor.fit_transform(synthetic_4d_nifti)
        
        # Should return 2D array: (n_parcels, n_timepoints)
        assert result.shape[0] == 5  # 5 parcels in test atlas
        assert result.shape[1] == 50  # 50 timepoints in synthetic data

    def test_fit_transform_extracts_nonzero_signals(self, synthetic_4d_nifti, test_atlas_nifti):
        """Test that fit_transform actually extracts non-zero signals from data."""
        extractor = ParcelExtractor(atlas=test_atlas_nifti, strategy='mean')
        result = extractor.fit_transform(synthetic_4d_nifti)
        
        # Result should not be all zeros (since synthetic data has random values)
        assert not np.allclose(result, 0)
        
        # Each parcel should have some variation across time
        for parcel_idx in range(result.shape[0]):
            parcel_timeseries = result[parcel_idx, :]
            assert np.std(parcel_timeseries) > 0  # Should have some variation

    def test_fit_transform_uses_different_strategies(self, synthetic_4d_nifti, test_atlas_nifti):
        """Test that different strategies produce different results."""
        # Extract with mean strategy
        extractor_mean = ParcelExtractor(atlas=test_atlas_nifti, strategy='mean')
        result_mean = extractor_mean.fit_transform(synthetic_4d_nifti)
        
        # Extract with median strategy  
        extractor_median = ParcelExtractor(atlas=test_atlas_nifti, strategy='median')
        result_median = extractor_median.fit_transform(synthetic_4d_nifti)
        
        # Results should be different (with random data, mean != median)
        assert not np.allclose(result_mean, result_median)
        
        # Both should have the same shape
        assert result_mean.shape == result_median.shape

    def test_invalid_strategy_raises_error(self, synthetic_4d_nifti, test_atlas_nifti):
        """Test that invalid strategy parameter raises ValueError."""
        with pytest.raises(ValueError, match="Unknown strategy: invalid"):
            extractor = ParcelExtractor(atlas=test_atlas_nifti, strategy='invalid')
            # This should trigger the error when _get_strategy() is called
            extractor.fit_transform(synthetic_4d_nifti)

    def test_pca_strategy_works(self, synthetic_4d_nifti, test_atlas_nifti):
        """Test that PCA strategy produces different results than mean."""
        # Extract with mean strategy
        extractor_mean = ParcelExtractor(atlas=test_atlas_nifti, strategy='mean')
        result_mean = extractor_mean.fit_transform(synthetic_4d_nifti)
        
        # Extract with PCA strategy
        extractor_pca = ParcelExtractor(atlas=test_atlas_nifti, strategy='pca')
        result_pca = extractor_pca.fit_transform(synthetic_4d_nifti)
        
        # Results should be different (PCA extracts first principal component)
        assert not np.allclose(result_mean, result_pca)
        
        # Both should have the same shape
        assert result_mean.shape == result_pca.shape
        
        # PCA results should have finite values (no NaN or inf)
        assert np.all(np.isfinite(result_pca))

    def test_fit_transform_validates_inputs(self, test_atlas_nifti):
        """Test that fit_transform validates its inputs using ValidationError."""
        extractor = ParcelExtractor(atlas=test_atlas_nifti, strategy='mean')
        
        # Test with None input should raise ValidationError
        with pytest.raises(ValidationError):
            extractor.fit_transform(None)
            
        # Test with non-existent file should raise ValidationError  
        with pytest.raises(ValidationError):
            extractor.fit_transform("/path/that/does/not/exist.nii.gz")