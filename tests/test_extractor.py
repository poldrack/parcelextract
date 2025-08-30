"""Test the main ParcelExtractor class."""

import numpy as np
import pytest

from parcelextract.core.extractor import ParcelExtractor


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