"""Test extraction strategy implementations."""

import numpy as np
import pytest
from sklearn.decomposition import PCA

from parcelextract.core.strategies import (
    ExtractionStrategy,
    MeanExtractionStrategy,
    MedianExtractionStrategy,
    PCAExtractionStrategy,
    WeightedMeanExtractionStrategy,
)


class TestExtractionStrategy:
    """Test the base ExtractionStrategy abstract class."""

    def test_base_strategy_cannot_be_instantiated(self):
        """Test that the abstract base class cannot be instantiated."""
        with pytest.raises(TypeError):
            ExtractionStrategy()

    def test_base_strategy_has_extract_method(self):
        """Test that the base class defines the extract method."""
        # Check that the method exists and is abstract
        assert hasattr(ExtractionStrategy, "extract")
        assert getattr(ExtractionStrategy.extract, "__isabstractmethod__", False)


class TestMeanExtractionStrategy:
    """Test mean extraction strategy."""

    @pytest.fixture
    def strategy(self):
        """Create a MeanExtractionStrategy instance."""
        return MeanExtractionStrategy()

    @pytest.fixture
    def simple_4d_data(self):
        """Create simple 4D test data with known mean values."""
        # Create 2x2x2x3 data where each parcel has predictable mean
        data = np.zeros((2, 2, 2, 3))
        
        # Fill with known values for easy testing
        data[:, :, :, 0] = 1.0  # First timepoint: all 1s
        data[:, :, :, 1] = 2.0  # Second timepoint: all 2s
        data[:, :, :, 2] = 3.0  # Third timepoint: all 3s
        
        return data

    @pytest.fixture
    def simple_parcel_mask(self):
        """Create a simple parcel mask."""
        mask = np.zeros((2, 2, 2), dtype=bool)
        mask[0, 0, 0] = True  # Single voxel parcel
        mask[1, 1, 1] = True  # Another single voxel
        return mask

    def test_extract_mean_single_voxel(self, strategy, simple_4d_data, simple_parcel_mask):
        """Test mean extraction from single-voxel parcel."""
        result = strategy.extract(simple_4d_data, simple_parcel_mask)
        
        # Should have 3 timepoints
        assert result.shape == (3,)
        
        # For single voxel, mean should equal the voxel value
        # Since mask selects voxel [0,0,0], result should be [1, 2, 3]
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_extract_mean_multiple_voxels(self, strategy):
        """Test mean extraction from multi-voxel parcel."""
        # Create data where parcel voxels have different values
        data_4d = np.zeros((3, 3, 3, 2))
        data_4d[0, 0, 0, :] = [1.0, 10.0]  # First voxel
        data_4d[0, 0, 1, :] = [3.0, 30.0]  # Second voxel
        
        # Create mask selecting both voxels
        mask = np.zeros((3, 3, 3), dtype=bool)
        mask[0, 0, 0] = True
        mask[0, 0, 1] = True
        
        result = strategy.extract(data_4d, mask)
        
        # Mean of [1,3] and [10,30] should be [2, 20]
        expected = np.array([2.0, 20.0])
        np.testing.assert_array_equal(result, expected)

    def test_extract_mean_with_nan_values(self, strategy):
        """Test mean extraction handles NaN values correctly."""
        data_4d = np.full((2, 2, 2, 3), 5.0)
        data_4d[0, 0, 0, :] = np.nan  # One voxel has NaN
        
        mask = np.zeros((2, 2, 2), dtype=bool)
        mask[0, 0, 0] = True  # NaN voxel
        mask[0, 0, 1] = True  # Regular voxel with value 5
        
        result = strategy.extract(data_4d, mask)
        
        # nanmean should ignore NaN and return 5.0 for all timepoints
        np.testing.assert_array_equal(result, [5.0, 5.0, 5.0])

    def test_extract_mean_all_nan_voxels(self, strategy):
        """Test mean extraction when all parcel voxels are NaN."""
        data_4d = np.full((2, 2, 2, 2), np.nan)
        
        mask = np.zeros((2, 2, 2), dtype=bool)
        mask[0, 0, 0] = True
        mask[0, 0, 1] = True
        
        result = strategy.extract(data_4d, mask)
        
        # All NaN inputs should produce NaN output
        assert np.all(np.isnan(result))

    def test_extract_mean_empty_mask(self, strategy, simple_4d_data):
        """Test extraction with empty mask raises appropriate error."""
        empty_mask = np.zeros((2, 2, 2), dtype=bool)
        
        with pytest.raises(ValueError, match="No voxels selected"):
            strategy.extract(simple_4d_data, empty_mask)

    def test_extract_mean_preserves_dtype(self, strategy):
        """Test that extraction preserves appropriate dtype."""
        data_4d = np.random.randn(3, 3, 3, 10).astype(np.float32)
        mask = np.zeros((3, 3, 3), dtype=bool)
        mask[1, 1, 1] = True
        
        result = strategy.extract(data_4d, mask)
        
        # Result should be float (at least float32 precision)
        assert np.issubdtype(result.dtype, np.floating)


class TestMedianExtractionStrategy:
    """Test median extraction strategy."""

    @pytest.fixture
    def strategy(self):
        """Create a MedianExtractionStrategy instance."""
        return MedianExtractionStrategy()

    def test_extract_median_odd_number_voxels(self, strategy):
        """Test median extraction with odd number of voxels."""
        # Create data with 3 voxels having values [1, 2, 3] at each timepoint
        data_4d = np.zeros((3, 1, 1, 2))
        data_4d[0, 0, 0, :] = [1.0, 10.0]  # First voxel
        data_4d[1, 0, 0, :] = [2.0, 20.0]  # Second voxel (median)
        data_4d[2, 0, 0, :] = [3.0, 30.0]  # Third voxel
        
        mask = np.ones((3, 1, 1), dtype=bool)
        
        result = strategy.extract(data_4d, mask)
        
        # Median should be [2, 20]
        np.testing.assert_array_equal(result, [2.0, 20.0])

    def test_extract_median_even_number_voxels(self, strategy):
        """Test median extraction with even number of voxels."""
        # Create data with 4 voxels
        data_4d = np.zeros((2, 2, 1, 1))
        data_4d[0, 0, 0, 0] = 1.0
        data_4d[0, 1, 0, 0] = 2.0
        data_4d[1, 0, 0, 0] = 3.0
        data_4d[1, 1, 0, 0] = 4.0
        
        mask = np.ones((2, 2, 1), dtype=bool)
        
        result = strategy.extract(data_4d, mask)
        
        # Median of [1,2,3,4] is 2.5
        np.testing.assert_array_equal(result, [2.5])

    def test_extract_median_single_voxel(self, strategy):
        """Test median extraction from single voxel."""
        data_4d = np.array([[[[5.0, 10.0]]]])
        mask = np.array([[[True]]])
        
        result = strategy.extract(data_4d, mask)
        
        # Median of single voxel is the voxel value
        np.testing.assert_array_equal(result, [5.0, 10.0])

    def test_extract_median_with_nan_values(self, strategy):
        """Test median extraction handles NaN values."""
        data_4d = np.zeros((3, 1, 1, 1))
        data_4d[0, 0, 0, 0] = 1.0
        data_4d[1, 0, 0, 0] = np.nan
        data_4d[2, 0, 0, 0] = 3.0
        
        mask = np.ones((3, 1, 1), dtype=bool)
        
        result = strategy.extract(data_4d, mask)
        
        # nanmedian should ignore NaN, median of [1, 3] is 2.0
        np.testing.assert_array_equal(result, [2.0])


class TestPCAExtractionStrategy:
    """Test PCA extraction strategy."""

    @pytest.fixture
    def strategy(self):
        """Create a PCAExtractionStrategy instance."""
        return PCAExtractionStrategy()

    def test_extract_pca_multiple_voxels(self, strategy):
        """Test PCA extraction from multiple voxels."""
        # Create data where first PC is obvious
        data_4d = np.zeros((2, 2, 1, 3))
        
        # Create pattern where voxels have correlated signal
        data_4d[0, 0, 0, :] = [1.0, 2.0, 3.0]  # Base signal
        data_4d[0, 1, 0, :] = [2.0, 4.0, 6.0]  # 2x base signal
        data_4d[1, 0, 0, :] = [0.5, 1.0, 1.5]  # 0.5x base signal
        
        mask = np.ones((2, 2, 1), dtype=bool)
        mask[1, 1, 0] = False  # Exclude one voxel
        
        result = strategy.extract(data_4d, mask)
        
        # Should return first PC timeseries (3 timepoints)
        assert result.shape == (3,)
        assert np.isfinite(result).all()

    def test_extract_pca_single_voxel(self, strategy):
        """Test PCA extraction from single voxel."""
        data_4d = np.array([[[[1.0, 2.0, 3.0]]]])
        mask = np.array([[[True]]])
        
        result = strategy.extract(data_4d, mask)
        
        # With single voxel, PC1 should be the normalized signal
        assert result.shape == (3,)

    def test_extract_pca_insufficient_timepoints(self, strategy):
        """Test PCA with insufficient timepoints raises error."""
        # Only 1 timepoint - PCA needs at least 2
        data_4d = np.random.randn(3, 3, 3, 1)
        mask = np.ones((3, 3, 3), dtype=bool)
        
        with pytest.raises(ValueError, match="at least 2 timepoints"):
            strategy.extract(data_4d, mask)

    def test_extract_pca_handles_constant_voxels(self, strategy):
        """Test PCA handles voxels with constant values."""
        # Create data where some voxels are constant
        data_4d = np.zeros((2, 1, 1, 4))
        data_4d[0, 0, 0, :] = [1.0, 1.0, 1.0, 1.0]  # Constant
        data_4d[1, 0, 0, :] = [1.0, 2.0, 3.0, 4.0]  # Variable
        
        mask = np.ones((2, 1, 1), dtype=bool)
        
        result = strategy.extract(data_4d, mask)
        
        assert result.shape == (4,)
        assert np.isfinite(result).all()


class TestWeightedMeanExtractionStrategy:
    """Test weighted mean extraction strategy."""

    @pytest.fixture
    def strategy(self):
        """Create a WeightedMeanExtractionStrategy instance."""
        return WeightedMeanExtractionStrategy()

    @pytest.fixture
    def probability_mask(self):
        """Create a probabilistic parcel mask."""
        prob_mask = np.zeros((3, 3, 1))
        prob_mask[0, 0, 0] = 0.1  # Low probability
        prob_mask[1, 1, 0] = 0.9  # High probability
        prob_mask[2, 2, 0] = 0.5  # Medium probability
        return prob_mask

    def test_extract_weighted_mean_probabilistic(self, strategy, probability_mask):
        """Test weighted mean extraction with probabilistic weights."""
        data_4d = np.zeros((3, 3, 1, 2))
        data_4d[0, 0, 0, :] = [10.0, 100.0]  # Low weight voxel
        data_4d[1, 1, 0, :] = [20.0, 200.0]  # High weight voxel  
        data_4d[2, 2, 0, :] = [30.0, 300.0]  # Medium weight voxel
        
        result = strategy.extract(data_4d, probability_mask)
        
        # Weighted mean: (10*0.1 + 20*0.9 + 30*0.5) / (0.1 + 0.9 + 0.5) = 34/1.5 ≈ 22.67
        expected_t1 = (10*0.1 + 20*0.9 + 30*0.5) / (0.1 + 0.9 + 0.5)
        expected_t2 = (100*0.1 + 200*0.9 + 300*0.5) / (0.1 + 0.9 + 0.5)
        
        np.testing.assert_array_almost_equal(result, [expected_t1, expected_t2])

    def test_extract_weighted_mean_binary_mask(self, strategy):
        """Test weighted mean with binary mask (equivalent to regular mean)."""
        data_4d = np.zeros((2, 1, 1, 2))
        data_4d[0, 0, 0, :] = [1.0, 10.0]
        data_4d[1, 0, 0, :] = [3.0, 30.0]
        
        # Binary mask (0 or 1)
        weights = np.zeros((2, 1, 1))
        weights[0, 0, 0] = 1.0
        weights[1, 0, 0] = 1.0
        
        result = strategy.extract(data_4d, weights)
        
        # Should equal regular mean: [2.0, 20.0]
        np.testing.assert_array_equal(result, [2.0, 20.0])

    def test_extract_weighted_mean_zero_weights(self, strategy):
        """Test weighted mean with all zero weights raises error."""
        data_4d = np.random.randn(2, 2, 2, 3)
        zero_weights = np.zeros((2, 2, 2))
        
        with pytest.raises(ValueError, match="Sum of weights is zero"):
            strategy.extract(data_4d, zero_weights)

    def test_extract_weighted_mean_single_voxel(self, strategy):
        """Test weighted mean extraction from single voxel."""
        data_4d = np.array([[[[5.0, 15.0]]]])
        weights = np.array([[[0.7]]])
        
        result = strategy.extract(data_4d, weights)
        
        # Single voxel weighted mean is just the voxel value
        np.testing.assert_array_almost_equal(result, [5.0, 15.0])

    def test_extract_weighted_mean_negative_weights_raises_error(self, strategy):
        """Test that negative weights raise an error."""
        data_4d = np.random.randn(2, 2, 1, 2)
        weights = np.array([[[0.5], [-0.3]]])  # Negative weight
        
        with pytest.raises(ValueError, match="Weights cannot be negative"):
            strategy.extract(data_4d, weights)


class TestStrategyIntegration:
    """Test integration between different strategies."""

    def test_all_strategies_same_interface(self):
        """Test that all strategies implement the same interface."""
        strategies = [
            MeanExtractionStrategy(),
            MedianExtractionStrategy(),
            PCAExtractionStrategy(),
            WeightedMeanExtractionStrategy(),
        ]
        
        data_4d = np.random.randn(5, 5, 5, 10)
        mask = np.zeros((5, 5, 5), dtype=bool)
        mask[2:4, 2:4, 2:4] = True  # Select a small region
        
        for strategy in strategies:
            if isinstance(strategy, WeightedMeanExtractionStrategy):
                # For weighted strategy, convert boolean mask to weights
                weights = mask.astype(float)
                result = strategy.extract(data_4d, weights)
            else:
                result = strategy.extract(data_4d, mask)
            
            # All should return 1D array with 10 timepoints
            assert result.shape == (10,)
            assert result.ndim == 1

    def test_strategies_handle_same_edge_cases(self):
        """Test that all strategies handle edge cases consistently."""
        data_4d = np.full((3, 3, 3, 5), np.nan)
        data_4d[1, 1, 1, :] = [1, 2, 3, 4, 5]  # Only one non-NaN voxel
        
        mask = np.zeros((3, 3, 3), dtype=bool)
        mask[1, 1, 1] = True
        
        mean_result = MeanExtractionStrategy().extract(data_4d, mask)
        median_result = MedianExtractionStrategy().extract(data_4d, mask)
        
        # For single voxel, mean and median should be identical
        np.testing.assert_array_equal(mean_result, median_result)
        np.testing.assert_array_equal(mean_result, [1, 2, 3, 4, 5])