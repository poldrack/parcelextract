"""Test signal extraction accuracy with known synthetic signals.

These tests generate 4D images with known signal patterns and verify
that the extracted time-series match expected values for each strategy.
"""

import pytest
import numpy as np
import nibabel as nib
from pathlib import Path

from parcelextract.core.extractor import ParcelExtractor
from parcelextract.core.strategies import (
    MeanExtractionStrategy, MedianExtractionStrategy, 
    PCAExtractionStrategy, WeightedMeanExtractionStrategy
)


class TestSignalValidation:
    """Test extraction accuracy with known synthetic signals."""

    def test_constant_signal_extraction(self, temp_dir):
        """Test extraction of constant signals."""
        # Create 4D data with constant values in each parcel
        img_shape = (20, 20, 20, 50)  # 50 timepoints
        img_4d = np.zeros(img_shape, dtype=np.float32)
        
        # Create atlas with 3 parcels
        atlas_3d = np.zeros((20, 20, 20), dtype=np.int16)
        
        # Define parcel regions and their constant values
        parcels = [
            ((5, 10), (5, 10), (5, 10), 1, 10.0),   # Parcel 1: constant 10.0
            ((10, 15), (5, 10), (5, 10), 2, 25.0),  # Parcel 2: constant 25.0
            ((5, 10), (10, 15), (5, 10), 3, -5.0),  # Parcel 3: constant -5.0
        ]
        
        expected_signals = {}
        
        for (x1, x2), (y1, y2), (z1, z2), label, value in parcels:
            # Set atlas labels
            atlas_3d[x1:x2, y1:y2, z1:z2] = label
            
            # Set constant signal values
            img_4d[x1:x2, y1:y2, z1:z2, :] = value
            expected_signals[label-1] = np.full(50, value)  # 0-indexed for results
        
        # Save test data
        img_file = temp_dir / "constant_signal.nii.gz"
        atlas_file = temp_dir / "constant_atlas.nii.gz"
        
        nib.save(nib.Nifti1Image(img_4d, np.eye(4)), img_file)
        nib.save(nib.Nifti1Image(atlas_3d, np.eye(4)), atlas_file)
        
        # Test extraction with all strategies
        strategies = ['mean', 'median', 'weighted_mean']  # PCA won't work with constant signals
        
        for strategy_name in strategies:
            extractor = ParcelExtractor(atlas=str(atlas_file), strategy=strategy_name)
            timeseries = extractor.fit_transform(str(img_file))
            
            assert timeseries.shape == (3, 50), f"Wrong shape for {strategy_name}"
            
            # Check each parcel's signal
            for parcel_idx in range(3):
                expected = expected_signals[parcel_idx]
                actual = timeseries[parcel_idx, :]
                
                np.testing.assert_allclose(
                    actual, expected, rtol=1e-6,
                    err_msg=f"Strategy {strategy_name}, parcel {parcel_idx}: "
                           f"expected {expected[0]}, got {actual[0]}"
                )

    def test_sinusoidal_signal_extraction(self, temp_dir):
        """Test extraction of sinusoidal signals with known frequencies."""
        img_shape = (16, 16, 16, 100)  # 100 timepoints
        n_timepoints = img_shape[3]
        
        # Create time vector
        t = np.arange(n_timepoints, dtype=np.float32)
        
        # Define different sinusoidal signals for each parcel
        frequencies = [0.05, 0.1, 0.2]  # Different frequencies
        phases = [0, np.pi/4, np.pi/2]  # Different phases
        amplitudes = [1.0, 2.0, 0.5]   # Different amplitudes
        
        img_4d = np.zeros(img_shape, dtype=np.float32)
        atlas_3d = np.zeros((16, 16, 16), dtype=np.int16)
        
        expected_signals = {}
        
        parcels = [
            ((2, 6), (2, 6), (2, 6), 1),    # Parcel 1
            ((8, 12), (2, 6), (2, 6), 2),   # Parcel 2
            ((2, 6), (8, 12), (2, 6), 3),   # Parcel 3
        ]
        
        for i, ((x1, x2), (y1, y2), (z1, z2), label) in enumerate(parcels):
            # Generate sinusoidal signal
            freq, phase, amp = frequencies[i], phases[i], amplitudes[i]
            signal = amp * np.sin(2 * np.pi * freq * t + phase)
            
            # Set atlas labels
            atlas_3d[x1:x2, y1:y2, z1:z2] = label
            
            # Set signal values (all voxels in parcel have same signal)
            img_4d[x1:x2, y1:y2, z1:z2, :] = signal[np.newaxis, np.newaxis, np.newaxis, :]
            expected_signals[label-1] = signal
        
        # Save test data
        img_file = temp_dir / "sinusoidal_signal.nii.gz"
        atlas_file = temp_dir / "sinusoidal_atlas.nii.gz"
        
        nib.save(nib.Nifti1Image(img_4d, np.eye(4)), img_file)
        nib.save(nib.Nifti1Image(atlas_3d, np.eye(4)), atlas_file)
        
        # Test with mean, median, weighted_mean (should be identical for homogeneous parcels)
        strategies = ['mean', 'median', 'weighted_mean']
        
        for strategy_name in strategies:
            extractor = ParcelExtractor(atlas=str(atlas_file), strategy=strategy_name)
            timeseries = extractor.fit_transform(str(img_file))
            
            assert timeseries.shape == (3, 100), f"Wrong shape for {strategy_name}"
            
            # Check each parcel's signal
            for parcel_idx in range(3):
                expected = expected_signals[parcel_idx]
                actual = timeseries[parcel_idx, :]
                
                # Use correlation to test signal preservation (handles sign flips)
                correlation = np.corrcoef(expected, actual)[0, 1]
                assert correlation > 0.99, f"Low correlation for {strategy_name}, parcel {parcel_idx}: {correlation:.4f}"
                
                # Also test that the extracted signal has the correct frequency content
                fft_expected = np.fft.fft(expected)
                fft_actual = np.fft.fft(actual)
                
                # The dominant frequency should be the same
                freq_expected = np.argmax(np.abs(fft_expected[1:n_timepoints//2])) + 1
                freq_actual = np.argmax(np.abs(fft_actual[1:n_timepoints//2])) + 1
                
                assert freq_expected == freq_actual, f"Frequency mismatch for {strategy_name}, parcel {parcel_idx}"

    def test_mixed_voxel_signal_extraction(self, temp_dir):
        """Test extraction from parcels with mixed voxel signals."""
        img_shape = (12, 12, 12, 50)
        n_timepoints = img_shape[3]
        
        # Create parcel with different signals in different voxels
        img_4d = np.random.randn(*img_shape).astype(np.float32) * 0.1  # Small noise
        atlas_3d = np.zeros((12, 12, 12), dtype=np.int16)
        
        # Define parcel 1 with known mixed signals
        parcel_region = ((4, 8), (4, 8), (4, 8))
        x1, x2 = parcel_region[0]
        y1, y2 = parcel_region[1] 
        z1, z2 = parcel_region[2]
        
        atlas_3d[x1:x2, y1:y2, z1:z2] = 1
        
        # Create known signals for specific voxels
        voxel_signals = {}
        signal_sum = np.zeros(n_timepoints)
        n_voxels = 0
        
        for x in range(x1, x2):
            for y in range(y1, y2):
                for z in range(z1, z2):
                    # Create unique signal for each voxel
                    t = np.arange(n_timepoints)
                    signal = np.sin(2 * np.pi * 0.1 * t + (x + y + z) * 0.1) + (x + y + z) * 0.1
                    
                    img_4d[x, y, z, :] = signal
                    voxel_signals[(x, y, z)] = signal
                    signal_sum += signal
                    n_voxels += 1
        
        expected_mean = signal_sum / n_voxels
        expected_median = np.median([voxel_signals[key] for key in voxel_signals.keys()], axis=0)
        
        # Save test data
        img_file = temp_dir / "mixed_signal.nii.gz"
        atlas_file = temp_dir / "mixed_atlas.nii.gz"
        
        nib.save(nib.Nifti1Image(img_4d, np.eye(4)), img_file)
        nib.save(nib.Nifti1Image(atlas_3d, np.eye(4)), atlas_file)
        
        # Test mean strategy
        extractor_mean = ParcelExtractor(atlas=str(atlas_file), strategy='mean')
        timeseries_mean = extractor_mean.fit_transform(str(img_file))
        
        np.testing.assert_allclose(
            timeseries_mean[0, :], expected_mean, rtol=1e-5,
            err_msg="Mean extraction doesn't match expected average"
        )
        
        # Test median strategy
        extractor_median = ParcelExtractor(atlas=str(atlas_file), strategy='median')
        timeseries_median = extractor_median.fit_transform(str(img_file))
        
        np.testing.assert_allclose(
            timeseries_median[0, :], expected_median, rtol=1e-5,
            err_msg="Median extraction doesn't match expected median"
        )

    def test_pca_signal_extraction(self, temp_dir):
        """Test PCA extraction with known principal component."""
        img_shape = (10, 10, 10, 30)
        n_timepoints = img_shape[3]
        
        img_4d = np.zeros(img_shape, dtype=np.float32)
        atlas_3d = np.zeros((10, 10, 10), dtype=np.int16)
        
        # Create parcel region
        parcel_region = ((3, 7), (3, 7), (3, 7))
        x1, x2 = parcel_region[0]
        y1, y2 = parcel_region[1]
        z1, z2 = parcel_region[2]
        
        atlas_3d[x1:x2, y1:y2, z1:z2] = 1
        
        # Create signals that are linear combinations of a known component
        t = np.arange(n_timepoints, dtype=np.float32)
        base_signal = np.sin(2 * np.pi * 0.2 * t)  # Known principal component
        
        # Each voxel gets a weighted version of the base signal + small noise
        weights = []
        for x in range(x1, x2):
            for y in range(y1, y2):
                for z in range(z1, z2):
                    weight = 0.5 + 0.5 * ((x + y + z) / 20)  # Varying weights
                    noise = np.random.randn(n_timepoints) * 0.05  # Small noise
                    signal = weight * base_signal + noise
                    
                    img_4d[x, y, z, :] = signal
                    weights.append(weight)
        
        # Save test data
        img_file = temp_dir / "pca_signal.nii.gz"
        atlas_file = temp_dir / "pca_atlas.nii.gz"
        
        nib.save(nib.Nifti1Image(img_4d, np.eye(4)), img_file)
        nib.save(nib.Nifti1Image(atlas_3d, np.eye(4)), atlas_file)
        
        # Test PCA extraction
        extractor_pca = ParcelExtractor(atlas=str(atlas_file), strategy='pca')
        timeseries_pca = extractor_pca.fit_transform(str(img_file))
        
        # The first PC should be highly correlated with the base signal
        pc1 = timeseries_pca[0, :]
        
        # Check correlation (allowing for sign flip)
        correlation = abs(np.corrcoef(base_signal, pc1)[0, 1])
        assert correlation > 0.95, f"PCA didn't extract the principal component correctly: correlation = {correlation:.4f}"

    def test_weighted_mean_strategy_direct(self, temp_dir):
        """Test weighted mean extraction strategy directly with probabilistic weights."""
        # This test directly calls the WeightedMeanExtractionStrategy
        # to test its functionality with probabilistic weights
        
        img_shape = (6, 6, 6, 15)
        n_timepoints = img_shape[3]
        
        # Create 4D data
        img_4d = np.zeros(img_shape, dtype=np.float32)
        
        # Create probabilistic weights (not discrete parcels)
        weights_3d = np.zeros((6, 6, 6), dtype=np.float32)
        
        # Define specific voxels with known weights and signals
        voxel_data = [
            # (x, y, z, weight, signal_amplitude)
            (2, 2, 2, 1.0, 2.0),   # Center voxel, highest weight
            (2, 2, 3, 0.8, 1.5),   # Adjacent voxel
            (2, 3, 2, 0.6, 1.0),   # Another adjacent voxel  
            (3, 2, 2, 0.4, 0.5),   # Further voxel
        ]
        
        # Create time vector
        t = np.arange(n_timepoints, dtype=np.float32)
        base_signal = np.sin(2 * np.pi * 0.2 * t)  # Common signal pattern
        
        expected_weighted_sum = np.zeros(n_timepoints, dtype=np.float32)
        total_weight = 0.0
        
        for x, y, z, weight, amplitude in voxel_data:
            # Set weight 
            weights_3d[x, y, z] = weight
            
            # Create signal: amplitude * base_signal
            signal = amplitude * base_signal
            img_4d[x, y, z, :] = signal
            
            # Accumulate for expected result
            expected_weighted_sum += weight * signal
            total_weight += weight
        
        expected_weighted_mean = expected_weighted_sum / total_weight
        
        # Test weighted mean strategy directly
        from parcelextract.core.strategies import WeightedMeanExtractionStrategy
        strategy = WeightedMeanExtractionStrategy()
        
        # Call strategy directly with probabilistic weights
        actual_signal = strategy.extract(img_4d, weights_3d)
        
        # Check that extracted signal matches expected weighted mean
        np.testing.assert_allclose(
            actual_signal, expected_weighted_mean, rtol=1e-6,
            err_msg="Weighted mean extraction doesn't match expected weighted average"
        )
        
        # Additional check: correlation should be perfect
        correlation = np.corrcoef(actual_signal, expected_weighted_mean)[0, 1]
        assert correlation > 0.999, f"Correlation too low: {correlation:.6f}"

    def test_noise_robustness(self, temp_dir):
        """Test extraction robustness to different noise levels."""
        img_shape = (12, 12, 12, 40)
        n_timepoints = img_shape[3]
        
        # Create base signal
        t = np.arange(n_timepoints, dtype=np.float32)
        base_signal = 2.0 * np.sin(2 * np.pi * 0.1 * t) + 1.0 * np.cos(2 * np.pi * 0.25 * t)
        
        # Test different noise levels
        noise_levels = [0.0, 0.1, 0.5, 1.0]
        
        for noise_level in noise_levels:
            # Create data with signal + noise
            img_4d = np.zeros(img_shape, dtype=np.float32)
            atlas_3d = np.zeros((12, 12, 12), dtype=np.int16)
            
            # Define parcel region
            parcel_region = ((4, 8), (4, 8), (4, 8))
            x1, x2 = parcel_region[0]
            y1, y2 = parcel_region[1]
            z1, z2 = parcel_region[2]
            
            atlas_3d[x1:x2, y1:y2, z1:z2] = 1
            
            # Add signal + noise to all parcel voxels
            for x in range(x1, x2):
                for y in range(y1, y2):
                    for z in range(z1, z2):
                        noise = np.random.randn(n_timepoints) * noise_level
                        img_4d[x, y, z, :] = base_signal + noise
            
            # Save test data
            img_file = temp_dir / f"noise_{noise_level}_signal.nii.gz"
            atlas_file = temp_dir / f"noise_{noise_level}_atlas.nii.gz"
            
            nib.save(nib.Nifti1Image(img_4d, np.eye(4)), img_file)
            nib.save(nib.Nifti1Image(atlas_3d, np.eye(4)), atlas_file)
            
            # Test mean extraction (should reduce noise through averaging)
            extractor = ParcelExtractor(atlas=str(atlas_file), strategy='mean')
            timeseries = extractor.fit_transform(str(img_file))
            
            extracted_signal = timeseries[0, :]
            
            # Correlation with true signal should remain high even with noise
            correlation = np.corrcoef(base_signal, extracted_signal)[0, 1]
            
            # Expected correlation should decrease with noise but remain reasonable
            if noise_level == 0.0:
                assert correlation > 0.999, f"Perfect signal should have perfect correlation"
            elif noise_level <= 0.1:
                assert correlation > 0.95, f"Low noise should preserve signal well: corr={correlation:.3f}"
            elif noise_level <= 0.5:
                assert correlation > 0.8, f"Medium noise should still preserve signal: corr={correlation:.3f}"
            else:  # noise_level == 1.0
                assert correlation > 0.5, f"High noise should still show signal: corr={correlation:.3f}"

    def test_strategy_consistency_with_identical_data(self, temp_dir):
        """Test that different strategies give consistent results when appropriate."""
        img_shape = (10, 10, 10, 25)
        n_timepoints = img_shape[3]
        
        # Create identical signals in all voxels of a parcel
        t = np.arange(n_timepoints, dtype=np.float32)
        identical_signal = np.sin(2 * np.pi * 0.2 * t) + 0.5 * np.cos(2 * np.pi * 0.4 * t)
        
        img_4d = np.zeros(img_shape, dtype=np.float32)
        atlas_3d = np.zeros((10, 10, 10), dtype=np.int16)
        
        # Define parcel region
        parcel_region = ((3, 7), (3, 7), (3, 7))
        x1, x2 = parcel_region[0]
        y1, y2 = parcel_region[1]
        z1, z2 = parcel_region[2]
        
        atlas_3d[x1:x2, y1:y2, z1:z2] = 1
        
        # Set identical signal in all parcel voxels
        img_4d[x1:x2, y1:y2, z1:z2, :] = identical_signal[np.newaxis, np.newaxis, np.newaxis, :]
        
        # Save test data
        img_file = temp_dir / "identical_signal.nii.gz"
        atlas_file = temp_dir / "identical_atlas.nii.gz"
        
        nib.save(nib.Nifti1Image(img_4d, np.eye(4)), img_file)
        nib.save(nib.Nifti1Image(atlas_3d, np.eye(4)), atlas_file)
        
        # Test all strategies - they should give identical results for identical signals
        strategies = ['mean', 'median', 'weighted_mean']
        results = {}
        
        for strategy_name in strategies:
            extractor = ParcelExtractor(atlas=str(atlas_file), strategy=strategy_name)
            timeseries = extractor.fit_transform(str(img_file))
            results[strategy_name] = timeseries[0, :]
        
        # All strategies should give identical results
        for strategy1 in strategies:
            for strategy2 in strategies:
                if strategy1 != strategy2:
                    np.testing.assert_allclose(
                        results[strategy1], results[strategy2], rtol=1e-6,
                        err_msg=f"Strategies {strategy1} and {strategy2} should give identical results for identical signals"
                    )
        
        # All should match the original signal
        for strategy_name in strategies:
            np.testing.assert_allclose(
                results[strategy_name], identical_signal, rtol=1e-6,
                err_msg=f"Strategy {strategy_name} should exactly recover identical signal"
            )