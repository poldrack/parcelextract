"""Signal extraction strategies for parcel-wise analysis."""

from abc import ABC, abstractmethod
from typing import Union

import numpy as np
from sklearn.decomposition import PCA


class ExtractionStrategy(ABC):
    """
    Abstract base class for signal extraction strategies.
    
    All extraction strategies must implement the extract method to
    extract a representative timeseries from a parcel's voxels.
    """

    @abstractmethod
    def extract(self, data_4d: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Extract representative timeseries from parcel voxels.

        Parameters
        ----------
        data_4d : np.ndarray
            4D neuroimaging data with shape (x, y, z, timepoints).
        mask : np.ndarray
            3D boolean array or probabilistic weights with shape (x, y, z).

        Returns
        -------
        np.ndarray
            1D timeseries array with shape (timepoints,).

        Raises
        ------
        ValueError
            If the mask is invalid or no voxels are selected.
        """
        pass


class MeanExtractionStrategy(ExtractionStrategy):
    """Extract mean signal across parcel voxels."""

    def extract(self, data_4d: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Extract mean timeseries from parcel voxels.

        Parameters
        ----------
        data_4d : np.ndarray
            4D neuroimaging data with shape (x, y, z, timepoints).
        mask : np.ndarray
            3D boolean array with shape (x, y, z) indicating parcel voxels.

        Returns
        -------
        np.ndarray
            1D timeseries array with shape (timepoints,) containing mean signal.

        Raises
        ------
        ValueError
            If no voxels are selected by the mask.
        """
        # Check if mask selects any voxels
        if not np.any(mask):
            raise ValueError("No voxels selected by the mask")

        # Extract timeseries from masked voxels
        parcel_data = data_4d[mask]  # Shape: (n_voxels, timepoints)
        
        # Calculate mean across voxels (axis=0), handling NaN values
        mean_timeseries = np.nanmean(parcel_data, axis=0)
        
        return mean_timeseries


class MedianExtractionStrategy(ExtractionStrategy):
    """Extract median signal across parcel voxels."""

    def extract(self, data_4d: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Extract median timeseries from parcel voxels.

        Parameters
        ----------
        data_4d : np.ndarray
            4D neuroimaging data with shape (x, y, z, timepoints).
        mask : np.ndarray
            3D boolean array with shape (x, y, z) indicating parcel voxels.

        Returns
        -------
        np.ndarray
            1D timeseries array with shape (timepoints,) containing median signal.

        Raises
        ------
        ValueError
            If no voxels are selected by the mask.
        """
        # Check if mask selects any voxels
        if not np.any(mask):
            raise ValueError("No voxels selected by the mask")

        # Extract timeseries from masked voxels
        parcel_data = data_4d[mask]  # Shape: (n_voxels, timepoints)
        
        # Calculate median across voxels (axis=0), handling NaN values
        median_timeseries = np.nanmedian(parcel_data, axis=0)
        
        return median_timeseries


class PCAExtractionStrategy(ExtractionStrategy):
    """Extract first principal component from parcel voxels."""

    def extract(self, data_4d: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Extract first principal component timeseries from parcel voxels.

        Parameters
        ----------
        data_4d : np.ndarray
            4D neuroimaging data with shape (x, y, z, timepoints).
        mask : np.ndarray
            3D boolean array with shape (x, y, z) indicating parcel voxels.

        Returns
        -------
        np.ndarray
            1D timeseries array with shape (timepoints,) containing first PC.

        Raises
        ------
        ValueError
            If no voxels are selected, insufficient timepoints for PCA,
            or PCA decomposition fails.
        """
        # Check if mask selects any voxels
        if not np.any(mask):
            raise ValueError("No voxels selected by the mask")

        # Extract timeseries from masked voxels
        parcel_data = data_4d[mask]  # Shape: (n_voxels, timepoints)
        
        n_voxels, n_timepoints = parcel_data.shape
        
        # PCA requires at least 2 timepoints
        if n_timepoints < 2:
            raise ValueError("PCA requires at least 2 timepoints")
        
        # Handle single voxel case
        if n_voxels == 1:
            # For single voxel, return normalized timeseries
            timeseries = parcel_data[0]
            # Center and normalize
            centered = timeseries - np.nanmean(timeseries)
            norm = np.linalg.norm(centered)
            if norm > 0:
                return centered / norm
            else:
                return centered
        
        # Transpose to shape (timepoints, voxels) for PCA
        parcel_data_t = parcel_data.T
        
        # Remove any voxels that are all NaN or constant
        valid_voxels = []
        for i in range(n_voxels):
            voxel_data = parcel_data_t[:, i]
            if not np.all(np.isnan(voxel_data)) and np.nanstd(voxel_data) > 1e-10:
                valid_voxels.append(i)
        
        if len(valid_voxels) == 0:
            raise ValueError("No valid voxels for PCA (all NaN or constant)")
        
        # Use only valid voxels
        valid_data = parcel_data_t[:, valid_voxels]
        
        # Handle NaN values by mean imputation within each voxel
        for i in range(valid_data.shape[1]):
            voxel_data = valid_data[:, i]
            nan_mask = np.isnan(voxel_data)
            if np.any(nan_mask) and not np.all(nan_mask):
                valid_data[nan_mask, i] = np.nanmean(voxel_data)
        
        # Check if any NaNs remain
        if np.any(np.isnan(valid_data)):
            raise ValueError("Cannot perform PCA: data contains NaN values after imputation")
        
        try:
            # Perform PCA
            pca = PCA(n_components=1)
            pca.fit(valid_data)
            
            # Extract first principal component timeseries
            pc1_timeseries = pca.transform(valid_data)[:, 0]
            
            return pc1_timeseries
        
        except Exception as e:
            raise ValueError(f"PCA decomposition failed: {e}")


class WeightedMeanExtractionStrategy(ExtractionStrategy):
    """Extract weighted mean signal using probabilistic weights."""

    def extract(self, data_4d: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        Extract weighted mean timeseries from parcel voxels.

        Parameters
        ----------
        data_4d : np.ndarray
            4D neuroimaging data with shape (x, y, z, timepoints).
        weights : np.ndarray
            3D array with shape (x, y, z) containing probabilistic weights.

        Returns
        -------
        np.ndarray
            1D timeseries array with shape (timepoints,) containing weighted mean.

        Raises
        ------
        ValueError
            If weights are invalid, negative, or sum to zero.
        """
        # Check for negative weights
        if np.any(weights < 0):
            raise ValueError("Weights cannot be negative")
        
        # Check if any weights are positive
        total_weight = np.sum(weights)
        if total_weight == 0:
            raise ValueError("Sum of weights is zero, no voxels contribute to extraction")
        
        # Get indices where weights > 0
        weight_mask = weights > 0
        
        if not np.any(weight_mask):
            raise ValueError("No voxels have positive weights")
        
        # Extract data and weights for contributing voxels
        contributing_data = data_4d[weight_mask]  # Shape: (n_contributing, timepoints)
        contributing_weights = weights[weight_mask]  # Shape: (n_contributing,)
        
        # Calculate weighted mean
        # Expand weights to match data dimensions: (n_contributing, 1)
        weights_expanded = contributing_weights[:, np.newaxis]
        
        # Calculate weighted sum and normalize by total weight
        weighted_sum = np.sum(contributing_data * weights_expanded, axis=0)
        weighted_mean = weighted_sum / total_weight
        
        return weighted_mean