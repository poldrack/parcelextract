"""Main ParcelExtractor class."""

import numpy as np
import nibabel as nib

from .strategies import (
    MeanExtractionStrategy,
    MedianExtractionStrategy,
    PCAExtractionStrategy,
    WeightedMeanExtractionStrategy,
)
from .validators import validate_input_image, validate_spatial_compatibility, detect_image_resolution


class ParcelExtractor:
    """Main class for extracting time-series signals from 4D neuroimaging data."""
    
    def __init__(self, atlas=None, strategy='mean'):
        self.atlas = atlas
        self.strategy = strategy
        self._atlas_data = None
        self._is_probabilistic = None

    def fit_transform(self, input_img):
        """Extract timeseries from 4D image using atlas parcellation."""
        # Validate input image first
        validated_img = validate_input_image(input_img)
        
        # If atlas is a TemplateFlow name, resolve it with auto-resolution matching
        if self._is_templateflow_atlas():
            self.atlas = self._resolve_templateflow_atlas(validated_img)
        
        # Load atlas data
        atlas_data = self._load_atlas_data()
        atlas_img = nib.load(self.atlas)
        
        # Validate spatial compatibility
        validate_spatial_compatibility(validated_img, atlas_img)
        
        # Use validated image data
        img_data = validated_img.get_fdata()
        n_timepoints = img_data.shape[-1]
        
        # Initialize extraction strategy
        strategy = self._get_strategy()
        
        if self.is_probabilistic_atlas():
            # Handle probabilistic atlas
            if len(atlas_data.shape) == 4:
                # 4D probabilistic atlas (e.g., DiFuMo): each volume is a probability map
                n_parcels = atlas_data.shape[3]
                timeseries_matrix = np.zeros((n_parcels, n_timepoints))
                
                for parcel_id in range(n_parcels):
                    # Extract probability map for this parcel
                    parcel_weights = atlas_data[:, :, :, parcel_id]
                    
                    # Only extract if there are non-zero weights
                    if np.any(parcel_weights > 0):
                        timeseries_matrix[parcel_id, :] = strategy.extract(img_data, parcel_weights)
            else:
                # 3D probabilistic atlas: single volume with continuous weights
                timeseries_matrix = np.zeros((1, n_timepoints))
                timeseries_matrix[0, :] = strategy.extract(img_data, atlas_data)
            
        else:
            # For discrete atlas, extract each parcel separately
            n_parcels = int(np.max(atlas_data))  # Assumes parcels are numbered 1, 2, 3, ...
            timeseries_matrix = np.zeros((n_parcels, n_timepoints))
            
            for parcel_id in range(1, n_parcels + 1):  # Parcels numbered 1, 2, 3, ...
                # Create binary mask for this parcel
                parcel_mask = (atlas_data == parcel_id)
                
                if np.any(parcel_mask):  # Only extract if parcel has voxels
                    timeseries_matrix[parcel_id - 1, :] = strategy.extract(img_data, parcel_mask)
        
        return timeseries_matrix

    def _load_atlas_data(self):
        """Load and cache atlas data."""
        if self._atlas_data is None:
            atlas_img = nib.load(self.atlas)
            self._atlas_data = atlas_img.get_fdata()
        return self._atlas_data

    def is_probabilistic_atlas(self):
        """
        Determine if the atlas contains probabilistic (continuous) values.
        
        Returns
        -------
        bool
            True if atlas is probabilistic, False if discrete
        """
        if self._is_probabilistic is None:
            atlas_data = self._load_atlas_data()
            
            # Check if atlas has any non-zero values
            non_zero_values = atlas_data[atlas_data > 0]
            if len(non_zero_values) == 0:
                raise ValueError("Atlas contains no non-zero values")
            
            # First check: 4D atlases are inherently probabilistic
            # Each volume in the 4th dimension represents a probability map for that component
            if len(atlas_data.shape) == 4:
                self._is_probabilistic = True
                return self._is_probabilistic
            
            # For 3D atlases, check if values are effectively integers (within tolerance)
            tolerance = 1e-6
            is_integer_like = np.allclose(non_zero_values, np.round(non_zero_values), atol=tolerance)
            
            # Additional checks for probabilistic characteristics:
            # 1. Contains values between 0 and 1 (excluding exactly 0 and 1)
            has_fractional_values = np.any((non_zero_values > 0) & (non_zero_values < 1) & 
                                         ~np.isclose(non_zero_values, 1.0, atol=tolerance))
            
            # 2. Has many unique values (suggests continuous distribution)
            unique_values = np.unique(non_zero_values)
            has_many_unique_values = len(unique_values) > 10
            
            # 3D Atlas is probabilistic if:
            # - Values are not integer-like AND (has fractional values OR many unique values)
            self._is_probabilistic = (not is_integer_like) and (has_fractional_values or has_many_unique_values)
        
        return self._is_probabilistic

    def get_effective_strategy(self):
        """
        Get the strategy that will actually be used for extraction.
        
        For probabilistic atlases, this returns 'weighted_mean' regardless of
        the original strategy setting.
        
        Returns
        -------
        str
            The effective strategy name
        """
        if self.is_probabilistic_atlas():
            return 'weighted_mean'
        else:
            return self.strategy

    def _is_templateflow_atlas(self) -> bool:
        """Check if atlas is a TemplateFlow atlas name rather than a file path."""
        from pathlib import Path
        
        # If it's a file path that exists, it's not a TemplateFlow name
        atlas_path = Path(self.atlas)
        if atlas_path.exists():
            return False
            
        # Check if it looks like a TemplateFlow atlas name
        templateflow_patterns = [
            'schaefer2018', 'aal', 'harvardoxford', 'destrieux', 'desikankilliany', 'difumo'
        ]
        
        return any(pattern.lower() in self.atlas.lower() for pattern in templateflow_patterns)
    
    def _resolve_templateflow_atlas(self, input_img) -> str:
        """Resolve TemplateFlow atlas name to file path with auto-resolution matching."""
        try:
            from ..atlases.templateflow import TemplateFlowManager
            tf_manager = TemplateFlowManager()
            
            # Use default space - this could be made configurable
            space = 'MNI152NLin2009cAsym'
            
            # Get atlas with auto-detected resolution
            return tf_manager.get_atlas(self.atlas, space, input_img=input_img)
            
        except Exception as e:
            # Fall back to original atlas if resolution matching fails
            print(f"Warning: TemplateFlow resolution matching failed: {e}")
            return self.atlas

    def _get_strategy(self):
        """Get the extraction strategy instance based on effective strategy."""
        effective_strategy = self.get_effective_strategy()
        
        if effective_strategy == 'mean':
            return MeanExtractionStrategy()
        elif effective_strategy == 'median':
            return MedianExtractionStrategy()
        elif effective_strategy == 'pca':
            return PCAExtractionStrategy()
        elif effective_strategy == 'weighted_mean':
            return WeightedMeanExtractionStrategy()
        else:
            raise ValueError(f"Unknown strategy: {effective_strategy}")