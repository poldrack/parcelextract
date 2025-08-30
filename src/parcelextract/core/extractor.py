"""Main ParcelExtractor class."""

import numpy as np
import nibabel as nib

from .strategies import (
    MeanExtractionStrategy,
    MedianExtractionStrategy,
    PCAExtractionStrategy,
    WeightedMeanExtractionStrategy,
)
from .validators import validate_input_image


class ParcelExtractor:
    """Main class for extracting time-series signals from 4D neuroimaging data."""
    
    def __init__(self, atlas=None, strategy='mean'):
        self.atlas = atlas
        self.strategy = strategy

    def fit_transform(self, input_img):
        """Extract timeseries from 4D image using atlas parcellation."""
        # Validate input image first
        validated_img = validate_input_image(input_img)
        
        # Load the atlas and get parcel information
        atlas_img = nib.load(self.atlas)
        atlas_data = atlas_img.get_fdata()
        n_parcels = int(np.max(atlas_data))  # Assumes parcels are numbered 1, 2, 3, ...
        
        # Use validated image data
        img_data = validated_img.get_fdata()
        n_timepoints = img_data.shape[-1]
        
        # Initialize extraction strategy based on parameter
        strategy = self._get_strategy()
        
        # Extract timeseries for each parcel
        timeseries_matrix = np.zeros((n_parcels, n_timepoints))
        
        for parcel_id in range(1, n_parcels + 1):  # Parcels numbered 1, 2, 3, ...
            # Create mask for this parcel
            parcel_mask = (atlas_data == parcel_id)
            
            if np.any(parcel_mask):  # Only extract if parcel has voxels
                timeseries_matrix[parcel_id - 1, :] = strategy.extract(img_data, parcel_mask)
        
        return timeseries_matrix

    def _get_strategy(self):
        """Get the extraction strategy instance based on strategy parameter."""
        if self.strategy == 'mean':
            return MeanExtractionStrategy()
        elif self.strategy == 'median':
            return MedianExtractionStrategy()
        elif self.strategy == 'pca':
            return PCAExtractionStrategy()
        elif self.strategy == 'weighted_mean':
            return WeightedMeanExtractionStrategy()
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")