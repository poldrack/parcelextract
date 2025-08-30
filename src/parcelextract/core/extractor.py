"""Main ParcelExtractor class."""

import numpy as np
import nibabel as nib

from .strategies import MeanExtractionStrategy


class ParcelExtractor:
    """Main class for extracting time-series signals from 4D neuroimaging data."""
    
    def __init__(self, atlas=None, strategy='mean'):
        self.atlas = atlas
        self.strategy = strategy

    def fit_transform(self, input_img):
        """Extract timeseries from 4D image using atlas parcellation."""
        # Load the atlas and get parcel information
        atlas_img = nib.load(self.atlas)
        atlas_data = atlas_img.get_fdata()
        n_parcels = int(np.max(atlas_data))  # Assumes parcels are numbered 1, 2, 3, ...
        
        # Load input image
        img = nib.load(input_img)
        img_data = img.get_fdata()
        n_timepoints = img_data.shape[-1]
        
        # Initialize extraction strategy (only mean for now)
        strategy = MeanExtractionStrategy()
        
        # Extract timeseries for each parcel
        timeseries_matrix = np.zeros((n_parcels, n_timepoints))
        
        for parcel_id in range(1, n_parcels + 1):  # Parcels numbered 1, 2, 3, ...
            # Create mask for this parcel
            parcel_mask = (atlas_data == parcel_id)
            
            if np.any(parcel_mask):  # Only extract if parcel has voxels
                timeseries_matrix[parcel_id - 1, :] = strategy.extract(img_data, parcel_mask)
        
        return timeseries_matrix