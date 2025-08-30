"""Atlas management functionality."""

from pathlib import Path
from typing import Union
import nibabel as nib
import numpy as np


class Atlas:
    """Represents a loaded atlas with data and metadata."""
    
    def __init__(self, data: np.ndarray, labels: list = None):
        self.data = data
        self.labels = labels or []


class AtlasManager:
    """Manages atlas loading and operations."""
    
    def load_atlas(self, atlas_spec: Union[str, Path]) -> Atlas:
        """
        Load atlas from file path.
        
        Parameters
        ----------
        atlas_spec : str or Path
            Path to atlas file (.nii or .nii.gz)
            
        Returns
        -------
        Atlas
            Loaded atlas object with data and labels
        """
        atlas_path = Path(atlas_spec)
        img = nib.load(atlas_path)
        data = img.get_fdata().astype(np.int16)
        
        # Extract unique labels from atlas data
        unique_labels = np.unique(data)
        unique_labels = unique_labels[unique_labels != 0]  # Remove background
        
        return Atlas(data=data, labels=unique_labels.tolist())
    
    def get_metadata(self, atlas_spec: Union[str, Path]) -> dict:
        """
        Get metadata for an atlas file.
        
        Parameters
        ----------
        atlas_spec : str or Path
            Path to atlas file
            
        Returns
        -------
        dict
            Dictionary containing atlas metadata
        """
        atlas_path = Path(atlas_spec)
        img = nib.load(atlas_path)
        data = img.get_fdata()
        
        # Extract unique labels (excluding background)
        unique_labels = np.unique(data)
        unique_labels = unique_labels[unique_labels != 0]
        
        metadata = {
            'shape': data.shape,
            'n_labels': len(unique_labels),
            'labels': unique_labels.tolist(),
            'dtype': str(data.dtype)
        }
        
        return metadata
    
    def validate_atlas(self, atlas_spec: Union[str, Path]) -> bool:
        """
        Validate that an atlas file is properly formatted.
        
        Parameters
        ----------
        atlas_spec : str or Path
            Path to atlas file to validate
            
        Returns
        -------
        bool
            True if atlas is valid, raises exception if invalid
            
        Raises
        ------
        FileNotFoundError
            If atlas file doesn't exist
        ValueError
            If atlas file is not a valid Nifti file or has invalid structure
        """
        atlas_path = Path(atlas_spec)
        
        if not atlas_path.exists():
            raise FileNotFoundError(f"Atlas file not found: {atlas_path}")
        
        try:
            img = nib.load(atlas_path)
            data = img.get_fdata()
            
            # Check that atlas has at least 3 dimensions
            if len(data.shape) < 3:
                raise ValueError(f"Atlas must be 3D or 4D, got {len(data.shape)}D")
            
            # Check that atlas contains integer labels
            if not np.issubdtype(data.dtype, np.integer) and not np.allclose(data, data.astype(int)):
                raise ValueError("Atlas must contain integer labels")
                
            # Check that atlas has at least one non-zero label
            unique_labels = np.unique(data)
            non_zero_labels = unique_labels[unique_labels != 0]
            if len(non_zero_labels) == 0:
                raise ValueError("Atlas must contain at least one non-zero label")
                
            return True
            
        except Exception as e:
            if isinstance(e, (FileNotFoundError, ValueError)):
                raise
            else:
                raise ValueError(f"Invalid atlas file format: {e}")