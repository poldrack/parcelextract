"""File reading utilities for neuroimaging data."""

from pathlib import Path
from typing import Dict, Optional, Union

import nibabel as nib
import numpy as np


class InvalidNiftiError(Exception):
    """Custom exception for invalid Nifti files."""

    pass


def load_nifti(filepath: Union[str, Path, None]) -> nib.Nifti1Image:
    """
    Load a Nifti file from disk.

    Parameters
    ----------
    filepath : str, Path, or None
        Path to the Nifti file (.nii or .nii.gz).

    Returns
    -------
    nibabel.Nifti1Image
        Loaded Nifti image.

    Raises
    ------
    InvalidNiftiError
        If the file path is invalid or the file cannot be loaded.
    """
    if filepath is None:
        raise InvalidNiftiError("File path cannot be None")

    file_path = Path(filepath)

    if not file_path.exists():
        raise InvalidNiftiError(f"File does not exist: {file_path}")

    try:
        img = nib.load(file_path)
    except Exception as e:
        raise InvalidNiftiError(f"Failed to load Nifti file {file_path}: {e}")

    return img


def get_image_metadata(img: nib.Nifti1Image) -> Dict[str, Union[int, tuple, str, float, None]]:
    """
    Extract metadata from a Nifti image.

    Parameters
    ----------
    img : nibabel.Nifti1Image
        Nifti image object.

    Returns
    -------
    dict
        Dictionary containing image metadata including dimensions, shape,
        voxel size, dtype, TR, and units information.
    """
    header = img.header
    
    # Basic shape and dimension info
    shape = img.shape
    ndim = img.ndim
    dtype = str(img.get_fdata().dtype)
    
    # Number of timepoints (None for 3D images)
    n_timepoints = shape[-1] if ndim == 4 else None
    
    # Voxel sizes from affine matrix diagonal
    affine = img.affine
    voxel_size = (
        abs(float(affine[0, 0])),
        abs(float(affine[1, 1])),
        abs(float(affine[2, 2]))
    )
    
    # Repetition time (TR) - pixdim[4] if temporal units are set
    tr = None
    if ndim == 4:
        tr_raw = float(header["pixdim"][4])
        # Only consider as valid TR if it's positive and temporal units are set
        if tr_raw > 0:
            tr = tr_raw
        else:
            tr = None
    
    # Spatial and temporal units
    spatial_units = None
    temporal_units = None
    
    try:
        units = header.get_xyzt_units()
        spatial_units = units[0] if units[0] else None
        temporal_units = units[1] if units[1] else None
    except Exception:
        # If units extraction fails, set to None
        pass
    
    metadata = {
        "ndim": ndim,
        "shape": shape,
        "n_timepoints": n_timepoints,
        "voxel_size": voxel_size,
        "dtype": dtype,
        "tr": tr,
        "spatial_units": spatial_units,
        "temporal_units": temporal_units,
    }
    
    return metadata


def validate_4d_image(img: Optional[nib.Nifti1Image]) -> None:
    """
    Validate that an image is 4D with at least one timepoint.

    Parameters
    ----------
    img : nibabel.Nifti1Image or None
        Image to validate.

    Raises
    ------
    InvalidNiftiError
        If the image is not 4D or has invalid properties.
    """
    if img is None:
        raise InvalidNiftiError("Image cannot be None")

    if img.ndim != 4:
        raise InvalidNiftiError(f"Image must be 4D, got {img.ndim}D")

    if img.shape[-1] == 0:
        raise InvalidNiftiError("Image must have at least 1 timepoint")