"""Input validation functions for parcelextract."""

from pathlib import Path
from typing import Union

import nibabel as nib
import numpy as np


class ValidationError(Exception):
    """Custom exception for validation errors."""

    pass


def validate_input_image(
    input_img: Union[str, Path, nib.Nifti1Image, None]
) -> nib.Nifti1Image:
    """
    Validate input 4D neuroimaging data.

    Parameters
    ----------
    input_img : str, Path, nibabel.Nifti1Image, or None
        Input image to validate. Can be a file path or nibabel image object.

    Returns
    -------
    nibabel.Nifti1Image
        Validated 4D neuroimaging image.

    Raises
    ------
    ValidationError
        If the input image is invalid, not 4D, or has no timepoints.
    """
    if input_img is None:
        raise ValidationError("Input image cannot be None")

    # Handle file path input
    if isinstance(input_img, (str, Path)):
        img_path = Path(input_img)
        
        if not img_path.exists():
            raise ValidationError(f"Input image file does not exist: {img_path}")
        
        try:
            img = nib.load(img_path)
        except Exception as e:
            raise ValidationError(f"File is not a valid Nifti: {e}")
    
    # Handle nibabel image object input
    elif isinstance(input_img, nib.Nifti1Image):
        img = input_img
    else:
        raise ValidationError(f"Input image must be str, Path, or nibabel image, got {type(input_img)}")

    # Validate dimensions
    if img.ndim != 4:
        raise ValidationError(f"Input image must be 4D, got {img.ndim}D")

    # Check for timepoints
    if img.shape[-1] == 0:
        raise ValidationError("Input image must have at least 1 timepoint")

    return img


def validate_atlas_spec(
    atlas_spec: Union[str, Path, nib.Nifti1Image, None]
) -> Union[str, nib.Nifti1Image]:
    """
    Validate atlas specification.

    Parameters
    ----------
    atlas_spec : str, Path, nibabel.Nifti1Image, or None
        Atlas specification. Can be a TemplateFlow atlas name, file path, 
        or nibabel image object.

    Returns
    -------
    str or nibabel.Nifti1Image
        Validated atlas specification (string for TemplateFlow, image for files).

    Raises
    ------
    ValidationError
        If the atlas specification is invalid.
    """
    if atlas_spec is None:
        raise ValidationError("Atlas specification cannot be None")

    # Handle string input (TemplateFlow atlas name)
    if isinstance(atlas_spec, str):
        if not atlas_spec.strip():
            raise ValidationError("Atlas specification cannot be empty")
        return atlas_spec
    
    # Handle file path input
    elif isinstance(atlas_spec, Path):
        if not atlas_spec.exists():
            raise ValidationError(f"Atlas file does not exist: {atlas_spec}")
        
        try:
            img = nib.load(atlas_spec)
        except Exception as e:
            raise ValidationError(f"Atlas file is not a valid Nifti: {e}")
        
        # Validate atlas properties
        _validate_atlas_image(img)
        return img
    
    # Handle nibabel image object input
    elif isinstance(atlas_spec, nib.Nifti1Image):
        _validate_atlas_image(atlas_spec)
        return atlas_spec
    
    else:
        raise ValidationError(f"Atlas must be str, Path, or nibabel image, got {type(atlas_spec)}")


def _validate_atlas_image(img: nib.Nifti1Image) -> None:
    """
    Validate atlas image properties.

    Parameters
    ----------
    img : nibabel.Nifti1Image
        Atlas image to validate.

    Raises
    ------
    ValidationError
        If the atlas image is invalid.
    """
    # Check dimensions
    if img.ndim != 3:
        raise ValidationError(f"Atlas must be 3D, got {img.ndim}D")
    
    # Get atlas data
    atlas_data = img.get_fdata()
    
    # Check for negative values
    if np.any(atlas_data < 0):
        raise ValidationError("Atlas cannot contain negative values")
    
    # Check for at least one parcel (non-zero values)
    if np.all(atlas_data == 0):
        raise ValidationError("Atlas must contain at least one parcel (non-zero values)")


def validate_output_dir(output_dir: Union[str, Path, None]) -> Path:
    """
    Validate and create output directory if needed.

    Parameters
    ----------
    output_dir : str, Path, or None
        Output directory path.

    Returns
    -------
    Path
        Validated output directory path.

    Raises
    ------
    ValidationError
        If the output directory path is invalid or cannot be created.
    """
    if output_dir is None:
        raise ValidationError("Output directory cannot be None")

    dir_path = Path(output_dir)
    
    # Check if path exists as a file (not a directory)
    if dir_path.exists() and dir_path.is_file():
        raise ValidationError(f"Output directory path exists as a file: {dir_path}")
    
    # Create directory if it doesn't exist
    if not dir_path.exists():
        try:
            dir_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise ValidationError(f"Cannot create output directory {dir_path}: {e}")
    
    return dir_path