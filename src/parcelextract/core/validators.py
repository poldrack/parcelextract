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


def validate_spatial_compatibility(input_img: nib.Nifti1Image, atlas_img: nib.Nifti1Image) -> None:
    """
    Validate that input image and atlas have compatible spatial dimensions.
    
    Parameters
    ----------
    input_img : nibabel.Nifti1Image
        4D input neuroimaging data
    atlas_img : nibabel.Nifti1Image  
        3D atlas data
        
    Raises
    ------
    ValidationError
        If spatial dimensions don't match or affine matrices are incompatible
    """
    # Get spatial dimensions (first 3 dimensions)
    input_spatial_shape = input_img.shape[:3]
    atlas_spatial_shape = atlas_img.shape[:3]
    
    if input_spatial_shape != atlas_spatial_shape:
        raise ValidationError(
            f"Input image and atlas have incompatible spatial dimensions. "
            f"Input: {input_spatial_shape}, Atlas: {atlas_spatial_shape}. "
            f"Images must be in the same coordinate space (e.g., both in MNI152NLin2009cAsym space). "
            f"Consider resampling your input image to match the atlas space using tools like "
            f"nilearn.image.resample_to_img() or FSL's flirt/applywarp."
        )
    
    # Check if affine matrices are reasonably similar (allowing for small floating point differences)
    input_affine = input_img.affine
    atlas_affine = atlas_img.affine
    
    # Compare affine matrices with some tolerance for floating point precision
    if not np.allclose(input_affine, atlas_affine, rtol=1e-3, atol=1e-3):
        # Calculate the maximum difference for reporting
        max_diff = np.max(np.abs(input_affine - atlas_affine))
        
        # Only warn if the difference is significant (more than just floating point precision)
        if max_diff > 1e-2:  # 0.01 threshold for significant differences
            import warnings
            warnings.warn(
                f"Input image and atlas have different affine matrices. "
                f"Maximum difference: {max_diff:.6f}. "
                f"This may indicate the images are not in the same coordinate space. "
                f"Results may be inaccurate if images are not properly aligned.",
                UserWarning
            )


def detect_image_resolution(img: nib.Nifti1Image) -> int:
    """
    Detect the spatial resolution (voxel size) of a neuroimaging volume.
    
    Parameters
    ----------
    img : nibabel.Nifti1Image
        Input neuroimaging image
        
    Returns
    -------
    int
        Detected resolution in mm (1, 2, or 3), rounded to nearest integer
        
    Raises
    ------
    ValidationError
        If resolution cannot be determined or is invalid
    """
    try:
        # Get voxel sizes from the affine matrix
        voxel_sizes = nib.affines.voxel_sizes(img.affine)
        
        # Take the mean of the first 3 dimensions (spatial)
        mean_voxel_size = np.mean(voxel_sizes[:3])
        
        # Round to nearest integer mm
        resolution = int(round(mean_voxel_size))
        
        # Validate that it's a reasonable resolution
        if resolution < 1 or resolution > 10:
            raise ValidationError(
                f"Detected unusual voxel resolution: {resolution}mm "
                f"(voxel sizes: {voxel_sizes[:3]}). "
                f"Expected values between 1-10mm."
            )
        
        return resolution
        
    except Exception as e:
        if isinstance(e, ValidationError):
            raise
        raise ValidationError(
            f"Could not determine image resolution from affine matrix: {e}. "
            f"Affine matrix: {img.affine}"
        )