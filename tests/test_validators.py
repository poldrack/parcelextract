"""Test input validation functions."""

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from parcelextract.core.validators import (
    ValidationError,
    validate_atlas_spec,
    validate_input_image,
    validate_output_dir,
)


class TestValidateInputImage:
    """Test input image validation."""

    def test_validate_4d_nifti_file_valid(self, synthetic_4d_nifti):
        """Test validation of valid 4D Nifti file."""
        # Should not raise any exception
        img = validate_input_image(synthetic_4d_nifti)
        assert img.ndim == 4
        assert img.shape[-1] == 50  # 50 timepoints

    def test_validate_4d_nifti_object_valid(self, synthetic_4d_data):
        """Test validation of valid 4D Nifti image object."""
        affine = np.eye(4)
        img = nib.Nifti1Image(synthetic_4d_data, affine)
        
        validated_img = validate_input_image(img)
        assert validated_img.ndim == 4
        assert validated_img.shape == synthetic_4d_data.shape

    def test_validate_3d_image_raises_error(self, synthetic_3d_nifti):
        """Test that 3D images raise ValidationError."""
        with pytest.raises(ValidationError, match="must be 4D"):
            validate_input_image(synthetic_3d_nifti)

    def test_validate_nonexistent_file_raises_error(self, non_existent_file):
        """Test that non-existent files raise ValidationError."""
        with pytest.raises(ValidationError, match="does not exist"):
            validate_input_image(non_existent_file)

    def test_validate_invalid_nifti_raises_error(self, invalid_nifti_file):
        """Test that invalid Nifti files raise ValidationError."""
        with pytest.raises(ValidationError, match="not a valid Nifti"):
            validate_input_image(invalid_nifti_file)

    def test_validate_string_path_conversion(self, synthetic_4d_nifti):
        """Test that string paths are converted to Path objects."""
        img = validate_input_image(str(synthetic_4d_nifti))
        assert img.ndim == 4

    def test_validate_none_input_raises_error(self):
        """Test that None input raises ValidationError."""
        with pytest.raises(ValidationError, match="Input image cannot be None"):
            validate_input_image(None)

    def test_validate_empty_4d_raises_error(self):
        """Test that empty 4D image raises ValidationError."""
        empty_data = np.zeros((10, 10, 10, 0))  # 0 timepoints
        affine = np.eye(4)
        img = nib.Nifti1Image(empty_data, affine)
        
        with pytest.raises(ValidationError, match="must have at least 1 timepoint"):
            validate_input_image(img)


class TestValidateAtlasSpec:
    """Test atlas specification validation."""

    def test_validate_atlas_file_valid(self, test_atlas_nifti):
        """Test validation of valid atlas file."""
        atlas_img = validate_atlas_spec(test_atlas_nifti)
        assert atlas_img.ndim == 3
        assert np.max(atlas_img.get_fdata()) == 5  # 5 parcels

    def test_validate_atlas_string_templateflow(self):
        """Test validation of TemplateFlow atlas string."""
        # This should not raise an error for known atlas names
        atlas_spec = "schaefer2018"
        result = validate_atlas_spec(atlas_spec)
        assert result == atlas_spec

    def test_validate_atlas_nonexistent_file_raises_error(self, non_existent_file):
        """Test that non-existent atlas files raise ValidationError."""
        with pytest.raises(ValidationError, match="Atlas file does not exist"):
            validate_atlas_spec(non_existent_file)

    def test_validate_atlas_4d_raises_error(self, synthetic_4d_nifti):
        """Test that 4D atlas raises ValidationError."""
        with pytest.raises(ValidationError, match="Atlas must be 3D"):
            validate_atlas_spec(synthetic_4d_nifti)

    def test_validate_atlas_none_raises_error(self):
        """Test that None atlas raises ValidationError."""
        with pytest.raises(ValidationError, match="Atlas specification cannot be None"):
            validate_atlas_spec(None)

    def test_validate_atlas_empty_string_raises_error(self):
        """Test that empty string atlas raises ValidationError."""
        with pytest.raises(ValidationError, match="Atlas specification cannot be empty"):
            validate_atlas_spec("")

    def test_validate_atlas_negative_values_raises_error(self, temp_dir):
        """Test that atlas with negative values raises ValidationError."""
        atlas_data = np.array([[[1, 2], [3, -1]], [[4, 5], [6, 7]]], dtype=np.float32)
        affine = np.eye(4)
        img = nib.Nifti1Image(atlas_data, affine)
        filepath = temp_dir / "negative_atlas.nii.gz"
        nib.save(img, filepath)
        
        with pytest.raises(ValidationError, match="Atlas cannot contain negative values"):
            validate_atlas_spec(filepath)

    def test_validate_atlas_all_zeros_raises_error(self, temp_dir):
        """Test that atlas with all zeros raises ValidationError."""
        atlas_data = np.zeros((10, 10, 10))
        affine = np.eye(4)
        img = nib.Nifti1Image(atlas_data, affine)
        filepath = temp_dir / "zeros_atlas.nii.gz"
        nib.save(img, filepath)
        
        with pytest.raises(ValidationError, match="Atlas must contain at least one parcel"):
            validate_atlas_spec(filepath)


class TestValidateOutputDir:
    """Test output directory validation."""

    def test_validate_existing_directory(self, temp_dir):
        """Test validation of existing directory."""
        validated_dir = validate_output_dir(temp_dir)
        assert validated_dir == temp_dir
        assert validated_dir.is_dir()

    def test_validate_nonexistent_directory_creates(self, temp_dir):
        """Test that non-existent directory is created."""
        new_dir = temp_dir / "new_output_dir"
        assert not new_dir.exists()
        
        validated_dir = validate_output_dir(new_dir)
        assert validated_dir.is_dir()
        assert validated_dir == new_dir

    def test_validate_string_path_conversion(self, temp_dir):
        """Test that string paths are converted to Path objects."""
        validated_dir = validate_output_dir(str(temp_dir))
        assert isinstance(validated_dir, Path)
        assert validated_dir.is_dir()

    def test_validate_none_output_dir_raises_error(self):
        """Test that None output directory raises ValidationError."""
        with pytest.raises(ValidationError, match="Output directory cannot be None"):
            validate_output_dir(None)

    def test_validate_file_as_output_dir_raises_error(self, temp_dir):
        """Test that existing file raises ValidationError when used as output dir."""
        filepath = temp_dir / "existing_file.txt"
        filepath.write_text("test content")
        
        with pytest.raises(ValidationError, match="Output directory path exists as a file"):
            validate_output_dir(filepath)

    def test_validate_nested_directory_creation(self, temp_dir):
        """Test that nested directories are created."""
        nested_dir = temp_dir / "level1" / "level2" / "level3"
        assert not nested_dir.exists()
        
        validated_dir = validate_output_dir(nested_dir)
        assert validated_dir.is_dir()
        assert validated_dir == nested_dir