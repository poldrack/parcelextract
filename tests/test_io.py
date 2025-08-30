"""Test file I/O operations."""

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from parcelextract.io.readers import (
    InvalidNiftiError,
    get_image_metadata,
    load_nifti,
    validate_4d_image,
)


class TestLoadNifti:
    """Test Nifti file loading functionality."""

    def test_load_valid_4d_nifti_file(self, synthetic_4d_nifti):
        """Test loading a valid 4D Nifti file."""
        img = load_nifti(synthetic_4d_nifti)
        assert isinstance(img, nib.Nifti1Image)
        assert img.ndim == 4
        assert img.shape[-1] == 50  # 50 timepoints

    def test_load_valid_3d_nifti_file(self, synthetic_3d_nifti):
        """Test loading a valid 3D Nifti file."""
        img = load_nifti(synthetic_3d_nifti)
        assert isinstance(img, nib.Nifti1Image)
        assert img.ndim == 3

    def test_load_string_path(self, synthetic_4d_nifti):
        """Test loading with string path instead of Path object."""
        img = load_nifti(str(synthetic_4d_nifti))
        assert isinstance(img, nib.Nifti1Image)
        assert img.ndim == 4

    def test_load_nonexistent_file_raises_error(self, non_existent_file):
        """Test that loading non-existent file raises InvalidNiftiError."""
        with pytest.raises(InvalidNiftiError, match="File does not exist"):
            load_nifti(non_existent_file)

    def test_load_invalid_nifti_raises_error(self, invalid_nifti_file):
        """Test that loading invalid Nifti file raises InvalidNiftiError."""
        with pytest.raises(InvalidNiftiError, match="Failed to load Nifti file"):
            load_nifti(invalid_nifti_file)

    def test_load_none_input_raises_error(self):
        """Test that None input raises InvalidNiftiError."""
        with pytest.raises(InvalidNiftiError, match="File path cannot be None"):
            load_nifti(None)

    def test_load_preserves_affine(self, synthetic_4d_nifti):
        """Test that loading preserves the affine transformation."""
        img = load_nifti(synthetic_4d_nifti)
        expected_affine = np.eye(4)
        expected_affine[0, 0] = 3.0
        expected_affine[1, 1] = 3.0
        expected_affine[2, 2] = 3.0
        
        np.testing.assert_array_equal(img.affine, expected_affine)

    def test_load_compressed_nifti(self, temp_dir, synthetic_4d_data):
        """Test loading compressed .nii.gz files."""
        affine = np.eye(4)
        img = nib.Nifti1Image(synthetic_4d_data, affine)
        compressed_path = temp_dir / "compressed.nii.gz"
        nib.save(img, compressed_path)
        
        loaded_img = load_nifti(compressed_path)
        assert loaded_img.ndim == 4
        np.testing.assert_array_equal(loaded_img.get_fdata(), synthetic_4d_data)

    def test_load_uncompressed_nifti(self, temp_dir, synthetic_4d_data):
        """Test loading uncompressed .nii files."""
        affine = np.eye(4)
        img = nib.Nifti1Image(synthetic_4d_data, affine)
        uncompressed_path = temp_dir / "uncompressed.nii"
        nib.save(img, uncompressed_path)
        
        loaded_img = load_nifti(uncompressed_path)
        assert loaded_img.ndim == 4
        np.testing.assert_array_equal(loaded_img.get_fdata(), synthetic_4d_data)


class TestGetImageMetadata:
    """Test image metadata extraction."""

    def test_get_4d_metadata(self, synthetic_4d_nifti):
        """Test extracting metadata from 4D image."""
        img = nib.load(synthetic_4d_nifti)
        metadata = get_image_metadata(img)
        
        assert metadata["ndim"] == 4
        assert metadata["shape"] == (10, 10, 10, 50)
        assert metadata["n_timepoints"] == 50
        assert metadata["voxel_size"] == (3.0, 3.0, 3.0)
        assert metadata["dtype"] in ["float32", "float64"]

    def test_get_3d_metadata(self, synthetic_3d_nifti):
        """Test extracting metadata from 3D image."""
        img = nib.load(synthetic_3d_nifti)
        metadata = get_image_metadata(img)
        
        assert metadata["ndim"] == 3
        assert metadata["shape"] == (10, 10, 10)
        assert metadata["n_timepoints"] is None
        assert metadata["voxel_size"] == (3.0, 3.0, 3.0)
        assert metadata["dtype"] in ["float32", "float64"]

    def test_get_metadata_with_different_voxel_sizes(self, temp_dir, synthetic_4d_data):
        """Test metadata extraction with different voxel sizes."""
        affine = np.eye(4)
        affine[0, 0] = 2.0  # 2mm x-voxel
        affine[1, 1] = 2.5  # 2.5mm y-voxel
        affine[2, 2] = 3.5  # 3.5mm z-voxel
        
        img = nib.Nifti1Image(synthetic_4d_data, affine)
        filepath = temp_dir / "different_voxels.nii.gz"
        nib.save(img, filepath)
        
        loaded_img = nib.load(filepath)
        metadata = get_image_metadata(loaded_img)
        
        assert metadata["voxel_size"] == (2.0, 2.5, 3.5)

    def test_get_metadata_includes_tr(self, temp_dir, synthetic_4d_data):
        """Test that metadata includes TR (repetition time) if available."""
        affine = np.eye(4)
        img = nib.Nifti1Image(synthetic_4d_data, affine)
        
        # Set TR in header
        img.header.set_xyzt_units(xyz="mm", t="sec")
        img.header["pixdim"][4] = 2.5  # TR = 2.5 seconds
        
        filepath = temp_dir / "with_tr.nii.gz"
        nib.save(img, filepath)
        
        loaded_img = nib.load(filepath)
        metadata = get_image_metadata(loaded_img)
        
        assert metadata["tr"] == 2.5

    def test_get_metadata_none_tr(self, synthetic_4d_nifti):
        """Test that TR is None when not set or invalid."""
        img = nib.load(synthetic_4d_nifti)
        metadata = get_image_metadata(img)
        
        # TR may be None, 0.0, or 1.0 (default value) if not properly set
        assert metadata["tr"] in [None, 0.0, 1.0]

    def test_metadata_preserves_units(self, temp_dir, synthetic_4d_data):
        """Test that metadata preserves spatial and temporal units."""
        affine = np.eye(4)
        img = nib.Nifti1Image(synthetic_4d_data, affine)
        
        # Set units explicitly
        img.header.set_xyzt_units(xyz="mm", t="sec")
        
        filepath = temp_dir / "with_units.nii.gz"
        nib.save(img, filepath)
        
        loaded_img = nib.load(filepath)
        metadata = get_image_metadata(loaded_img)
        
        assert metadata["spatial_units"] == "mm"
        assert metadata["temporal_units"] == "sec"


class TestValidate4dImage:
    """Test 4D image validation."""

    def test_validate_valid_4d_image(self, synthetic_4d_nifti):
        """Test validation of valid 4D image."""
        img = nib.load(synthetic_4d_nifti)
        # Should not raise any exception
        validate_4d_image(img)

    def test_validate_3d_image_raises_error(self, synthetic_3d_nifti):
        """Test that 3D images raise InvalidNiftiError."""
        img = nib.load(synthetic_3d_nifti)
        with pytest.raises(InvalidNiftiError, match="must be 4D"):
            validate_4d_image(img)

    def test_validate_empty_4d_raises_error(self):
        """Test that 4D image with no timepoints raises error."""
        empty_data = np.zeros((10, 10, 10, 0))
        affine = np.eye(4)
        img = nib.Nifti1Image(empty_data, affine)
        
        with pytest.raises(InvalidNiftiError, match="must have at least 1 timepoint"):
            validate_4d_image(img)

    def test_validate_single_timepoint_valid(self):
        """Test that 4D image with single timepoint is valid."""
        single_tp_data = np.random.randn(10, 10, 10, 1).astype(np.float32)
        affine = np.eye(4)
        img = nib.Nifti1Image(single_tp_data, affine)
        
        # Should not raise any exception
        validate_4d_image(img)

    def test_validate_large_4d_valid(self):
        """Test validation of large 4D image."""
        large_data = np.random.randn(64, 64, 32, 200).astype(np.float32)
        affine = np.eye(4)
        img = nib.Nifti1Image(large_data, affine)
        
        # Should not raise any exception
        validate_4d_image(img)

    def test_validate_none_image_raises_error(self):
        """Test that None image raises InvalidNiftiError."""
        with pytest.raises(InvalidNiftiError, match="Image cannot be None"):
            validate_4d_image(None)

    def test_validate_5d_image_raises_error(self):
        """Test that 5D images raise InvalidNiftiError."""
        data_5d = np.random.randn(10, 10, 10, 50, 2).astype(np.float32)
        affine = np.eye(4)
        img = nib.Nifti1Image(data_5d, affine)
        
        with pytest.raises(InvalidNiftiError, match="must be 4D"):
            validate_4d_image(img)

    def test_validate_with_nan_values(self):
        """Test validation of image containing NaN values."""
        data_with_nan = np.random.randn(10, 10, 10, 50).astype(np.float32)
        data_with_nan[5, 5, 5, :] = np.nan
        affine = np.eye(4)
        img = nib.Nifti1Image(data_with_nan, affine)
        
        # Should not raise exception - NaN values are allowed
        validate_4d_image(img)

    def test_validate_with_inf_values(self):
        """Test validation of image containing infinite values."""
        data_with_inf = np.random.randn(10, 10, 10, 50).astype(np.float32)
        data_with_inf[0, 0, 0, :] = np.inf
        data_with_inf[1, 1, 1, :] = -np.inf
        affine = np.eye(4)
        img = nib.Nifti1Image(data_with_inf, affine)
        
        # Should not raise exception - inf values are allowed
        validate_4d_image(img)