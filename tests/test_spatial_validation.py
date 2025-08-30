"""Test spatial dimension validation for ParcelExtract."""

import pytest
import numpy as np
import nibabel as nib
import tempfile
from pathlib import Path

from parcelextract.core.validators import validate_spatial_compatibility, ValidationError
from parcelextract.core.extractor import ParcelExtractor


class TestSpatialValidation:
    """Test spatial dimension validation functionality."""

    def test_compatible_dimensions_pass_validation(self):
        """Test that images with matching dimensions pass validation."""
        # Create compatible images
        input_data = np.random.randn(10, 10, 10, 50).astype(np.float32)
        atlas_data = np.random.randint(0, 5, size=(10, 10, 10)).astype(np.int16)
        
        affine = np.eye(4)
        input_img = nib.Nifti1Image(input_data, affine)
        atlas_img = nib.Nifti1Image(atlas_data, affine)
        
        # Should not raise any exception
        validate_spatial_compatibility(input_img, atlas_img)

    def test_incompatible_dimensions_fail_validation(self):
        """Test that images with different dimensions fail validation."""
        # Create incompatible images
        input_data = np.random.randn(10, 10, 10, 50).astype(np.float32)  # 10x10x10 spatial
        atlas_data = np.random.randint(0, 5, size=(12, 10, 10)).astype(np.int16)  # 12x10x10 spatial
        
        affine = np.eye(4)
        input_img = nib.Nifti1Image(input_data, affine)
        atlas_img = nib.Nifti1Image(atlas_data, affine)
        
        with pytest.raises(ValidationError) as exc_info:
            validate_spatial_compatibility(input_img, atlas_img)
        
        error_msg = str(exc_info.value)
        assert "incompatible spatial dimensions" in error_msg
        assert "(10, 10, 10)" in error_msg  # Input dimensions
        assert "(12, 10, 10)" in error_msg  # Atlas dimensions
        assert "coordinate space" in error_msg
        assert "resample" in error_msg.lower()

    def test_different_affine_matrices_warning(self):
        """Test that significantly different affine matrices produce a warning."""
        # Create images with same dimensions but different affines
        input_data = np.random.randn(10, 10, 10, 50).astype(np.float32)
        atlas_data = np.random.randint(0, 5, size=(10, 10, 10)).astype(np.int16)
        
        input_affine = np.eye(4)
        atlas_affine = np.eye(4)
        atlas_affine[0, 3] = 5.0  # Significant translation difference
        
        input_img = nib.Nifti1Image(input_data, input_affine)
        atlas_img = nib.Nifti1Image(atlas_data, atlas_affine)
        
        with pytest.warns(UserWarning) as record:
            validate_spatial_compatibility(input_img, atlas_img)
        
        # Check that warning was issued
        assert len(record) == 1
        warning_msg = str(record[0].message)
        assert "affine matrices" in warning_msg
        assert "coordinate space" in warning_msg

    def test_similar_affine_matrices_no_warning(self):
        """Test that very similar affine matrices don't produce warnings."""
        # Create images with same dimensions and very similar affines
        input_data = np.random.randn(10, 10, 10, 50).astype(np.float32)
        atlas_data = np.random.randint(0, 5, size=(10, 10, 10)).astype(np.int16)
        
        input_affine = np.eye(4)
        atlas_affine = np.eye(4)
        atlas_affine[0, 3] = 0.001  # Very small difference (floating point precision)
        
        input_img = nib.Nifti1Image(input_data, input_affine)
        atlas_img = nib.Nifti1Image(atlas_data, atlas_affine)
        
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_spatial_compatibility(input_img, atlas_img)
            
            # No warnings should be issued
            assert len(w) == 0

    def test_extractor_validates_spatial_compatibility(self, temp_dir):
        """Test that ParcelExtractor validates spatial compatibility."""
        # Create incompatible image and atlas
        input_data = np.random.randn(8, 8, 8, 20).astype(np.float32)
        atlas_data = np.random.randint(0, 3, size=(10, 8, 8)).astype(np.int16)  # Different first dimension
        
        affine = np.eye(4)
        
        # Save files
        input_file = temp_dir / "input_image.nii.gz"
        atlas_file = temp_dir / "atlas.nii.gz"
        
        nib.save(nib.Nifti1Image(input_data, affine), input_file)
        nib.save(nib.Nifti1Image(atlas_data, affine), atlas_file)
        
        # Create extractor
        extractor = ParcelExtractor(atlas=str(atlas_file), strategy='mean')
        
        # Should raise ValidationError during fit_transform
        with pytest.raises(ValidationError) as exc_info:
            extractor.fit_transform(str(input_file))
        
        error_msg = str(exc_info.value)
        assert "incompatible spatial dimensions" in error_msg
        assert "(8, 8, 8)" in error_msg  # Input dimensions
        assert "(10, 8, 8)" in error_msg  # Atlas dimensions

    def test_helpful_error_message_content(self):
        """Test that error message provides helpful guidance."""
        # Create incompatible images with realistic dimensions
        input_data = np.random.randn(97, 115, 97, 200).astype(np.float32)  # User's dimensions
        atlas_data = np.random.randint(0, 5, size=(104, 123, 104)).astype(np.int16)  # DiFuMo dimensions
        
        affine = np.eye(4)
        input_img = nib.Nifti1Image(input_data, affine)
        atlas_img = nib.Nifti1Image(atlas_data, affine)
        
        with pytest.raises(ValidationError) as exc_info:
            validate_spatial_compatibility(input_img, atlas_img)
        
        error_msg = str(exc_info.value)
        
        # Check that error message contains helpful guidance
        assert "Input: (97, 115, 97)" in error_msg
        assert "Atlas: (104, 123, 104)" in error_msg
        assert "same coordinate space" in error_msg
        assert "MNI152NLin2009cAsym" in error_msg
        assert "resample" in error_msg.lower()
        assert "nilearn.image.resample_to_img" in error_msg
        assert "flirt/applywarp" in error_msg

    def test_multiple_dimension_mismatches(self):
        """Test error message with multiple dimension mismatches."""
        # Create images where all three spatial dimensions differ
        input_data = np.random.randn(50, 60, 50, 100).astype(np.float32)
        atlas_data = np.random.randint(0, 5, size=(104, 123, 104)).astype(np.int16)
        
        affine = np.eye(4)
        input_img = nib.Nifti1Image(input_data, affine)
        atlas_img = nib.Nifti1Image(atlas_data, affine)
        
        with pytest.raises(ValidationError) as exc_info:
            validate_spatial_compatibility(input_img, atlas_img)
        
        error_msg = str(exc_info.value)
        assert "Input: (50, 60, 50)" in error_msg
        assert "Atlas: (104, 123, 104)" in error_msg

    def test_realistic_scenario_dimensions(self):
        """Test with realistic neuroimaging dimensions."""
        # Common scenarios in neuroimaging
        test_cases = [
            # (input_shape, atlas_shape, should_pass)
            ((91, 109, 91), (91, 109, 91), True),  # Standard MNI 2mm
            ((182, 218, 182), (182, 218, 182), True),  # Standard MNI 1mm
            ((104, 123, 104), (104, 123, 104), True),  # DiFuMo dimensions
            ((91, 109, 91), (104, 123, 104), False),  # 2mm vs DiFuMo
            ((64, 64, 64), (91, 109, 91), False),  # Custom vs standard
        ]
        
        for input_shape, atlas_shape, should_pass in test_cases:
            # Create test data
            input_data = np.random.randn(*input_shape, 10).astype(np.float32)
            atlas_data = np.random.randint(0, 5, size=atlas_shape).astype(np.int16)
            
            affine = np.eye(4)
            input_img = nib.Nifti1Image(input_data, affine)
            atlas_img = nib.Nifti1Image(atlas_data, affine)
            
            if should_pass:
                # Should not raise exception
                validate_spatial_compatibility(input_img, atlas_img)
            else:
                # Should raise ValidationError
                with pytest.raises(ValidationError):
                    validate_spatial_compatibility(input_img, atlas_img)

    def test_integration_with_real_atlas_workflow(self, temp_dir):
        """Test integration with realistic atlas workflow."""
        # Create compatible input and atlas that should work
        input_data = np.random.randn(10, 12, 10, 30).astype(np.float32)
        atlas_data = np.zeros((10, 12, 10), dtype=np.int16)
        
        # Create simple 2-parcel atlas
        atlas_data[2:5, 3:6, 3:6] = 1  # Parcel 1
        atlas_data[6:9, 6:9, 3:6] = 2  # Parcel 2
        
        affine = np.eye(4)
        
        # Save files
        input_file = temp_dir / "compatible_input.nii.gz"
        atlas_file = temp_dir / "compatible_atlas.nii.gz"
        
        nib.save(nib.Nifti1Image(input_data, affine), input_file)
        nib.save(nib.Nifti1Image(atlas_data, affine), atlas_file)
        
        # This should work without spatial errors
        extractor = ParcelExtractor(atlas=str(atlas_file), strategy='mean')
        timeseries = extractor.fit_transform(str(input_file))
        
        # Should extract timeseries for 2 parcels
        assert timeseries.shape == (2, 30)
        assert not np.any(np.isnan(timeseries))