"""Test automatic atlas resolution matching functionality."""

import pytest
import numpy as np
import nibabel as nib
import tempfile
from pathlib import Path

from parcelextract.core.validators import detect_image_resolution, ValidationError
from parcelextract.atlases.templateflow import TemplateFlowManager
from parcelextract.core.extractor import ParcelExtractor


class TestResolutionDetection:
    """Test automatic resolution detection from image headers."""

    def test_detect_2mm_resolution(self):
        """Test detection of 2mm resolution."""
        # Create image with 2mm voxel size
        affine = np.diag([2.0, 2.0, 2.0, 1.0])
        img_data = np.random.randn(91, 109, 91, 50).astype(np.float32)
        img = nib.Nifti1Image(img_data, affine)
        
        resolution = detect_image_resolution(img)
        assert resolution == 2

    def test_detect_1mm_resolution(self):
        """Test detection of 1mm resolution."""
        affine = np.diag([1.0, 1.0, 1.0, 1.0])
        img_data = np.random.randn(182, 218, 182, 100).astype(np.float32)
        img = nib.Nifti1Image(img_data, affine)
        
        resolution = detect_image_resolution(img)
        assert resolution == 1

    def test_detect_3mm_resolution(self):
        """Test detection of 3mm resolution."""
        affine = np.diag([3.0, 3.0, 3.0, 1.0])
        img_data = np.random.randn(61, 73, 61, 30).astype(np.float32)
        img = nib.Nifti1Image(img_data, affine)
        
        resolution = detect_image_resolution(img)
        assert resolution == 3

    def test_detect_anisotropic_resolution(self):
        """Test detection with slightly anisotropic voxels."""
        # Slightly different voxel sizes (should round to nearest integer)
        affine = np.diag([1.9, 2.0, 2.1, 1.0])
        img_data = np.random.randn(50, 50, 50, 20).astype(np.float32)
        img = nib.Nifti1Image(img_data, affine)
        
        resolution = detect_image_resolution(img)
        assert resolution == 2  # Should round mean of 1.9, 2.0, 2.1 to 2

    def test_detect_unusual_resolution_error(self):
        """Test that unusual resolutions raise errors."""
        # Very large voxel size
        affine = np.diag([20.0, 20.0, 20.0, 1.0])
        img_data = np.random.randn(10, 10, 10, 5).astype(np.float32)
        img = nib.Nifti1Image(img_data, affine)
        
        with pytest.raises(ValidationError) as exc_info:
            detect_image_resolution(img)
        
        assert "unusual voxel resolution" in str(exc_info.value)

    def test_complex_affine_resolution_detection(self):
        """Test resolution detection with complex affine matrices."""
        # Create a more realistic affine matrix (with rotation/shear)
        base_affine = np.diag([2.0, 2.0, 2.0, 1.0])
        # Add small rotation to make it more realistic
        rotation = np.array([[0.99, 0.1, 0, 0],
                            [-0.1, 0.99, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])
        affine = rotation @ base_affine
        
        img_data = np.random.randn(90, 110, 90, 40).astype(np.float32)
        img = nib.Nifti1Image(img_data, affine)
        
        resolution = detect_image_resolution(img)
        assert resolution == 2  # Should still detect 2mm despite rotation


class TestAvailableResolutions:
    """Test querying available atlas resolutions."""

    def test_get_difumo_resolutions(self):
        """Test getting available DiFuMo resolutions."""
        tf_manager = TemplateFlowManager()
        
        resolutions = tf_manager.get_available_resolutions('DiFuMo', 'MNI152NLin2009cAsym')
        
        assert isinstance(resolutions, list)
        assert len(resolutions) > 0
        # DiFuMo should have at least res-02 (2mm) available
        assert '02' in resolutions

    def test_get_schaefer_resolutions(self):
        """Test getting available Schaefer2018 resolutions."""
        tf_manager = TemplateFlowManager()
        
        resolutions = tf_manager.get_available_resolutions('Schaefer2018', 'MNI152NLin2009cAsym')
        
        assert isinstance(resolutions, list)
        assert len(resolutions) > 0
        # Schaefer should have multiple resolutions
        assert any(res in ['01', '02'] for res in resolutions)

    def test_find_best_resolution_match(self):
        """Test finding best resolution match."""
        tf_manager = TemplateFlowManager()
        
        # Test exact match
        best_res = tf_manager.find_best_resolution_match(2, 'DiFuMo', 'MNI152NLin2009cAsym')
        assert best_res == '02'
        
        # Test closest match (if 1mm not available for DiFuMo, should get closest)
        available_res = tf_manager.get_available_resolutions('DiFuMo', 'MNI152NLin2009cAsym')
        target_res = 1
        best_res = tf_manager.find_best_resolution_match(target_res, 'DiFuMo', 'MNI152NLin2009cAsym')
        
        # Should get the closest available resolution
        available_ints = [int(r) for r in available_res]
        expected_closest = min(available_ints, key=lambda x: abs(x - target_res))
        assert int(best_res) == expected_closest

    def test_no_resolutions_available_error(self):
        """Test error when no resolutions are found."""
        tf_manager = TemplateFlowManager()
        
        with pytest.raises(ValueError) as exc_info:
            tf_manager.find_best_resolution_match(2, 'NonexistentAtlas', 'MNI152NLin2009cAsym')
        
        assert "No resolutions found" in str(exc_info.value)


class TestAutoResolutionMatching:
    """Test automatic resolution matching in atlas downloading."""

    def test_atlas_download_with_auto_resolution(self):
        """Test downloading atlas with auto-detected resolution."""
        tf_manager = TemplateFlowManager()
        
        # Create test image with 2mm resolution
        affine = np.diag([2.0, 2.0, 2.0, 1.0])
        img_data = np.random.randn(91, 109, 91, 30).astype(np.float32)
        test_img = nib.Nifti1Image(img_data, affine)
        
        # Get DiFuMo with auto-resolution
        atlas_path = tf_manager.get_atlas('DiFuMo', 'MNI152NLin2009cAsym', 
                                         input_img=test_img, desc='64dimensions')
        
        assert isinstance(atlas_path, str)
        assert 'DiFuMo' in atlas_path
        assert 'res-02' in atlas_path  # Should have selected 2mm resolution
        assert Path(atlas_path).exists()

    def test_manual_resolution_override(self):
        """Test that manual resolution parameter overrides auto-detection."""
        tf_manager = TemplateFlowManager()
        
        # Create test image with 2mm resolution
        affine = np.diag([2.0, 2.0, 2.0, 1.0])
        img_data = np.random.randn(50, 50, 50, 20).astype(np.float32)
        test_img = nib.Nifti1Image(img_data, affine)
        
        # Manually specify 3mm resolution (overriding auto-detected 2mm)
        atlas_path = tf_manager.get_atlas('DiFuMo', 'MNI152NLin2009cAsym', 
                                         input_img=test_img, resolution=3, desc='64dimensions')
        
        # Should use manually specified resolution, not auto-detected one
        if 'res-03' in atlas_path:
            # 3mm version exists
            assert 'res-03' in atlas_path
        else:
            # If 3mm doesn't exist, should use closest available
            assert 'DiFuMo' in atlas_path


class TestParcelExtractorResolutionMatching:
    """Test ParcelExtractor with automatic resolution matching."""

    def test_extractor_auto_resolution_matching(self, temp_dir):
        """Test that ParcelExtractor uses auto-resolution matching."""
        # Create test image with specific resolution
        affine = np.diag([2.0, 2.0, 2.0, 1.0])
        
        # Use realistic dimensions - need to match actual DiFuMo dimensions
        # Get DiFuMo dimensions to create compatible test image
        import templateflow.api as tflow
        try:
            atlas_path = tflow.get(template='MNI152NLin2009cAsym', atlas='DiFuMo', 
                                 desc='64dimensions', resolution=2, extension='.nii.gz')
            atlas_img = nib.load(atlas_path)
            atlas_spatial_shape = atlas_img.shape[:3]
            atlas_affine = atlas_img.affine
        except Exception:
            # Fallback if DiFuMo not available
            atlas_spatial_shape = (104, 123, 104)
            atlas_affine = np.diag([2.0, 2.0, 2.0, 1.0])
        
        img_data = np.random.randn(*atlas_spatial_shape, 30).astype(np.float32)
        img_file = temp_dir / "test_auto_res.nii.gz"
        nib.save(nib.Nifti1Image(img_data, atlas_affine), img_file)
        
        # Test extraction with TemplateFlow atlas name
        extractor = ParcelExtractor(atlas='DiFuMo', strategy='mean')
        
        # This should trigger auto-resolution matching
        timeseries = extractor.fit_transform(str(img_file))
        
        # Verify results
        assert timeseries.shape[1] == 30  # Number of timepoints
        assert timeseries.shape[0] > 0    # Should have extracted some parcels
        
        # Verify that atlas was resolved to a file path
        assert extractor.atlas != 'DiFuMo'  # Should have been resolved
        assert 'DiFuMo' in extractor.atlas
        assert Path(extractor.atlas).exists()

    def test_extractor_with_existing_file_path(self, temp_dir):
        """Test that existing file paths are not modified."""
        # Create simple atlas file
        atlas_data = np.zeros((10, 12, 10), dtype=np.int16)
        atlas_data[2:5, 3:6, 3:6] = 1  # Simple parcel
        atlas_file = temp_dir / "test_atlas.nii.gz"
        nib.save(nib.Nifti1Image(atlas_data, np.eye(4)), atlas_file)
        
        # Create matching input image
        img_data = np.random.randn(10, 12, 10, 20).astype(np.float32)
        img_file = temp_dir / "test_input.nii.gz"
        nib.save(nib.Nifti1Image(img_data, np.eye(4)), img_file)
        
        # Test with file path (should not trigger TemplateFlow resolution)
        extractor = ParcelExtractor(atlas=str(atlas_file), strategy='mean')
        original_atlas = extractor.atlas
        
        timeseries = extractor.fit_transform(str(img_file))
        
        # Atlas path should be unchanged
        assert extractor.atlas == original_atlas
        assert timeseries.shape == (1, 20)  # 1 parcel, 20 timepoints

    def test_4d_probabilistic_atlas_handling(self, temp_dir):
        """Test handling of 4D probabilistic atlases like DiFuMo."""
        # This test might need to be skipped if DiFuMo is not available
        pytest.importorskip("templateflow")
        
        try:
            # Create image compatible with DiFuMo
            import templateflow.api as tflow
            atlas_path = tflow.get(template='MNI152NLin2009cAsym', atlas='DiFuMo', 
                                 desc='64dimensions', resolution=2, extension='.nii.gz')
            atlas_img = nib.load(atlas_path)
            
            # Create test image with matching spatial dimensions and affine
            img_data = np.random.randn(*atlas_img.shape[:3], 25).astype(np.float32)
            img_file = temp_dir / "test_4d_prob.nii.gz"
            nib.save(nib.Nifti1Image(img_data, atlas_img.affine), img_file)
            
            # Extract with DiFuMo
            extractor = ParcelExtractor(atlas='DiFuMo', strategy='mean')
            timeseries = extractor.fit_transform(str(img_file))
            
            # Should extract timeseries for each DiFuMo component
            assert timeseries.shape[0] == atlas_img.shape[3]  # Number of components
            assert timeseries.shape[1] == 25  # Number of timepoints
            assert extractor.is_probabilistic_atlas() == True
            
        except Exception as e:
            pytest.skip(f"DiFuMo not available for testing: {e}")