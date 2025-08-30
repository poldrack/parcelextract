"""Test atlas management functionality."""

import numpy as np
import pytest
from pathlib import Path

from parcelextract.atlases.manager import AtlasManager


class TestAtlasManager:
    """Test Atlas manager functionality."""

    def test_can_create_atlas_manager_instance(self):
        """Test that AtlasManager can be instantiated."""
        manager = AtlasManager()
        assert isinstance(manager, AtlasManager)

    def test_load_atlas_from_file(self, temp_dir):
        """Test loading atlas from file path."""
        # Create a simple test atlas file
        atlas_data = np.array([
            [[0, 0, 1], [1, 1, 2], [2, 2, 0]],
            [[0, 1, 1], [2, 2, 2], [0, 0, 1]],
            [[1, 1, 2], [0, 0, 0], [1, 1, 1]]
        ])
        
        # This test should fail initially since AtlasManager doesn't exist yet
        manager = AtlasManager()
        
        # Create test atlas file (we'll use nibabel format)
        import nibabel as nib
        atlas_img = nib.Nifti1Image(atlas_data.astype(np.int16), affine=np.eye(4))
        atlas_file = temp_dir / "test_atlas.nii.gz"
        nib.save(atlas_img, atlas_file)
        
        # Should load atlas successfully
        atlas = manager.load_atlas(atlas_file)
        
        # Should return atlas with expected properties
        assert atlas is not None
        assert hasattr(atlas, 'data')
        assert hasattr(atlas, 'labels')
        np.testing.assert_array_equal(atlas.data, atlas_data)

    def test_get_atlas_labels(self, temp_dir):
        """Test extracting unique labels from atlas."""
        # Create atlas with known labels: 0 (background), 1, 2
        atlas_data = np.array([
            [[0, 0, 1], [1, 1, 2], [2, 2, 0]],
            [[0, 1, 1], [2, 2, 2], [0, 0, 1]],
            [[1, 1, 2], [0, 0, 0], [1, 1, 1]]
        ])
        
        import nibabel as nib
        atlas_img = nib.Nifti1Image(atlas_data.astype(np.int16), affine=np.eye(4))
        atlas_file = temp_dir / "label_atlas.nii.gz"
        nib.save(atlas_img, atlas_file)
        
        manager = AtlasManager()
        atlas = manager.load_atlas(atlas_file)
        
        # Should extract non-zero labels
        expected_labels = [1, 2]  # Background (0) should be excluded
        assert atlas.labels == expected_labels

    def test_get_atlas_metadata(self, temp_dir):
        """Test getting atlas metadata."""
        atlas_data = np.ones((3, 3, 3), dtype=np.int16)
        
        import nibabel as nib
        atlas_img = nib.Nifti1Image(atlas_data, affine=np.eye(4))
        atlas_file = temp_dir / "meta_atlas.nii.gz"
        nib.save(atlas_img, atlas_file)
        
        manager = AtlasManager()
        
        # Should fail initially - get_metadata method doesn't exist
        metadata = manager.get_metadata(atlas_file)
        
        # Should return metadata dictionary
        assert isinstance(metadata, dict)
        assert 'shape' in metadata
        assert 'n_labels' in metadata
        assert metadata['shape'] == (3, 3, 3)
        assert metadata['n_labels'] == 1

    def test_load_nonexistent_atlas_raises_error(self):
        """Test that loading nonexistent atlas raises appropriate error."""
        manager = AtlasManager()
        
        with pytest.raises(FileNotFoundError):
            manager.load_atlas("nonexistent_atlas.nii.gz")

    def test_load_invalid_atlas_raises_error(self, temp_dir):
        """Test that loading invalid atlas file raises error."""
        # Create a text file instead of nifti
        invalid_file = temp_dir / "invalid.txt"
        invalid_file.write_text("not a nifti file")
        
        manager = AtlasManager()
        
        # Should fail - validate_atlas method doesn't exist yet
        with pytest.raises(Exception):  # Will be more specific once implemented
            manager.validate_atlas(invalid_file)

    def test_validate_atlas_with_valid_file(self, temp_dir):
        """Test atlas validation with valid file."""
        atlas_data = np.array([[[1, 2], [0, 1]], [[2, 0], [1, 2]]])
        
        import nibabel as nib
        atlas_img = nib.Nifti1Image(atlas_data.astype(np.int16), affine=np.eye(4))
        atlas_file = temp_dir / "valid_atlas.nii.gz"
        nib.save(atlas_img, atlas_file)
        
        manager = AtlasManager()
        
        # Should return True for valid atlas
        is_valid = manager.validate_atlas(atlas_file)
        assert is_valid is True


class TestAtlasIntegration:
    """Test atlas integration with existing extraction system."""

    def test_atlas_manager_with_parcel_extractor(self, temp_dir):
        """Test that AtlasManager works with ParcelExtractor."""
        from parcelextract.core.extractor import ParcelExtractor
        
        # Create test 4D data
        img_4d = np.random.randn(4, 4, 4, 10).astype(np.float32)
        
        # Create test atlas with 2 parcels
        atlas_data = np.array([
            [[0, 0, 1, 1], [1, 1, 2, 2], [2, 2, 0, 0], [0, 1, 1, 2]],
            [[1, 1, 0, 0], [2, 2, 1, 1], [0, 0, 2, 2], [1, 2, 0, 1]],
            [[2, 0, 1, 2], [0, 1, 2, 0], [1, 2, 0, 1], [2, 0, 1, 2]],
            [[1, 2, 1, 0], [0, 1, 0, 2], [2, 0, 2, 1], [1, 2, 1, 0]]
        ])
        
        # Save test data as Nifti files
        import nibabel as nib
        
        img_4d_nii = nib.Nifti1Image(img_4d, affine=np.eye(4))
        img_file = temp_dir / "test_img.nii.gz"
        nib.save(img_4d_nii, img_file)
        
        atlas_nii = nib.Nifti1Image(atlas_data.astype(np.int16), affine=np.eye(4))
        atlas_file = temp_dir / "test_atlas.nii.gz"
        nib.save(atlas_nii, atlas_file)
        
        # Load atlas using AtlasManager
        manager = AtlasManager()
        atlas = manager.load_atlas(atlas_file)
        
        # Should be able to create ParcelExtractor with loaded atlas
        extractor = ParcelExtractor(atlas=atlas_file, strategy='mean')
        
        # Should be able to extract signals
        timeseries = extractor.fit_transform(img_file)
        
        # Should get timeseries for 2 parcels (labels 1, 2)
        assert timeseries.shape[0] == 2  # 2 parcels
        assert timeseries.shape[1] == 10  # 10 timepoints
        assert not np.any(np.isnan(timeseries))