"""Test DiFuMo atlas support in ParcelExtract."""

import pytest
from pathlib import Path

from parcelextract.atlases.templateflow import TemplateFlowManager
from parcelextract.cli.main import resolve_atlas_path


class TestDiFuMoSupport:
    """Test support for DiFuMo atlas from TemplateFlow."""

    def test_difumo_in_atlas_configs(self):
        """Test that DiFuMo is included in supported atlas configurations."""
        tf_manager = TemplateFlowManager()
        
        # Should not raise an error for DiFuMo
        config = tf_manager.resolve_atlas_name('DiFuMo', 'MNI152NLin2009cAsym')
        
        assert config['atlas'] == 'DiFuMo'
        assert config['template'] == 'MNI152NLin2009cAsym'
        assert 'desc' in config
        assert config['desc'] == '64dimensions'  # Default description

    def test_list_difumo_descriptions(self):
        """Test listing available descriptions for DiFuMo."""
        tf_manager = TemplateFlowManager()
        
        descriptions = tf_manager.list_available_descriptions('DiFuMo', 'MNI152NLin2009cAsym')
        
        # Should return a list of available descriptions
        assert isinstance(descriptions, list)
        assert len(descriptions) > 0
        
        # Should include known DiFuMo dimensions
        expected_dims = ['64dimensions', '128dimensions', '256dimensions', '512dimensions', '1024dimensions']
        for dim in expected_dims:
            assert dim in descriptions

    def test_list_available_atlases_includes_difumo(self):
        """Test that DiFuMo is listed as an available atlas."""
        tf_manager = TemplateFlowManager()
        
        available_atlases = tf_manager.list_available_atlases('MNI152NLin2009cAsym')
        
        assert isinstance(available_atlases, list)
        assert 'DiFuMo' in available_atlases

    def test_download_difumo_atlas(self):
        """Test downloading DiFuMo atlas with default description."""
        tf_manager = TemplateFlowManager()
        
        # Should succeed with default description
        atlas_path = tf_manager.get_atlas('DiFuMo', 'MNI152NLin2009cAsym')
        
        assert isinstance(atlas_path, str)
        assert len(atlas_path) > 0
        assert 'DiFuMo' in atlas_path
        assert 'probseg' in atlas_path  # DiFuMo uses probseg not dseg
        assert Path(atlas_path).exists()

    def test_download_difumo_with_specific_desc(self):
        """Test downloading DiFuMo atlas with specific description."""
        tf_manager = TemplateFlowManager()
        
        # Should succeed with specific valid description
        atlas_path = tf_manager.get_atlas('DiFuMo', 'MNI152NLin2009cAsym', desc='128dimensions')
        
        assert isinstance(atlas_path, str)
        assert 'DiFuMo' in atlas_path
        assert '128dimensions' in atlas_path
        assert 'probseg' in atlas_path
        assert Path(atlas_path).exists()

    def test_difumo_invalid_desc_error_message(self):
        """Test helpful error message for invalid DiFuMo description."""
        tf_manager = TemplateFlowManager()
        
        with pytest.raises(ValueError) as exc_info:
            tf_manager.get_atlas('DiFuMo', 'MNI152NLin2009cAsym', desc='invalid_dimensions')
        
        error_msg = str(exc_info.value)
        
        # Should mention DiFuMo and the invalid description
        assert 'DiFuMo' in error_msg
        assert 'invalid_dimensions' in error_msg
        
        # Should list available descriptions
        assert 'available descriptions' in error_msg.lower()
        assert '64dimensions' in error_msg
        assert '128dimensions' in error_msg

    def test_cli_resolve_difumo_atlas(self):
        """Test CLI atlas resolution for DiFuMo."""
        # Should recognize DiFuMo as a TemplateFlow atlas
        atlas_path = resolve_atlas_path('DiFuMo', 'MNI152NLin2009cAsym')
        
        assert isinstance(atlas_path, str)
        assert 'DiFuMo' in atlas_path
        assert Path(atlas_path).exists()

    def test_cli_resolve_difumo_with_desc(self):
        """Test CLI atlas resolution for DiFuMo with specific description."""
        atlas_path = resolve_atlas_path('DiFuMo', 'MNI152NLin2009cAsym', desc='256dimensions')
        
        assert isinstance(atlas_path, str)
        assert 'DiFuMo' in atlas_path
        assert '256dimensions' in atlas_path
        assert Path(atlas_path).exists()

    def test_difumo_case_insensitive(self):
        """Test that DiFuMo atlas name is case insensitive."""
        tf_manager = TemplateFlowManager()
        
        # Different case variations should work
        variations = ['DiFuMo', 'difumo', 'DIFUMO', 'Difumo']
        
        for variation in variations:
            config = tf_manager.resolve_atlas_name(variation, 'MNI152NLin2009cAsym')
            assert config['atlas'] == 'DiFuMo'  # Should normalize to correct case

    def test_difumo_probseg_vs_dseg(self):
        """Test that DiFuMo uses probseg files, not dseg."""
        tf_manager = TemplateFlowManager()
        
        atlas_path = tf_manager.get_atlas('DiFuMo', 'MNI152NLin2009cAsym', desc='64dimensions')
        
        # DiFuMo should use probseg (probabilistic segmentation) files
        assert 'probseg' in atlas_path
        assert 'dseg' not in atlas_path

    def test_difumo_vs_schaefer_file_types(self):
        """Test that different atlases use appropriate file types."""
        tf_manager = TemplateFlowManager()
        
        # DiFuMo should use probseg
        difumo_path = tf_manager.get_atlas('DiFuMo', 'MNI152NLin2009cAsym')
        assert 'probseg' in difumo_path
        
        # Schaefer should use dseg
        schaefer_path = tf_manager.get_atlas('Schaefer2018', 'MNI152NLin2009cAsym')
        assert 'dseg' in schaefer_path

    def test_mixed_atlas_description_listing(self):
        """Test that description listing works for both dseg and probseg atlases."""
        tf_manager = TemplateFlowManager()
        
        # Both should return non-empty description lists
        schaefer_descs = tf_manager.list_available_descriptions('Schaefer2018', 'MNI152NLin2009cAsym')
        difumo_descs = tf_manager.list_available_descriptions('DiFuMo', 'MNI152NLin2009cAsym')
        
        assert len(schaefer_descs) > 0
        assert len(difumo_descs) > 0
        
        # Should have different description formats
        assert any('Parcels' in desc for desc in schaefer_descs)  # e.g., "400Parcels7Networks"
        assert any('dimensions' in desc for desc in difumo_descs)  # e.g., "64dimensions"