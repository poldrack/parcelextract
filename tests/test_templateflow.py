"""Test TemplateFlow integration functionality."""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from parcelextract.atlases.templateflow import TemplateFlowManager


class TestTemplateFlowManager:
    """Test TemplateFlow manager functionality."""

    def test_can_create_templateflow_manager(self):
        """Test that TemplateFlowManager can be instantiated."""
        manager = TemplateFlowManager()
        assert isinstance(manager, TemplateFlowManager)

    def test_resolve_atlas_name_schaefer2018(self):
        """Test resolving Schaefer2018 atlas name."""
        manager = TemplateFlowManager()
        
        # Should resolve Schaefer2018 to proper TemplateFlow query
        atlas_info = manager.resolve_atlas_name("Schaefer2018", space="MNI152NLin2009cAsym")
        
        assert atlas_info['template'] == 'MNI152NLin2009cAsym'
        assert atlas_info['atlas'] == 'Schaefer2018'
        assert 'resolution' in atlas_info
        assert 'desc' in atlas_info

    @patch('templateflow.api.get')
    def test_download_atlas_calls_templateflow(self, mock_get, temp_dir):
        """Test that download_atlas calls TemplateFlow API."""
        # Mock TemplateFlow API response
        mock_atlas_file = temp_dir / "schaefer_atlas.nii.gz"
        mock_atlas_file.touch()
        mock_get.return_value = str(mock_atlas_file)
        
        manager = TemplateFlowManager()
        
        # Should call TemplateFlow API and return path
        result = manager.download_atlas(
            template='MNI152NLin2009cAsym',
            atlas='Schaefer2018',
            resolution=2,
            desc='400Parcels17Networks'
        )
        
        # Should call templateflow.api.get with correct parameters
        mock_get.assert_called_once()
        call_kwargs = mock_get.call_args[1]
        assert call_kwargs['template'] == 'MNI152NLin2009cAsym'
        assert call_kwargs['atlas'] == 'Schaefer2018'
        assert call_kwargs['resolution'] == 2
        assert call_kwargs['desc'] == '400Parcels17Networks'
        
        # Should return the atlas file path
        assert result == str(mock_atlas_file)

    def test_get_atlas_with_defaults(self):
        """Test getting atlas with default parameters."""
        manager = TemplateFlowManager()
        
        with patch.object(manager, 'download_atlas') as mock_download:
            mock_download.return_value = "/path/to/atlas.nii.gz"
            
            # Should use reasonable defaults
            result = manager.get_atlas("Schaefer2018", space="MNI152NLin2009cAsym")
            
            mock_download.assert_called_once()
            call_kwargs = mock_download.call_args[1]
            assert call_kwargs['template'] == 'MNI152NLin2009cAsym'
            assert call_kwargs['atlas'] == 'Schaefer2018'
            assert 'resolution' in call_kwargs
            assert 'desc' in call_kwargs
            
            assert result == "/path/to/atlas.nii.gz"