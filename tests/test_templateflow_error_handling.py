"""Test TemplateFlow error handling and available atlas listing."""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch

from parcelextract.atlases.templateflow import TemplateFlowManager


class TestTemplateFlowErrorHandling:
    """Test comprehensive error handling for TemplateFlow atlases."""

    def test_nonexistent_desc_error_message(self):
        """Test helpful error message when desc doesn't exist."""
        tf_manager = TemplateFlowManager()
        
        with pytest.raises(ValueError) as exc_info:
            tf_manager.get_atlas('Schaefer2018', 'MNI152NLin2009cAsym', desc='nonexistent_desc')
        
        error_msg = str(exc_info.value)
        # Should mention the specific atlas that failed
        assert 'Schaefer2018' in error_msg
        assert 'nonexistent_desc' in error_msg
        # Should suggest available alternatives
        assert 'available descriptions' in error_msg.lower() or 'available' in error_msg.lower()

    def test_nonexistent_atlas_error_message(self):
        """Test helpful error message when atlas doesn't exist."""
        tf_manager = TemplateFlowManager()
        
        with pytest.raises(ValueError) as exc_info:
            tf_manager.get_atlas('NonexistentAtlas2024', 'MNI152NLin2009cAsym')
        
        error_msg = str(exc_info.value)
        # Should mention the specific atlas that failed
        assert 'NonexistentAtlas2024' in error_msg
        # Should mention available atlases
        assert 'available' in error_msg.lower() or 'supported' in error_msg.lower()

    def test_list_available_descriptions_for_atlas(self):
        """Test that manager can list available descriptions for an atlas."""
        tf_manager = TemplateFlowManager()
        
        # This should work for real atlas
        descriptions = tf_manager.list_available_descriptions('Schaefer2018', 'MNI152NLin2009cAsym')
        
        # Should return a list of strings
        assert isinstance(descriptions, list)
        assert len(descriptions) > 0
        assert all(isinstance(desc, str) for desc in descriptions)
        
        # Should include known descriptions
        assert any('400Parcels' in desc for desc in descriptions)
        assert any('7Networks' in desc or '17Networks' in desc for desc in descriptions)

    def test_list_available_atlases_for_space(self):
        """Test that manager can list available atlases for a template space."""
        tf_manager = TemplateFlowManager()
        
        # This should work for real space
        atlases = tf_manager.list_available_atlases('MNI152NLin2009cAsym')
        
        # Should return a list of strings
        assert isinstance(atlases, list)
        assert len(atlases) > 0
        assert all(isinstance(atlas, str) for atlas in atlases)
        
        # Should include some known atlases
        known_atlases = ['Schaefer2018', 'AAL', 'HarvardOxford']
        found_atlases = [atlas for atlas in known_atlases if atlas in atlases]
        assert len(found_atlases) > 0  # At least one should be available

    def test_empty_templateflow_result_handling(self):
        """Test handling of empty results from TemplateFlow."""
        tf_manager = TemplateFlowManager()
        
        # Mock TemplateFlow to return empty list (simulating not found)
        with patch('templateflow.api.get', return_value=[]):
            with pytest.raises(ValueError) as exc_info:
                tf_manager.download_atlas(
                    template='MNI152NLin2009cAsym',
                    atlas='Schaefer2018', 
                    resolution=2,
                    desc='nonexistent'
                )
            
            error_msg = str(exc_info.value)
            assert 'not found' in error_msg.lower() or 'no atlas' in error_msg.lower()

    @patch('templateflow.api.get')
    def test_templateflow_api_exception_handling(self, mock_get):
        """Test handling of TemplateFlow API exceptions."""
        tf_manager = TemplateFlowManager()
        
        # Mock TemplateFlow to raise an exception
        mock_get.side_effect = Exception("TemplateFlow connection error")
        
        with pytest.raises(ValueError) as exc_info:
            tf_manager.download_atlas(
                template='MNI152NLin2009cAsym',
                atlas='Schaefer2018', 
                resolution=2
            )
        
        error_msg = str(exc_info.value)
        # Should wrap the original error helpfully
        assert 'TemplateFlow connection error' in error_msg

    def test_invalid_space_error_message(self):
        """Test helpful error when template space doesn't exist."""
        tf_manager = TemplateFlowManager()
        
        with pytest.raises(ValueError) as exc_info:
            tf_manager.get_atlas('Schaefer2018', 'InvalidSpace2024')
        
        error_msg = str(exc_info.value)
        # Should mention the invalid space
        assert 'InvalidSpace2024' in error_msg
        # Should suggest available spaces (or at least mention the issue)
        assert 'space' in error_msg.lower() or 'template' in error_msg.lower()

    def test_error_includes_query_parameters(self):
        """Test that error messages include the attempted query parameters."""
        tf_manager = TemplateFlowManager()
        
        with pytest.raises(ValueError) as exc_info:
            tf_manager.download_atlas(
                template='MNI152NLin2009cAsym',
                atlas='Schaefer2018',
                resolution=2,
                desc='invalid_desc_12345'
            )
        
        error_msg = str(exc_info.value)
        # Should include key parameters that failed
        assert 'Schaefer2018' in error_msg
        assert 'invalid_desc_12345' in error_msg
        assert 'MNI152NLin2009cAsym' in error_msg

    def test_helpful_suggestion_format(self):
        """Test that error messages provide well-formatted suggestions."""
        tf_manager = TemplateFlowManager()
        
        with pytest.raises(ValueError) as exc_info:
            tf_manager.get_atlas('Schaefer2018', 'MNI152NLin2009cAsym', desc='badformat')
        
        error_msg = str(exc_info.value)
        
        # Should be readable and helpful
        assert len(error_msg) > 50  # Not just a terse error
        # Should provide actionable information
        assert 'available' in error_msg.lower() or 'supported' in error_msg.lower()
        
        # Should format lists nicely (not just dump a repr)
        # If it includes a list, it should be reasonably formatted
        if '[' in error_msg and ']' in error_msg:
            # Should not just be a raw Python list repr
            assert error_msg.count("'") < 20 or ',' in error_msg