"""TemplateFlow integration for automatic atlas downloading."""

from pathlib import Path
from typing import Dict, Any, Optional
import warnings

try:
    import templateflow.api as tflow
    TEMPLATEFLOW_AVAILABLE = True
except ImportError:
    TEMPLATEFLOW_AVAILABLE = False
    tflow = None


class TemplateFlowManager:
    """Manager for TemplateFlow atlas downloading and caching."""
    
    def __init__(self):
        """Initialize TemplateFlow manager."""
        if not TEMPLATEFLOW_AVAILABLE:
            raise ImportError(
                "TemplateFlow is not available. Please install it with: "
                "uv add templateflow"
            )
    
    def resolve_atlas_name(self, atlas_name: str, space: str) -> Dict[str, Any]:
        """
        Resolve atlas name to TemplateFlow parameters.
        
        Parameters
        ----------
        atlas_name : str
            Atlas name (e.g., 'Schaefer2018', 'AAL')
        space : str
            Template space (e.g., 'MNI152NLin2009cAsym')
            
        Returns
        -------
        dict
            Dictionary with TemplateFlow query parameters
        """
        atlas_configs = {
            'schaefer2018': {
                'template': space,
                'atlas': 'Schaefer2018',
                'resolution': 2,
                'desc': '400Parcels17Networks'
            },
            'aal': {
                'template': space,
                'atlas': 'AAL',
                'resolution': 2,
                'desc': 'SPM12'
            },
            'harvardoxford': {
                'template': space, 
                'atlas': 'HarvardOxford',
                'resolution': 2,
                'desc': 'cort-maxprob-thr25'
            }
        }
        
        atlas_key = atlas_name.lower()
        if atlas_key not in atlas_configs:
            raise ValueError(
                f"Unsupported atlas: {atlas_name}. "
                f"Supported atlases: {list(atlas_configs.keys())}"
            )
        
        return atlas_configs[atlas_key]
    
    def download_atlas(self, template: str, atlas: str, resolution: int, 
                      desc: Optional[str] = None, **kwargs) -> str:
        """
        Download atlas from TemplateFlow.
        
        Parameters
        ----------
        template : str
            Template space
        atlas : str
            Atlas name
        resolution : int
            Resolution in mm
        desc : str, optional
            Atlas description/variant
        **kwargs
            Additional TemplateFlow query parameters
            
        Returns
        -------
        str
            Path to downloaded atlas file
        """
        query_params = {
            'template': template,
            'atlas': atlas,
            'resolution': resolution,
            'extension': '.nii.gz',
            **kwargs
        }
        
        if desc is not None:
            query_params['desc'] = desc
        
        try:
            atlas_path = tflow.get(**query_params)
            return str(atlas_path)
        except Exception as e:
            raise RuntimeError(
                f"Failed to download atlas {atlas} from TemplateFlow: {e}"
            ) from e
    
    def get_atlas(self, atlas_name: str, space: str, **kwargs) -> str:
        """
        Get atlas file, downloading if necessary.
        
        Parameters
        ----------
        atlas_name : str
            Atlas name (e.g., 'Schaefer2018')
        space : str
            Template space (e.g., 'MNI152NLin2009cAsym')
        **kwargs
            Override default atlas parameters
            
        Returns
        -------
        str
            Path to atlas file
        """
        # Resolve atlas configuration
        atlas_config = self.resolve_atlas_name(atlas_name, space)
        
        # Allow kwargs to override defaults
        atlas_config.update(kwargs)
        
        # Download atlas
        return self.download_atlas(**atlas_config)