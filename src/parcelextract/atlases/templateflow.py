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
            },
            'difumo': {
                'template': space,
                'atlas': 'DiFuMo',
                'resolution': 2,
                'desc': '64dimensions'
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
            
        Raises
        ------
        ValueError
            If atlas or description not found in TemplateFlow
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
            
            # Check if TemplateFlow returned empty result (atlas not found)
            if not atlas_path or atlas_path == []:
                # Generate helpful error message with available options
                error_msg = self._generate_atlas_not_found_error(
                    template, atlas, desc, query_params
                )
                raise ValueError(error_msg)
            
            return str(atlas_path)
        except ValueError:
            # Re-raise ValueError (our custom error above)
            raise
        except Exception as e:
            # Wrap other exceptions with helpful context
            error_msg = self._generate_download_error_message(
                template, atlas, desc, query_params, e
            )
            raise ValueError(error_msg) from e
    
    def get_available_resolutions(self, atlas_name: str, template: str) -> list:
        """
        Get available resolutions for a given atlas and template.
        
        Parameters
        ----------
        atlas_name : str
            Atlas name (e.g., 'DiFuMo', 'Schaefer2018')
        template : str
            Template space (e.g., 'MNI152NLin2009cAsym')
            
        Returns
        -------
        list
            List of available resolution strings (e.g., ['01', '02', '03'])
        """
        try:
            # Get all available atlases for this template/atlas combination
            atlases = []
            
            # Try both dseg and probseg files
            for suffix in ['dseg', 'probseg']:
                try:
                    files = tflow.get(
                        template=template,
                        atlas=atlas_name,
                        suffix=suffix,
                        extension='.nii.gz'
                    )
                    if files and files != []:
                        atlases.extend(files if isinstance(files, list) else [files])
                except Exception:
                    continue
            
            # Extract unique resolutions
            resolutions = set()
            for atlas_path in atlases:
                path_str = str(atlas_path)
                if 'res-' in path_str:
                    res_part = path_str.split('res-')[1].split('_')[0]
                    resolutions.add(res_part)
            
            # Convert to integers for sorting, then back to zero-padded strings
            try:
                sorted_resolutions = []
                for res in sorted(resolutions):
                    # Try to convert to int and back to ensure proper format
                    res_int = int(res)
                    sorted_resolutions.append(f"{res_int:02d}")
                return sorted_resolutions
            except ValueError:
                # If conversion fails, return as-is
                return sorted(resolutions)
                
        except Exception:
            return []

    def find_best_resolution_match(self, target_resolution: int, atlas_name: str, template: str) -> str:
        """
        Find the best available atlas resolution for a target resolution.
        
        Parameters
        ----------
        target_resolution : int
            Target resolution in mm (e.g., 2 for 2mm)
        atlas_name : str
            Atlas name (e.g., 'DiFuMo', 'Schaefer2018')
        template : str
            Template space (e.g., 'MNI152NLin2009cAsym')
            
        Returns
        -------
        str
            Best matching resolution string (e.g., '02')
            
        Raises
        ------
        ValueError
            If no resolutions are available for the atlas
        """
        available_resolutions = self.get_available_resolutions(atlas_name, template)
        
        if not available_resolutions:
            raise ValueError(
                f"No resolutions found for atlas '{atlas_name}' in template '{template}'"
            )
        
        # Convert to integers for comparison
        available_ints = []
        for res_str in available_resolutions:
            try:
                available_ints.append(int(res_str))
            except ValueError:
                continue
        
        if not available_ints:
            raise ValueError(
                f"Could not parse resolution values for atlas '{atlas_name}': {available_resolutions}"
            )
        
        # Find closest match
        target_int = int(target_resolution)
        closest_res = min(available_ints, key=lambda x: abs(x - target_int))
        
        # Convert back to zero-padded string
        return f"{closest_res:02d}"

    def get_atlas(self, atlas_name: str, space: str, input_img=None, **kwargs) -> str:
        """
        Get atlas file, downloading if necessary.
        
        Parameters
        ----------
        atlas_name : str
            Atlas name (e.g., 'Schaefer2018')
        space : str
            Template space (e.g., 'MNI152NLin2009cAsym')
        input_img : nibabel.Nifti1Image, optional
            Input image to detect resolution from. If provided, will automatically
            select atlas resolution to match input image resolution.
        **kwargs
            Override default atlas parameters
            
        Returns
        -------
        str
            Path to atlas file
        """
        # Resolve atlas configuration
        atlas_config = self.resolve_atlas_name(atlas_name, space)
        
        # Auto-detect resolution from input image if provided
        if input_img is not None and 'resolution' not in kwargs:
            try:
                # Import here to avoid circular imports
                import sys
                if 'parcelextract.core.validators' in sys.modules:
                    detect_image_resolution = sys.modules['parcelextract.core.validators'].detect_image_resolution
                else:
                    from ..core.validators import detect_image_resolution
                
                detected_res = detect_image_resolution(input_img)
                
                # Find best matching atlas resolution
                best_res_str = self.find_best_resolution_match(detected_res, atlas_name, space)
                best_res_int = int(best_res_str)
                
                print(f"Auto-detected input resolution: {detected_res}mm")
                print(f"Using atlas resolution: {best_res_int}mm (res-{best_res_str})")
                
                atlas_config['resolution'] = best_res_int
                
            except Exception as e:
                print(f"Warning: Could not auto-detect resolution: {e}")
                print(f"Using default resolution: {atlas_config['resolution']}mm")
        
        # Allow kwargs to override defaults (including auto-detected resolution)
        atlas_config.update(kwargs)
        
        # Download atlas
        return self.download_atlas(**atlas_config)
    
    def list_available_descriptions(self, atlas_name: str, template: str) -> list:
        """
        List available descriptions for a given atlas and template.
        
        Parameters
        ----------
        atlas_name : str
            Atlas name (e.g., 'Schaefer2018')
        template : str
            Template space (e.g., 'MNI152NLin2009cAsym')
            
        Returns
        -------
        list
            List of available description strings
        """
        try:
            # Get all available atlases for this template/atlas combination
            # Try both dseg (discrete segmentation) and probseg (probabilistic segmentation)
            atlases = []
            
            # Try dseg files first (most common)
            try:
                dseg_files = tflow.get(
                    template=template,
                    atlas=atlas_name,
                    suffix='dseg',
                    extension='.nii.gz'
                )
                if dseg_files and dseg_files != []:
                    atlases.extend(dseg_files if isinstance(dseg_files, list) else [dseg_files])
            except Exception:
                pass
            
            # Try probseg files (for atlases like DiFuMo)
            try:
                probseg_files = tflow.get(
                    template=template,
                    atlas=atlas_name,
                    suffix='probseg',
                    extension='.nii.gz'
                )
                if probseg_files and probseg_files != []:
                    atlases.extend(probseg_files if isinstance(probseg_files, list) else [probseg_files])
            except Exception:
                pass
            
            # Extract unique descriptions
            descriptions = set()
            for atlas_path in atlases:
                path_str = str(atlas_path)
                if 'desc-' in path_str:
                    # Extract the desc value
                    desc_part = path_str.split('desc-')[1].split('_')[0]
                    descriptions.add(desc_part)
            
            return sorted(descriptions)
        except Exception:
            # If we can't query, return empty list
            return []
    
    def list_available_atlases(self, template: str) -> list:
        """
        List available atlases for a given template space.
        
        Parameters
        ----------
        template : str
            Template space (e.g., 'MNI152NLin2009cAsym')
            
        Returns
        -------
        list
            List of available atlas names
        """
        try:
            # Get all available atlases for this template
            # We'll query known atlas types and see what's available
            known_atlases = ['Schaefer2018', 'AAL', 'HarvardOxford', 'Destrieux', 'DesikanKilliany', 'DiFuMo']
            available_atlases = []
            
            for atlas_name in known_atlases:
                atlas_found = False
                
                # Try dseg files first (most atlases)
                try:
                    result = tflow.get(
                        template=template,
                        atlas=atlas_name,
                        suffix='dseg',
                        extension='.nii.gz'
                    )
                    if result and result != []:
                        atlas_found = True
                except Exception:
                    pass
                
                # Try probseg files (for atlases like DiFuMo)
                if not atlas_found:
                    try:
                        result = tflow.get(
                            template=template,
                            atlas=atlas_name,
                            suffix='probseg',
                            extension='.nii.gz'
                        )
                        if result and result != []:
                            atlas_found = True
                    except Exception:
                        pass
                
                if atlas_found:
                    available_atlases.append(atlas_name)
            
            return sorted(available_atlases)
        except Exception:
            # If we can't query, return known atlases as fallback
            return ['Schaefer2018', 'AAL', 'HarvardOxford', 'DiFuMo']
    
    def _generate_atlas_not_found_error(self, template: str, atlas: str, 
                                      desc: Optional[str], query_params: dict) -> str:
        """Generate helpful error message when atlas is not found."""
        # Build base error message
        if desc is not None:
            error_msg = (
                f"Atlas '{atlas}' with description '{desc}' not found in TemplateFlow "
                f"for template space '{template}'."
            )
        else:
            error_msg = (
                f"Atlas '{atlas}' not found in TemplateFlow "
                f"for template space '{template}'."
            )
        
        # Add query parameters for debugging
        error_msg += f"\n\nAttempted query parameters: {query_params}"
        
        # Try to provide helpful suggestions
        if desc is not None:
            # If desc was specified, list available descriptions for this atlas
            try:
                available_descs = self.list_available_descriptions(atlas, template)
                if available_descs:
                    desc_list = ', '.join(f"'{d}'" for d in available_descs[:10])
                    if len(available_descs) > 10:
                        desc_list += f" (and {len(available_descs) - 10} more)"
                    error_msg += f"\n\nAvailable descriptions for {atlas}: {desc_list}"
                else:
                    error_msg += f"\n\nNo descriptions found for {atlas} in {template}."
            except Exception:
                pass
        else:
            # If atlas itself wasn't found, list available atlases
            try:
                available_atlases = self.list_available_atlases(template)
                if available_atlases:
                    atlas_list = ', '.join(f"'{a}'" for a in available_atlases)
                    error_msg += f"\n\nAvailable atlases for {template}: {atlas_list}"
                else:
                    error_msg += f"\n\nNo atlases found for template {template}."
            except Exception:
                pass
        
        return error_msg
    
    def _generate_download_error_message(self, template: str, atlas: str, 
                                       desc: Optional[str], query_params: dict, 
                                       original_error: Exception) -> str:
        """Generate helpful error message when download fails."""
        if desc is not None:
            error_msg = (
                f"Failed to download atlas '{atlas}' with description '{desc}' "
                f"from TemplateFlow for template '{template}': {original_error}"
            )
        else:
            error_msg = (
                f"Failed to download atlas '{atlas}' from TemplateFlow "
                f"for template '{template}': {original_error}"
            )
        
        error_msg += f"\n\nQuery parameters: {query_params}"
        error_msg += "\n\nThis could be due to:"
        error_msg += "\n- Network connectivity issues"
        error_msg += "\n- TemplateFlow server problems"
        error_msg += "\n- Invalid atlas/template combination"
        
        return error_msg