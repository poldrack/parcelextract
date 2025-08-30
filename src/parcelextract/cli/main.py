"""Command-line interface for ParcelExtract."""

import argparse
import sys
from pathlib import Path
from typing import Optional, List

from parcelextract.core.extractor import ParcelExtractor
from parcelextract.io.writers import write_timeseries_tsv, write_json_sidecar
from parcelextract.atlases.templateflow import TemplateFlowManager


def generate_output_filename(input_path: str, atlas_name: str, desc: str = None) -> str:
    """
    Generate BIDS-compliant output filename with atlas and desc information.
    
    Parameters
    ----------
    input_path : str
        Path to input file
    atlas_name : str
        Atlas name (e.g., 'Schaefer2018', 'AAL')
    desc : str, optional
        Atlas description/variant (e.g., '800Parcels7Networks')
        
    Returns
    -------
    str
        Base filename (without extension) for output files
    """
    input_stem = Path(input_path).stem.replace('.nii', '')
    
    # Remove common BIDS datatype suffixes to get the base filename
    bids_suffixes = ['_bold', '_T1w', '_T2w', '_dwi', '_epi', '_fieldmap', '_sbref']
    for suffix in bids_suffixes:
        if input_stem.endswith(suffix):
            input_stem = input_stem[:-len(suffix)]
            break
    
    # Clean atlas name for BIDS compliance (remove file extensions, paths)
    atlas_clean = Path(atlas_name).stem.replace('.nii', '')
    
    # Build filename with atlas information
    filename_parts = [input_stem, f"atlas-{atlas_clean}"]
    
    # Add desc if provided
    if desc is not None:
        filename_parts.append(f"desc-{desc}")
    
    filename_parts.append("timeseries")
    
    return "_".join(filename_parts)


def resolve_atlas_path(atlas_spec: str, space: str = "MNI152NLin2009cAsym", desc: str = None) -> str:
    """
    Resolve atlas specification to a file path.
    
    Supports both local file paths and TemplateFlow atlas names.
    
    Parameters
    ----------
    atlas_spec : str
        Atlas specification (file path or TemplateFlow name)
    space : str, optional
        Template space for TemplateFlow atlases (default: MNI152NLin2009cAsym)
    desc : str, optional
        Atlas description/variant (e.g., 800Parcels7Networks for Schaefer2018)
        
    Returns
    -------
    str
        Resolved atlas path
        
    Raises
    ------
    ValueError
        If TemplateFlow atlas name is unsupported
    FileNotFoundError
        If atlas file path doesn't exist
    RuntimeError
        If TemplateFlow download fails
    """
    atlas_path = Path(atlas_spec)
    
    # Check if it's a file path that exists
    if atlas_path.exists():
        return str(atlas_path)
    
    # Check if it looks like a TemplateFlow atlas name
    templateflow_patterns = [
        'schaefer2018', 'aal', 'harvardoxford', 'destrieux', 'desikankilliany', 'difumo'
    ]
    
    if any(pattern.lower() in atlas_spec.lower() for pattern in templateflow_patterns):
        try:
            # Try to download from TemplateFlow
            tf_manager = TemplateFlowManager()
            kwargs = {}
            if desc is not None:
                kwargs['desc'] = desc
            return tf_manager.get_atlas(atlas_spec, space, **kwargs)
        except ImportError:
            raise RuntimeError(
                f"TemplateFlow is not installed. To use atlas '{atlas_spec}', install it with:\n"
                f"uv add templateflow\n"
                f"Or provide a local atlas file path instead."
            )
        except ValueError as e:
            # ValueError from TemplateFlow contains helpful error messages, pass through
            raise ValueError(str(e)) from e
        except Exception as e:
            raise RuntimeError(
                f"Failed to download atlas '{atlas_spec}' from TemplateFlow: {e}"
            )
    
    # If it's not a TemplateFlow name and file doesn't exist, raise FileNotFoundError
    raise FileNotFoundError(f"Atlas file not found: {atlas_spec}")


def create_parser() -> argparse.ArgumentParser:
    """
    Create command-line argument parser.
    
    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        prog='parcelextract',
        description='Extract time-series signals from 4D neuroimaging data using brain parcellation schemes.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '--input',
        required=True,
        help='Path to input 4D Nifti file (.nii or .nii.gz)'
    )
    
    parser.add_argument(
        '--atlas', 
        required=True,
        help='Path to atlas Nifti file or TemplateFlow atlas name (e.g., Schaefer2018, AAL)'
    )
    
    parser.add_argument(
        '--output-dir',
        required=True,
        help='Output directory for results'
    )
    
    # Optional arguments
    parser.add_argument(
        '--strategy',
        choices=['mean', 'median', 'pca', 'weighted_mean'],
        default='mean',
        help='Signal extraction strategy'
    )
    
    parser.add_argument(
        '--space',
        default='MNI152NLin2009cAsym',
        help='Template space for TemplateFlow atlases (default: MNI152NLin2009cAsym)'
    )
    
    parser.add_argument(
        '--desc',
        help='Atlas description/variant (e.g., 800Parcels7Networks for Schaefer2018)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 1.0.0'
    )
    
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    """
    Main entry point for command-line interface.
    
    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments. If None, uses sys.argv.
    """
    parser = create_parser()
    args = parser.parse_args(argv)
    
    if args.verbose:
        print(f"ParcelExtract v1.0.0")
        print(f"Input: {args.input}")
        print(f"Atlas: {args.atlas}")
        print(f"Space: {args.space}")
        if args.desc:
            print(f"Description: {args.desc}")
        print(f"Output: {args.output_dir}")
        print(f"Strategy: {args.strategy}")
    
    try:
        # Resolve atlas path
        if args.verbose and not Path(args.atlas).exists():
            print(f"Downloading atlas '{args.atlas}' from TemplateFlow...")
        
        atlas_path = resolve_atlas_path(args.atlas, args.space, args.desc)
        
        # Create extractor
        extractor = ParcelExtractor(atlas=atlas_path, strategy=args.strategy)
        
        # Extract timeseries
        if args.verbose:
            print("Extracting timeseries...")
        
        timeseries = extractor.fit_transform(args.input)
        
        # Prepare output paths
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate output filename based on input, atlas, and desc
        output_stem = generate_output_filename(args.input, args.atlas, args.desc)
        tsv_file = output_dir / f"{output_stem}.tsv"
        json_file = output_dir / f"{output_stem}.json"
        
        # Write outputs
        write_timeseries_tsv(timeseries, tsv_file)
        
        metadata = {
            'extraction_strategy': args.strategy,
            'atlas': str(args.atlas),
            'n_parcels': timeseries.shape[0],
            'n_timepoints': timeseries.shape[1],
            'input_file': str(args.input)
        }
        write_json_sidecar(metadata, json_file)
        
        if args.verbose:
            print(f"Results written to: {output_dir}")
            print(f"Timeseries: {tsv_file}")
            print(f"Metadata: {json_file}")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()