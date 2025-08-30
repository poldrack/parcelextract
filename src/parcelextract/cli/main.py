"""Command-line interface for ParcelExtract."""

import argparse
import sys
from pathlib import Path
from typing import Optional, List

from parcelextract.core.extractor import ParcelExtractor
from parcelextract.io.writers import write_timeseries_tsv, write_json_sidecar


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
        help='Path to atlas Nifti file or TemplateFlow atlas name'
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
        print(f"Output: {args.output_dir}")
        print(f"Strategy: {args.strategy}")
    
    try:
        # Create extractor
        extractor = ParcelExtractor(atlas=args.atlas, strategy=args.strategy)
        
        # Extract timeseries
        if args.verbose:
            print("Extracting timeseries...")
        
        timeseries = extractor.fit_transform(args.input)
        
        # Prepare output paths
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate output filename based on input
        input_stem = Path(args.input).stem.replace('.nii', '')
        tsv_file = output_dir / f"{input_stem}_timeseries.tsv"
        json_file = output_dir / f"{input_stem}_timeseries.json"
        
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