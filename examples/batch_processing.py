#!/usr/bin/env python3
"""
ParcelExtract Batch Processing Example

This script demonstrates how to process multiple subjects/sessions
using ParcelExtract in a BIDS-like directory structure.

Usage:
    python examples/batch_processing.py [data_dir] [output_dir]

Example BIDS structure:
    data/
    ├── sub-01/
    │   ├── ses-1/
    │   │   └── func/
    │   │       ├── sub-01_ses-1_task-rest_bold.nii.gz
    │   │       └── sub-01_ses-1_task-motor_bold.nii.gz
    │   └── ses-2/
    │       └── func/
    │           └── sub-01_ses-2_task-rest_bold.nii.gz
    └── sub-02/
        └── func/
            └── sub-02_task-rest_bold.nii.gz

Output structure:
    derivatives/parcelextract/
    ├── sub-01/
    │   ├── ses-1/
    │   │   └── func/
    │   │       ├── sub-01_ses-1_task-rest_atlas-Schaefer2018_desc-400Parcels17Networks_timeseries.tsv
    │   │       ├── sub-01_ses-1_task-rest_atlas-Schaefer2018_desc-400Parcels17Networks_timeseries.json
    │   │       ├── sub-01_ses-1_task-motor_atlas-Schaefer2018_desc-400Parcels17Networks_timeseries.tsv
    │   │       └── sub-01_ses-1_task-motor_atlas-Schaefer2018_desc-400Parcels17Networks_timeseries.json
    │   └── ses-2/
    └── sub-02/
"""

import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import time

import nibabel as nib
import numpy as np
import pandas as pd

from parcelextract.core.extractor import ParcelExtractor
from parcelextract.io.writers import write_timeseries_tsv, write_json_sidecar
from parcelextract.atlases.templateflow import TemplateFlowManager


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BatchProcessor:
    """Batch processing of neuroimaging data with ParcelExtract."""
    
    def __init__(
        self, 
        atlas_spec: str, 
        strategy: str = 'mean',
        desc: Optional[str] = None,
        space: str = 'MNI152NLin2009cAsym',
        n_jobs: int = 1
    ):
        """
        Initialize batch processor.
        
        Parameters
        ----------
        atlas_spec : str
            Atlas specification (TemplateFlow name or file path)
        strategy : str
            Extraction strategy ('mean', 'median', 'pca', 'weighted_mean')
        desc : str, optional
            Atlas description/variant for TemplateFlow atlases
        space : str
            Template space for TemplateFlow atlases
        n_jobs : int
            Number of parallel jobs (1 = sequential)
        """
        self.atlas_spec = atlas_spec
        self.strategy = strategy
        self.desc = desc
        self.space = space
        self.n_jobs = n_jobs
        
        # Resolve atlas path
        self.atlas_path = self._resolve_atlas()
        
        logger.info(f"Initialized BatchProcessor:")
        logger.info(f"  Atlas: {self.atlas_spec} -> {self.atlas_path}")
        logger.info(f"  Strategy: {self.strategy}")
        logger.info(f"  Parallel jobs: {self.n_jobs}")
    
    def _resolve_atlas(self) -> str:
        """Resolve atlas specification to file path."""
        atlas_path = Path(self.atlas_spec)
        
        if atlas_path.exists():
            logger.info(f"Using local atlas file: {atlas_path}")
            return str(atlas_path)
        
        # Try TemplateFlow
        try:
            logger.info(f"Downloading TemplateFlow atlas: {self.atlas_spec}")
            tf_manager = TemplateFlowManager()
            
            kwargs = {}
            if self.desc is not None:
                kwargs['desc'] = self.desc
            
            atlas_path = tf_manager.get_atlas(
                self.atlas_spec, 
                self.space, 
                **kwargs
            )
            logger.info(f"TemplateFlow atlas downloaded: {atlas_path}")
            return atlas_path
            
        except Exception as e:
            raise ValueError(f"Could not resolve atlas '{self.atlas_spec}': {e}")
    
    def find_bold_files(self, data_dir: Path) -> List[Path]:
        """
        Find all BOLD files in BIDS-like structure.
        
        Parameters
        ----------
        data_dir : Path
            Root directory containing subject data
            
        Returns
        -------
        List[Path]
            List of BOLD file paths
        """
        patterns = [
            '**/*_bold.nii.gz',
            '**/*_bold.nii',
            '**/func/*_bold.nii.gz',
            '**/func/*_bold.nii'
        ]
        
        bold_files = []
        for pattern in patterns:
            bold_files.extend(data_dir.glob(pattern))
        
        # Remove duplicates and sort
        bold_files = sorted(set(bold_files))
        
        logger.info(f"Found {len(bold_files)} BOLD files in {data_dir}")
        return bold_files
    
    def process_single_file(
        self, 
        bold_file: Path, 
        output_dir: Path
    ) -> Dict[str, any]:
        """
        Process a single BOLD file.
        
        Parameters
        ----------
        bold_file : Path
            Path to BOLD file
        output_dir : Path
            Output directory for results
            
        Returns
        -------
        dict
            Processing results and metadata
        """
        start_time = time.time()
        
        try:
            logger.info(f"Processing: {bold_file.name}")
            
            # Initialize extractor
            extractor = ParcelExtractor(
                atlas=self.atlas_path,
                strategy=self.strategy
            )
            
            # Extract time-series
            timeseries = extractor.fit_transform(str(bold_file))
            
            # Generate output paths
            input_stem = bold_file.stem.replace('.nii', '')
            
            # Remove BIDS suffix (_bold) for output naming
            if input_stem.endswith('_bold'):
                input_stem = input_stem[:-5]  # Remove '_bold'
            
            # Build output filename with atlas info
            output_parts = [input_stem, f"atlas-{Path(self.atlas_spec).stem}"]
            if self.desc:
                output_parts.append(f"desc-{self.desc}")
            output_parts.append("timeseries")
            
            output_stem = "_".join(output_parts)
            
            tsv_file = output_dir / f"{output_stem}.tsv"
            json_file = output_dir / f"{output_stem}.json"
            
            # Create output directory
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save time-series
            write_timeseries_tsv(timeseries, tsv_file)
            
            # Create metadata
            metadata = {
                'extraction_strategy': self.strategy,
                'atlas': self.atlas_spec,
                'atlas_file': self.atlas_path,
                'n_parcels': int(timeseries.shape[0]),
                'n_timepoints': int(timeseries.shape[1]),
                'input_file': str(bold_file),
                'processing_time': time.time() - start_time,
                'parcelextract_version': '1.0.0'
            }
            
            if self.desc:
                metadata['atlas_desc'] = self.desc
            if hasattr(self, 'space'):
                metadata['template_space'] = self.space
            
            # Save metadata
            write_json_sidecar(metadata, json_file)
            
            result = {
                'bold_file': str(bold_file),
                'tsv_file': str(tsv_file),
                'json_file': str(json_file),
                'n_parcels': timeseries.shape[0],
                'n_timepoints': timeseries.shape[1],
                'processing_time': time.time() - start_time,
                'status': 'success',
                'error': None
            }
            
            logger.info(f"  ✓ Success: {timeseries.shape} -> {tsv_file.name}")
            return result
            
        except Exception as e:
            error_msg = f"Error processing {bold_file.name}: {e}"
            logger.error(error_msg)
            
            return {
                'bold_file': str(bold_file),
                'tsv_file': None,
                'json_file': None,
                'n_parcels': None,
                'n_timepoints': None,
                'processing_time': time.time() - start_time,
                'status': 'failed',
                'error': str(e)
            }
    
    def process_batch(
        self, 
        data_dir: Path, 
        output_dir: Path
    ) -> pd.DataFrame:
        """
        Process all BOLD files in batch.
        
        Parameters
        ----------
        data_dir : Path
            Input data directory
        output_dir : Path
            Output directory for all results
            
        Returns
        -------
        pandas.DataFrame
            Summary of processing results
        """
        # Find all BOLD files
        bold_files = self.find_bold_files(data_dir)
        
        if not bold_files:
            logger.warning(f"No BOLD files found in {data_dir}")
            return pd.DataFrame()
        
        # Prepare output directories
        results = []
        
        if self.n_jobs == 1:
            # Sequential processing
            logger.info("Processing files sequentially...")
            
            for bold_file in bold_files:
                # Preserve directory structure
                rel_path = bold_file.relative_to(data_dir)
                file_output_dir = output_dir / rel_path.parent
                
                result = self.process_single_file(bold_file, file_output_dir)
                results.append(result)
        
        else:
            # Parallel processing
            logger.info(f"Processing files in parallel ({self.n_jobs} jobs)...")
            
            with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                # Submit all jobs
                future_to_file = {}
                
                for bold_file in bold_files:
                    rel_path = bold_file.relative_to(data_dir)
                    file_output_dir = output_dir / rel_path.parent
                    
                    future = executor.submit(
                        self.process_single_file, 
                        bold_file, 
                        file_output_dir
                    )
                    future_to_file[future] = bold_file
                
                # Collect results
                for future in as_completed(future_to_file):
                    result = future.result()
                    results.append(result)
        
        # Create summary DataFrame
        df = pd.DataFrame(results)
        
        # Add summary statistics
        n_success = sum(r['status'] == 'success' for r in results)
        n_failed = sum(r['status'] == 'failed' for r in results)
        total_time = sum(r['processing_time'] for r in results)
        
        logger.info(f"\nBatch processing complete:")
        logger.info(f"  Total files: {len(results)}")
        logger.info(f"  Successful: {n_success}")
        logger.info(f"  Failed: {n_failed}")
        logger.info(f"  Total time: {total_time:.1f}s")
        logger.info(f"  Average time per file: {total_time/len(results):.1f}s")
        
        return df
    
    def generate_summary_report(
        self, 
        results_df: pd.DataFrame, 
        output_dir: Path
    ) -> None:
        """Generate processing summary report."""
        if results_df.empty:
            logger.warning("No results to summarize")
            return
        
        # Create summary
        summary = {
            'processing_summary': {
                'total_files': len(results_df),
                'successful': int(sum(results_df['status'] == 'success')),
                'failed': int(sum(results_df['status'] == 'failed')),
                'total_processing_time': float(results_df['processing_time'].sum()),
                'mean_processing_time': float(results_df['processing_time'].mean())
            },
            'extraction_parameters': {
                'atlas': self.atlas_spec,
                'strategy': self.strategy,
                'space': self.space
            },
            'file_details': results_df.to_dict('records')
        }
        
        if self.desc:
            summary['extraction_parameters']['atlas_desc'] = self.desc
        
        # Save summary
        summary_file = output_dir / 'processing_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Processing summary saved: {summary_file}")
        
        # Save results table
        results_file = output_dir / 'processing_results.tsv'
        results_df.to_csv(results_file, sep='\t', index=False)
        
        logger.info(f"Results table saved: {results_file}")


def create_demo_data(data_dir: Path, n_subjects: int = 2):
    """Create demo BIDS data for testing."""
    logger.info(f"Creating demo data with {n_subjects} subjects...")
    
    subjects = [f'sub-{i+1:02d}' for i in range(n_subjects)]
    tasks = ['rest', 'motor']
    
    for subject in subjects:
        # Some subjects have sessions
        if subject == 'sub-01':
            sessions = ['ses-1', 'ses-2']
        else:
            sessions = [None]
        
        for session in sessions:
            if session:
                func_dir = data_dir / subject / session / 'func'
                session_str = f"_{session}"
            else:
                func_dir = data_dir / subject / 'func'
                session_str = ""
            
            func_dir.mkdir(parents=True, exist_ok=True)
            
            # Create BOLD files for different tasks
            for task in tasks:
                if session == 'ses-2' and task == 'motor':
                    continue  # Skip motor task for ses-2
                
                # Create synthetic BOLD data
                bold_data = np.random.randn(32, 32, 20, 100).astype(np.float32)
                bold_img = nib.Nifti1Image(bold_data, affine=np.eye(4))
                
                bold_file = func_dir / f"{subject}{session_str}_task-{task}_bold.nii.gz"
                nib.save(bold_img, bold_file)
    
    logger.info(f"Demo data created in {data_dir}")


def main():
    """Main batch processing function."""
    parser = argparse.ArgumentParser(
        description='Batch processing with ParcelExtract',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        'data_dir',
        nargs='?',
        default='demo_data',
        help='Input data directory (BIDS-like structure)'
    )
    
    parser.add_argument(
        'output_dir', 
        nargs='?',
        default='derivatives/parcelextract',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--atlas',
        default='Schaefer2018',
        help='Atlas specification (TemplateFlow name or file path)'
    )
    
    parser.add_argument(
        '--desc',
        default='400Parcels17Networks',
        help='Atlas description/variant'
    )
    
    parser.add_argument(
        '--strategy',
        choices=['mean', 'median', 'pca', 'weighted_mean'],
        default='mean',
        help='Extraction strategy'
    )
    
    parser.add_argument(
        '--space',
        default='MNI152NLin2009cAsym',
        help='Template space for TemplateFlow atlases'
    )
    
    parser.add_argument(
        '--n-jobs',
        type=int,
        default=1,
        help='Number of parallel jobs (1 = sequential)'
    )
    
    parser.add_argument(
        '--create-demo',
        action='store_true',
        help='Create demo data before processing'
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    # Create demo data if requested
    if args.create_demo:
        create_demo_data(data_dir)
    
    # Check input directory
    if not data_dir.exists():
        logger.error(f"Data directory does not exist: {data_dir}")
        logger.info("Use --create-demo to create sample data")
        sys.exit(1)
    
    try:
        # Initialize batch processor
        processor = BatchProcessor(
            atlas_spec=args.atlas,
            strategy=args.strategy,
            desc=args.desc,
            space=args.space,
            n_jobs=args.n_jobs
        )
        
        # Process batch
        results_df = processor.process_batch(data_dir, output_dir)
        
        # Generate summary report
        if not results_df.empty:
            processor.generate_summary_report(results_df, output_dir)
        
        logger.info("Batch processing completed successfully!")
        
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()