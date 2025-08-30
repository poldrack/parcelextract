"""Test command-line interface functionality."""

import pytest
from pathlib import Path
from unittest.mock import patch
import sys

from parcelextract.cli.main import main, create_parser


class TestCLIMain:
    """Test CLI main function and entry point."""

    def test_main_function_exists(self):
        """Smoke test: main() function should exist and be callable."""
        # Should not raise an exception when imported
        assert callable(main)

    def test_main_with_help_flag(self):
        """Test main() with --help flag displays help and exits cleanly."""
        with patch.object(sys, 'argv', ['parcelextract', '--help']):
            with pytest.raises(SystemExit) as exc_info:
                main()
            # Help should exit with code 0
            assert exc_info.value.code == 0

    def test_main_with_no_args_shows_usage(self):
        """Test main() with no arguments shows usage information."""
        with patch.object(sys, 'argv', ['parcelextract']):
            with pytest.raises(SystemExit) as exc_info:
                main()
            # Should exit with error code (missing required args)
            assert exc_info.value.code != 0


class TestCLIParser:
    """Test command-line argument parser."""

    def test_create_parser_returns_parser(self):
        """Test that create_parser() returns an ArgumentParser."""
        parser = create_parser()
        assert hasattr(parser, 'parse_args')
        assert hasattr(parser, 'add_argument')

    def test_parser_has_required_arguments(self):
        """Test that parser includes required CLI arguments."""
        parser = create_parser()
        
        # Test with minimal required arguments
        args = parser.parse_args([
            '--input', 'test_input.nii.gz',
            '--atlas', 'test_atlas.nii.gz',
            '--output-dir', 'output'
        ])
        
        assert args.input == 'test_input.nii.gz'
        assert args.atlas == 'test_atlas.nii.gz'
        assert args.output_dir == 'output'

    def test_parser_has_optional_arguments(self):
        """Test that parser includes optional CLI arguments."""
        parser = create_parser()
        
        # Test with all arguments
        args = parser.parse_args([
            '--input', 'test_input.nii.gz',
            '--atlas', 'test_atlas.nii.gz', 
            '--output-dir', 'output',
            '--strategy', 'median',
            '--verbose'
        ])
        
        assert args.strategy == 'median'
        assert args.verbose is True


class TestCLIIntegration:
    """Test CLI end-to-end functionality."""

    def test_cli_full_workflow(self, temp_dir):
        """Test complete CLI workflow with real data."""
        import numpy as np
        import nibabel as nib
        from unittest.mock import patch
        
        # Create test 4D image
        img_4d = np.random.randn(4, 4, 4, 10).astype(np.float32)
        img_nii = nib.Nifti1Image(img_4d, affine=np.eye(4))
        img_file = temp_dir / "test_bold.nii.gz"
        nib.save(img_nii, img_file)
        
        # Create test atlas
        atlas_data = np.array([
            [[0, 0, 1, 1], [1, 1, 2, 2], [2, 2, 0, 0], [0, 1, 1, 2]],
            [[1, 1, 0, 0], [2, 2, 1, 1], [0, 0, 2, 2], [1, 2, 0, 1]], 
            [[2, 0, 1, 2], [0, 1, 2, 0], [1, 2, 0, 1], [2, 0, 1, 2]],
            [[1, 2, 1, 0], [0, 1, 0, 2], [2, 0, 2, 1], [1, 2, 1, 0]]
        ]).astype(np.int16)
        atlas_nii = nib.Nifti1Image(atlas_data, affine=np.eye(4))
        atlas_file = temp_dir / "test_atlas.nii.gz"
        nib.save(atlas_nii, atlas_file)
        
        # Create output directory
        output_dir = temp_dir / "results"
        
        # Prepare CLI arguments
        argv = [
            '--input', str(img_file),
            '--atlas', str(atlas_file),
            '--output-dir', str(output_dir),
            '--strategy', 'mean',
            '--verbose'
        ]
        
        # Run CLI main function
        main(argv)
        
        # Check outputs were created
        tsv_file = output_dir / "test_bold_timeseries.tsv"
        json_file = output_dir / "test_bold_timeseries.json"
        
        assert tsv_file.exists()
        assert json_file.exists()
        
        # Check TSV content
        import pandas as pd
        df = pd.read_csv(tsv_file, sep='\t')
        assert df.shape == (10, 2)  # 10 timepoints, 2 parcels
        assert list(df.columns) == ['parcel_0', 'parcel_1']
        
        # Check JSON content
        import json
        with open(json_file, 'r') as f:
            metadata = json.load(f)
        
        assert metadata['extraction_strategy'] == 'mean'
        assert metadata['n_parcels'] == 2
        assert metadata['n_timepoints'] == 10
        assert str(atlas_file) in metadata['atlas']
        assert str(img_file) in metadata['input_file']