"""Test command-line interface functionality."""

import pytest
from pathlib import Path
from unittest.mock import patch
import sys

from parcelextract.cli.main import main, create_parser, resolve_atlas_path


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
            '--space', 'MNI152NLin6Asym',
            '--verbose'
        ])
        
        assert args.strategy == 'median'
        assert args.space == 'MNI152NLin6Asym'
        assert args.verbose is True

    def test_parser_has_default_space(self):
        """Test that parser has default space argument."""
        parser = create_parser()
        
        # Test with minimal required arguments
        args = parser.parse_args([
            '--input', 'test_input.nii.gz',
            '--atlas', 'test_atlas.nii.gz',
            '--output-dir', 'output'
        ])
        
        # Should have default space
        assert args.space == 'MNI152NLin2009cAsym'

    def test_parser_has_desc_argument(self):
        """Test that parser includes --desc argument for atlas variants."""
        parser = create_parser()
        
        # Test with desc argument
        args = parser.parse_args([
            '--input', 'test_input.nii.gz',
            '--atlas', 'Schaefer2018',
            '--output-dir', 'output',
            '--desc', '800Parcels7Networks'
        ])
        
        assert args.desc == '800Parcels7Networks'

    def test_parser_desc_is_optional(self):
        """Test that --desc argument is optional."""
        parser = create_parser()
        
        # Test without desc argument
        args = parser.parse_args([
            '--input', 'test_input.nii.gz',
            '--atlas', 'Schaefer2018',
            '--output-dir', 'output'
        ])
        
        # Should be None when not specified
        assert args.desc is None


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
        
        # Check outputs were created (with local atlas file, no desc)
        tsv_file = output_dir / "test_atlas-test_atlas_timeseries.tsv"
        json_file = output_dir / "test_atlas-test_atlas_timeseries.json"
        
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

    def test_cli_templateflow_workflow_with_desc(self, temp_dir):
        """Test CLI workflow with TemplateFlow atlas and desc parameter."""
        import numpy as np
        import nibabel as nib
        from unittest.mock import patch
        
        # Create test 4D image
        img_4d = np.random.randn(4, 4, 4, 10).astype(np.float32)
        img_nii = nib.Nifti1Image(img_4d, affine=np.eye(4))
        img_file = temp_dir / "sub-01_ses-3T_task-motor_contrast-lh_stat-fixzscore_bold.nii.gz"
        nib.save(img_nii, img_file)
        
        # Create output directory
        output_dir = temp_dir / "results"
        
        # Mock TemplateFlow functionality
        with patch('parcelextract.cli.main.TemplateFlowManager') as mock_tf_manager:
            mock_manager = mock_tf_manager.return_value
            # Create mock atlas file
            mock_atlas_file = temp_dir / "downloaded_atlas.nii.gz"
            atlas_data = np.array([
                [[0, 0, 1, 1], [1, 1, 2, 2], [2, 2, 0, 0], [0, 1, 1, 2]],
                [[1, 1, 0, 0], [2, 2, 1, 1], [0, 0, 2, 2], [1, 2, 0, 1]], 
                [[2, 0, 1, 2], [0, 1, 2, 0], [1, 2, 0, 1], [2, 0, 1, 2]],
                [[1, 2, 1, 0], [0, 1, 0, 2], [2, 0, 2, 1], [1, 2, 1, 0]]
            ]).astype(np.int16)
            atlas_nii = nib.Nifti1Image(atlas_data, affine=np.eye(4))
            nib.save(atlas_nii, mock_atlas_file)
            mock_manager.get_atlas.return_value = str(mock_atlas_file)
            
            # Prepare CLI arguments with TemplateFlow atlas and desc
            argv = [
                '--input', str(img_file),
                '--atlas', 'Schaefer2018',
                '--desc', '800Parcels7Networks',
                '--output-dir', str(output_dir),
                '--strategy', 'mean',
                '--verbose'
            ]
            
            # Run CLI main function
            main(argv)
            
            # Check that TemplateFlow was called with desc parameter
            mock_manager.get_atlas.assert_called_once_with(
                'Schaefer2018', 'MNI152NLin2009cAsym', desc='800Parcels7Networks'
            )
        
        # Check outputs were created with proper BIDS naming
        expected_tsv = output_dir / "sub-01_ses-3T_task-motor_contrast-lh_stat-fixzscore_atlas-Schaefer2018_desc-800Parcels7Networks_timeseries.tsv"
        expected_json = output_dir / "sub-01_ses-3T_task-motor_contrast-lh_stat-fixzscore_atlas-Schaefer2018_desc-800Parcels7Networks_timeseries.json"
        
        assert expected_tsv.exists()
        assert expected_json.exists()
        
        # Check TSV content
        import pandas as pd
        df = pd.read_csv(expected_tsv, sep='\t')
        assert df.shape == (10, 2)  # 10 timepoints, 2 parcels
        assert list(df.columns) == ['parcel_0', 'parcel_1']
        
        # Check JSON content
        import json
        with open(expected_json, 'r') as f:
            metadata = json.load(f)
        
        assert metadata['extraction_strategy'] == 'mean'
        assert metadata['n_parcels'] == 2
        assert metadata['n_timepoints'] == 10
        assert 'Schaefer2018' in metadata['atlas']
        assert str(img_file) in metadata['input_file']

    def test_cli_templateflow_workflow_no_desc(self, temp_dir):
        """Test CLI workflow with TemplateFlow atlas but no desc parameter."""
        import numpy as np
        import nibabel as nib
        from unittest.mock import patch
        
        # Create test 4D image with complex BIDS filename
        img_4d = np.random.randn(4, 4, 4, 10).astype(np.float32)
        img_nii = nib.Nifti1Image(img_4d, affine=np.eye(4))
        img_file = temp_dir / "sub-01_task-rest_run-01_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
        nib.save(img_nii, img_file)
        
        # Create output directory
        output_dir = temp_dir / "results"
        
        # Mock TemplateFlow functionality
        with patch('parcelextract.cli.main.TemplateFlowManager') as mock_tf_manager:
            mock_manager = mock_tf_manager.return_value
            # Create mock atlas file
            mock_atlas_file = temp_dir / "downloaded_atlas.nii.gz"
            atlas_data = np.array([
                [[0, 0, 1, 1], [1, 1, 2, 2], [2, 2, 0, 0], [0, 1, 1, 2]],
                [[1, 1, 0, 0], [2, 2, 1, 1], [0, 0, 2, 2], [1, 2, 0, 1]], 
                [[2, 0, 1, 2], [0, 1, 2, 0], [1, 2, 0, 1], [2, 0, 1, 2]],
                [[1, 2, 1, 0], [0, 1, 0, 2], [2, 0, 2, 1], [1, 2, 1, 0]]
            ]).astype(np.int16)
            atlas_nii = nib.Nifti1Image(atlas_data, affine=np.eye(4))
            nib.save(atlas_nii, mock_atlas_file)
            mock_manager.get_atlas.return_value = str(mock_atlas_file)
            
            # Prepare CLI arguments with TemplateFlow atlas but no desc
            argv = [
                '--input', str(img_file),
                '--atlas', 'AAL',
                '--output-dir', str(output_dir),
                '--strategy', 'mean'
            ]
            
            # Run CLI main function
            main(argv)
            
            # Check that TemplateFlow was called without desc parameter
            mock_manager.get_atlas.assert_called_once_with(
                'AAL', 'MNI152NLin2009cAsym'
            )
        
        # Check outputs were created with proper BIDS naming (no desc)
        expected_tsv = output_dir / "sub-01_task-rest_run-01_space-MNI152NLin2009cAsym_desc-preproc_atlas-AAL_timeseries.tsv"
        expected_json = output_dir / "sub-01_task-rest_run-01_space-MNI152NLin2009cAsym_desc-preproc_atlas-AAL_timeseries.json"
        
        assert expected_tsv.exists()
        assert expected_json.exists()


class TestFilenameGeneration:
    """Test output filename generation functionality."""
    
    def test_generate_output_filename_basic(self):
        """Test basic filename generation."""
        from parcelextract.cli.main import generate_output_filename
        
        result = generate_output_filename(
            "sub-01_task-rest_bold.nii.gz", 
            "Schaefer2018"
        )
        expected = "sub-01_task-rest_atlas-Schaefer2018_timeseries"
        assert result == expected
    
    def test_generate_output_filename_with_desc(self):
        """Test filename generation with desc parameter."""
        from parcelextract.cli.main import generate_output_filename
        
        result = generate_output_filename(
            "sub-01_ses-3T_task-motor_contrast-lh_stat-fixzscore_bold.nii.gz",
            "Schaefer2018",
            "800Parcels7Networks"
        )
        expected = "sub-01_ses-3T_task-motor_contrast-lh_stat-fixzscore_atlas-Schaefer2018_desc-800Parcels7Networks_timeseries"
        assert result == expected
    
    def test_generate_output_filename_local_atlas(self):
        """Test filename generation with local atlas file."""
        from parcelextract.cli.main import generate_output_filename
        
        result = generate_output_filename(
            "test_data_bold.nii.gz",
            "/path/to/custom_atlas.nii.gz"
        )
        expected = "test_data_atlas-custom_atlas_timeseries"
        assert result == expected
    
    def test_generate_output_filename_removes_bold_suffix(self):
        """Test that _bold suffix is properly removed."""
        from parcelextract.cli.main import generate_output_filename
        
        result = generate_output_filename(
            "sub-01_task-rest_run-01_bold.nii.gz",
            "AAL"
        )
        expected = "sub-01_task-rest_run-01_atlas-AAL_timeseries"
        assert result == expected


class TestAtlasResolution:
    """Test atlas path resolution functionality."""

    def test_resolve_existing_atlas_file(self, temp_dir):
        """Test resolving existing atlas file."""
        # Create test atlas file
        atlas_file = temp_dir / "test_atlas.nii.gz"
        atlas_file.touch()
        
        # Should return the file path
        resolved = resolve_atlas_path(str(atlas_file))
        assert resolved == str(atlas_file)

    @patch('parcelextract.cli.main.TemplateFlowManager')
    def test_resolve_templateflow_atlas_calls_manager(self, mock_tf_manager):
        """Test that TemplateFlow atlas names call the manager."""
        mock_manager = mock_tf_manager.return_value
        mock_manager.get_atlas.return_value = "/path/to/downloaded/atlas.nii.gz"
        
        result = resolve_atlas_path("Schaefer2018", "MNI152NLin2009cAsym")
        
        mock_tf_manager.assert_called_once()
        mock_manager.get_atlas.assert_called_once_with("Schaefer2018", "MNI152NLin2009cAsym")
        assert result == "/path/to/downloaded/atlas.nii.gz"

    @patch('parcelextract.cli.main.TemplateFlowManager')
    def test_resolve_templateflow_atlas_with_desc(self, mock_tf_manager):
        """Test that TemplateFlow atlas resolution passes desc parameter."""
        mock_manager = mock_tf_manager.return_value
        mock_manager.get_atlas.return_value = "/path/to/atlas_800parcels.nii.gz"
        
        result = resolve_atlas_path("Schaefer2018", "MNI152NLin2009cAsym", "800Parcels7Networks")
        
        mock_tf_manager.assert_called_once()
        mock_manager.get_atlas.assert_called_once_with(
            "Schaefer2018", "MNI152NLin2009cAsym", desc="800Parcels7Networks"
        )
        assert result == "/path/to/atlas_800parcels.nii.gz"

    def test_resolve_nonexistent_file_raises_error(self):
        """Test that nonexistent files raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError) as exc_info:
            resolve_atlas_path("/nonexistent/atlas.nii.gz")
        
        assert "Atlas file not found" in str(exc_info.value)