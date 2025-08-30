"""Test output writing functionality."""

import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from parcelextract.io.writers import write_timeseries_tsv, write_json_sidecar


class TestWriteTimeseriesTSV:
    """Test TSV output writing functionality."""

    def test_write_basic_tsv_file(self, temp_dir):
        """Test writing basic timeseries data to TSV file."""
        # Create sample timeseries data
        timeseries_data = np.array([
            [1.0, 2.0, 3.0, 4.0, 5.0],  # Parcel 1
            [0.5, 1.5, 2.5, 3.5, 4.5],  # Parcel 2
            [0.1, 0.2, 0.3, 0.4, 0.5],  # Parcel 3
        ])
        
        output_file = temp_dir / "test_timeseries.tsv"
        
        # Should write TSV file successfully
        write_timeseries_tsv(timeseries_data, output_file)
        
        # File should exist
        assert output_file.exists()
        
        # Should be readable as TSV
        df = pd.read_csv(output_file, sep='\t')
        
        # Should have correct shape (3 parcels, 5 timepoints)
        assert df.shape == (5, 3)  # timepoints x parcels
        
        # Should have correct column names (parcel_0, parcel_1, parcel_2)
        expected_columns = ['parcel_0', 'parcel_1', 'parcel_2']
        assert list(df.columns) == expected_columns
        
        # Should have correct data (transposed - timepoints as rows)
        np.testing.assert_array_almost_equal(df.values, timeseries_data.T)

    def test_write_tsv_creates_output_directory(self, temp_dir):
        """Test that TSV writing creates output directory if it doesn't exist."""
        # Create sample data
        timeseries_data = np.array([[1.0, 2.0], [3.0, 4.0]])
        
        # Create nested directory path that doesn't exist
        nested_dir = temp_dir / "derivatives" / "parcelextract" / "sub-01" / "ses-01"
        output_file = nested_dir / "timeseries.tsv"
        
        # Directory should not exist initially
        assert not nested_dir.exists()
        
        # Should create directories and write file
        write_timeseries_tsv(timeseries_data, output_file)
        
        # Directory should now exist
        assert nested_dir.exists()
        assert nested_dir.is_dir()
        
        # File should exist and be readable
        assert output_file.exists()
        df = pd.read_csv(output_file, sep='\t')
        assert df.shape == (2, 2)  # 2 timepoints, 2 parcels


class TestWriteJSONSidecar:
    """Test JSON sidecar output writing functionality."""

    def test_write_basic_json_sidecar(self, temp_dir):
        """Test writing basic metadata to JSON sidecar file."""
        # Create sample metadata
        metadata = {
            "extraction_method": "mean",
            "atlas": "test_atlas",
            "n_parcels": 3,
            "n_timepoints": 5
        }
        
        output_file = temp_dir / "test_metadata.json"
        
        # Should write JSON file successfully
        write_json_sidecar(metadata, output_file)
        
        # File should exist
        assert output_file.exists()
        
        # Should be readable as JSON
        with open(output_file, 'r') as f:
            loaded_metadata = json.load(f)
        
        # Should have correct metadata
        assert loaded_metadata == metadata

    def test_write_json_creates_output_directory(self, temp_dir):
        """Test that JSON writing creates output directory if it doesn't exist."""
        # Create sample metadata
        metadata = {
            "extraction_method": "median",
            "atlas": "schaefer2018",
            "n_parcels": 100
        }
        
        # Create nested directory path that doesn't exist
        nested_dir = temp_dir / "derivatives" / "parcelextract" / "sub-01" / "ses-01"
        output_file = nested_dir / "metadata.json"
        
        # Directory should not exist initially
        assert not nested_dir.exists()
        
        # Should create directories and write file
        write_json_sidecar(metadata, output_file)
        
        # Directory should now exist
        assert nested_dir.exists()
        assert nested_dir.is_dir()
        
        # File should exist and be readable
        assert output_file.exists()
        with open(output_file, 'r') as f:
            loaded_metadata = json.load(f)
        assert loaded_metadata == metadata