"""Output writing utilities for timeseries data."""

from pathlib import Path
from typing import Union

import json
import numpy as np
import pandas as pd


def write_timeseries_tsv(timeseries_data: np.ndarray, output_file: Union[str, Path]) -> None:
    """
    Write timeseries data to a TSV file.

    Parameters
    ----------
    timeseries_data : np.ndarray
        2D array with shape (n_parcels, n_timepoints).
    output_file : str or Path
        Output file path for TSV file.
    """
    # Convert to Path object
    output_path = Path(output_file)
    
    # Create parent directories if they don't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Transpose data so timepoints are rows and parcels are columns
    data_transposed = timeseries_data.T
    
    # Create column names (parcel_0, parcel_1, etc.)
    n_parcels = timeseries_data.shape[0]
    column_names = [f"parcel_{i}" for i in range(n_parcels)]
    
    # Create DataFrame
    df = pd.DataFrame(data_transposed, columns=column_names)
    
    # Write to TSV file
    df.to_csv(output_path, sep='\t', index=False)


def write_json_sidecar(metadata: dict, output_file: Union[str, Path]) -> None:
    """
    Write metadata to a JSON sidecar file.
    
    Parameters
    ----------
    metadata : dict
        Dictionary containing metadata to write.
    output_file : str or Path
        Output file path for JSON file.
    """
    # Convert to Path object
    output_path = Path(output_file)
    
    # Create parent directories if they don't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write JSON file
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)