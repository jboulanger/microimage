import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import xarray as xr
import numpy as np

from microimage import ExperimentManager, MetadataParser


@pytest.fixture
def mock_dirs(tmp_path):
    """Create temporary data, results, and db directories for testing."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    db_path = tmp_path / "project.db"
    return data_dir, results_dir, db_path


def test_manager_scan_with_parser(mock_dirs):
    """
    Tests that the ExperimentManager can scan a directory and use the MetadataParser
    to register new items in the database.
    """
    data_dir, results_dir, db_path = mock_dirs

    # Create a dummy file for the scanner to find
    dummy_file = data_dir / "test_image.lsm"
    dummy_file.touch()

    # Create mock xarrays to satisfy the manager's requirement for dimensional extraction
    # and unique ID retrieval.
    mock_xarray_0 = xr.DataArray(
        np.zeros((1, 1)), dims=("Y", "X"), attrs={"unique_id": "id_0"}
    )
    mock_xarray_1 = xr.DataArray(
        np.zeros((1, 1)), dims=("Y", "X"), attrs={"unique_id": "id_1"}
    )

    # Mock the MetadataParser to avoid actual file reading
    mock_parser = MagicMock(spec=MetadataParser)
    mock_parser.return_value = [
        {"scene": "0", "metadata": {"channel": "DAPI"}, "_scene_xarray": mock_xarray_0},
        {"scene": "1", "metadata": {"channel": "GFP"}, "_scene_xarray": mock_xarray_1},
    ]

    manager = ExperimentManager(data_dir, results_dir, db_path)
    manager.scan_data_directory(extensions=[".lsm"], multi_file_parser=mock_parser)

    assert len(manager) == 2
    assert "id_0" in manager
    assert "id_1" in manager

    # Verify scene is mapped correctly
    assert manager["id_1"].scene == "1"

    # Note: Currently manager.py has metadata insertion commented out in _process_container_file.
    # If uncommented, we would assert:
    # assert manager["id_1"].metadata["channel"] == "GFP"

    mock_parser.assert_called_once_with(dummy_file)
