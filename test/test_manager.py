from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr

from microimage import DataItem, ExperimentManager, MetadataParser


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
        np.zeros((10, 10)), dims=("Y", "X"), attrs={"unique_id": "id_0"}
    )
    mock_xarray_1 = xr.DataArray(
        np.zeros((10, 10)), dims=("Y", "X"), attrs={"unique_id": "id_1"}
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


@pytest.fixture
def mock_data_item_2d():
    """Mocks a DataItem representing a 2D image."""
    return DataItem(
        id="mock_2d_id",
        relative_path=Path("mock_2d.tif"),
        scene="0",
        condition="control",
        annotated_channels=["channel_A"],
    )


@pytest.fixture
def mock_data_item_3d():
    """Mocks a DataItem representing a 3D (Z, Y, X) image."""
    return DataItem(
        id="mock_3d_id",
        relative_path=Path("mock_3d.czi"),
        scene="0",
        condition="treated",
        annotated_channels=["channel_B"],
    )


def test_load_item_data_normalized_dims(
    mock_dirs, mock_data_item_2d, mock_data_item_3d
):
    """
    Verify that load_item_data always returns a 5D xarray.DataArray
    with dimensions (T, C, Z, Y, X), regardless of the input file's
    original dimensionality.
    """
    data_dir, results_dir, db_path = mock_dirs
    manager = ExperimentManager(
        data_dir=data_dir, results_dir=results_dir, db_path=db_path
    )

    # --- Test 2D input (Y, X) ---
    mock_2d_xarray = xr.DataArray(
        np.zeros((10, 10), dtype=np.uint8),
        dims=("Y", "X"),
        attrs={"unique_id": mock_data_item_2d.id},
    )
    mock_reader_2d = MagicMock()
    mock_reader_2d.iter_scenes_normalized.return_value = iter([mock_2d_xarray])

    with patch("microimage.readers.Reader", return_value=mock_reader_2d):
        # Mock get_absolute_container_path to return a dummy path
        with patch.object(
            manager,
            "get_absolute_container_path",
            return_value=data_dir / mock_data_item_2d.relative_path,
        ):
            loaded_data_2d = manager.load_item_data(mock_data_item_2d)

            assert isinstance(loaded_data_2d, xr.DataArray)
            assert loaded_data_2d.ndim == 5
            assert loaded_data_2d.dims == ("T", "C", "Z", "Y", "X")
            assert loaded_data_2d.shape == (
                1,
                1,
                1,
                10,
                10,
            )  # T, C, Z should be padded to 1

    # --- Test 3D input (Z, Y, X) ---
    mock_3d_xarray = xr.DataArray(
        np.zeros((5, 20, 20), dtype=np.uint8),
        dims=("Z", "Y", "X"),
        attrs={"unique_id": mock_data_item_3d.id},
    )
    mock_reader_3d = MagicMock()
    mock_reader_3d.iter_scenes_normalized.return_value = iter([mock_3d_xarray])

    with patch("microimage.readers.Reader", return_value=mock_reader_3d):
        with patch.object(
            manager,
            "get_absolute_container_path",
            return_value=data_dir / mock_data_item_3d.relative_path,
        ):
            loaded_data_3d = manager.load_item_data(mock_data_item_3d)

            assert isinstance(loaded_data_3d, xr.DataArray)
            assert loaded_data_3d.ndim == 5
            assert loaded_data_3d.dims == ("T", "C", "Z", "Y", "X")
            assert loaded_data_3d.shape == (
                1,
                1,
                5,
                20,
                20,
            )  # T, C should be padded to 1
