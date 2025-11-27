import pytest
import numpy as np
import xarray as xr
from pathlib import Path
from unittest.mock import MagicMock, patch

from microimage.readers import LSMReader, TIFReader, CZIReader, Reader


@pytest.fixture
def mock_lsm_file_multi_scene():
    """Mocks a tifffile.TiffFile object for a multi-scene LSM file."""
    mock_tif = MagicMock()

    # --- Mock Series Data ---
    mock_series = MagicMock()
    mock_series.axes = "PZYXC"  # P for Position/Scene
    mock_series.shape = (2, 5, 64, 64, 3)  # 2 scenes, 5 z-slices, 64x64, 3 channels
    mock_tif.series = [mock_series]

    # --- Mock Metadata (LSMReader uses lsm_metadata dict) ---
    mock_tif.lsm_metadata = {
        "VoxelSizeZ": 0.5,
        "VoxelSizeY": 0.1,
        "VoxelSizeX": 0.1,
        "TimeIntervall": 0,
        "DimensionZ": 5,
        "DimensionY": 64,
        "DimensionX": 64,
        "DimensionTime": 1,
        "ChannelColors": {"ColorNames": ["DAPI", "GFP", "Cy5"]},
    }

    # --- Mock asarray method on the series object ---
    # The reader calls series.asarray(), so we mock that.
    mock_series.asarray.return_value = np.ones((2, 5, 64, 64, 3), dtype=np.uint8)

    return mock_tif


@pytest.fixture
def mock_lsm_file_single_scene():
    """Mocks a tifffile.TiffFile object for a single-scene LSM file."""
    mock_tif = MagicMock()
    mock_series = MagicMock()
    mock_series.axes = "ZYXC"  # No 'P' axis
    mock_series.shape = (5, 64, 64, 3)
    mock_tif.series = [mock_series]
    mock_tif.lsm_metadata = {
        "VoxelSizeZ": 0.5,
        "VoxelSizeY": 0.1,
        "VoxelSizeX": 0.1,
        "TimeIntervall": 0,
        "DimensionZ": 5,
        "DimensionY": 64,
        "DimensionX": 64,
        "DimensionTime": 1,
        "ChannelColors": {"ColorNames": ["DAPI", "GFP", "Cy5"]},
    }
    mock_tif.asarray.return_value = np.zeros((5, 64, 64, 3), dtype=np.uint8)
    return mock_tif


def test_lsm_reader_multi_scene_produces_xarray(mock_lsm_file_multi_scene):
    """
    Verify that LSMReader correctly yields xarray.DataArray objects for a
    multi-scene file, complete with dims, coords, and attrs.
    """
    dummy_path = Path("test.lsm")
    with patch(
        "tifffile.TiffFile", return_value=mock_lsm_file_multi_scene
    ) as mock_tiff_class:
        # Use the main Reader factory to get an LSMReader instance
        with Reader(dummy_path) as reader:
            assert isinstance(reader, LSMReader)
            assert len(reader) == 2  # From the 'P' axis shape

            scenes = list(reader.scenes())
            assert len(scenes) == 2

            # --- Verify the first scene ---
            scene_one = scenes[0]
            assert isinstance(scene_one, xr.DataArray)

            # Check dimensions and shape (P axis should be gone)
            assert scene_one.dims == ("Z", "Y", "X", "C")
            assert scene_one.shape == (5, 64, 64, 3)

            # Check coordinates were parsed from lsm_metadata
            assert "C" in scene_one.coords
            assert "Z" in scene_one.coords
            np.testing.assert_array_equal(
                scene_one.coords["C"].values, ["DAPI", "GFP", "Cy5"]
            )
            np.testing.assert_allclose(scene_one.coords["Z"].values, np.arange(5) * 0.5)

            # Check that the data is correct for the first scene
            assert np.all(scene_one.values == 1.0)

            # --- Verify the second scene ---
            scene_two = scenes[1]
            assert isinstance(scene_two, xr.DataArray)
            assert np.all(scene_two.values == 1.0)  # Data is the same in this mock


def test_lsm_reader_single_scene(mock_lsm_file_single_scene):
    """
    Verify that LSMReader handles a file without a 'P' axis correctly.
    """
    dummy_path = Path("test_single.lsm")
    with patch("tifffile.TiffFile", return_value=mock_lsm_file_single_scene):
        with Reader(dummy_path) as reader:
            assert isinstance(reader, LSMReader)
            assert len(reader) == 1

            scenes = list(reader.scenes())
            assert len(scenes) == 1

            scene = scenes[0]
            assert isinstance(scene, xr.DataArray)
            assert scene.dims == ("Z", "Y", "X", "C")
            assert scene.shape == (5, 64, 64, 3)
