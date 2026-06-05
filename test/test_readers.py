from pathlib import Path
from unittest.mock import MagicMock, patch

import dask.array as da
import numpy as np
import pytest
import xarray as xr
import zarr

from microimage.readers import (
    CZIReader,
    LSMReader,
    ND2Reader,
    Reader,
)


@pytest.fixture
def mock_lsm_file_multi_scene():
    """Mocks a tifffile.TiffFile object for a multi-scene LSM file."""
    mock_tif = MagicMock()
    shape = (2, 5, 64, 64, 3)  # 2 scenes, 5 z-slices, 64x64, 3 channels
    store = zarr.storage.MemoryStore()
    # --- Mock Series Data ---
    mock_series = MagicMock()
    mock_series.axes = "PZYXC"  # P for Position/Scene
    mock_series.shape = shape  # 2 scenes, 5 z-slices, 64x64, 3 channels
    mock_series.aszarr.return_value = zarr.create(shape=shape, store=store)
    mock_tif.series = [mock_series]

    # --- Mock Metadata (LSMReader uses lsm_metadata dict) ---
    mock_tif.lsm_metadata = {
        "VoxelSizeZ": 6.5e-7,
        "VoxelSizeY": 6.5e-8,
        "VoxelSizeX": 6.5e-8,
        "TimeIntervall": 0,
        "DimensionZ": shape[1],
        "DimensionY": shape[2],
        "DimensionX": shape[3],
        "DimensionTime": 1,
        "DimensionChannels": shape[4],
        "ChannelColors": {"ColorNames": ["DAPI", "GFP", "Cy5"]},
    }

    return mock_tif


@pytest.fixture
def mock_lsm_file_single_scene():
    """Mocks a tifffile.TiffFile object for a single-scene LSM file."""
    shape = (5, 64, 64, 3)  # 5 z-slices, 64x64, 3 channels
    store = zarr.storage.MemoryStore()
    mock_tif = MagicMock()
    mock_series = MagicMock()
    mock_series.axes = "ZYXC"  # No 'P' axis
    mock_series.shape = shape
    mock_series.aszarr.return_value = zarr.create(shape=shape, store=store)
    mock_tif.series = [mock_series]
    mock_tif.lsm_metadata = {
        "VoxelSizeZ": 6.5e-7,
        "VoxelSizeY": 6.5e-8,
        "VoxelSizeX": 6.5e-8,
        "TimeIntervall": 0,
        "DimensionZ": shape[0],
        "DimensionY": shape[1],
        "DimensionX": shape[2],
        "DimensionTime": 1,
        "DimensionChannels": shape[3],
        "ChannelColors": {"ColorNames": ["DAPI", "GFP", "Cy5"]},
    }

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

            scenes = list(reader.iter_scenes())
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
            np.testing.assert_allclose(
                scene_one.coords["Z"].values, np.arange(5) * 6.5e-7
            )

            # Check that the data is correct for the first scene
            assert np.all(scene_one.compute().values == 0.0)

            # --- Verify the second scene ---
            scene_two = scenes[1]
            assert isinstance(scene_two, xr.DataArray)
            assert np.all(
                scene_two.compute().values == 0.0
            )  # Data is the same in this mock


def test_lsm_reader_single_scene(mock_lsm_file_single_scene):
    """
    Verify that LSMReader handles a file without a 'P' axis correctly.
    """
    dummy_path = Path("test_single.lsm")
    with patch("tifffile.TiffFile", return_value=mock_lsm_file_single_scene):
        with Reader(dummy_path) as reader:
            assert isinstance(reader, LSMReader)
            assert len(reader) == 1

            scenes = list(reader.iter_scenes())
            assert len(scenes) == 1

            scene = scenes[0]
            assert isinstance(scene, xr.DataArray)
            assert scene.dims == ("Z", "Y", "X", "C")
            assert scene.shape == (5, 64, 64, 3)


@pytest.fixture
def mock_czi_file():
    """Mocks a pylibczirw.CziFile object for a multi-scene CZI."""
    mock_czi = MagicMock()

    # --- Mock Dimensions and Shape ---
    # 2 scenes, 5 Z, 64 Y, 64 X, 3 C
    dims_shape_dict = {
        "S": (0, 2),
        "Z": (0, 5),
        "Y": (0, 64),
        "X": (0, 64),
        "C": (0, 3),
    }
    all_dims = "SZCYX"
    mock_czi.get_dims_shape.return_value = (dims_shape_dict, all_dims)

    # --- Mock Scenes Bounding Rectangle ---
    # This must match the 'S' dimension size (2) for consistency.
    mock_scene_rect = MagicMock()
    mock_scene_rect.h = 64
    mock_scene_rect.w = 64
    mock_czi.scenes_bounding_rectangle = [mock_scene_rect, mock_scene_rect]

    # --- Mock Metadata ---
    mock_czi.metadata = {
        "ImageDocument": {
            "Metadata": {
                "Experiment": "Test1",
                "Sample": "CellCulture",
                "Scaling": {
                    "Items": {
                        "Distance": [
                            {"@Id": "X", "Value": 0.1},
                            {"@Id": "Y", "Value": 0.1},
                            {"@Id": "Z", "Value": 0.5},
                        ]
                    }
                },
            }
        }
    }
    mock_czi.channel_names = ["DAPI", "GFP", "mCherry"]
    mock_czi.get_dimension_pixel_size.side_effect = lambda dim: {
        "Z": 0.5,
        "Y": 0.1,
        "X": 0.1,
    }.get(dim)

    # --- Mock Data Reading ---
    mock_czi.pixel_type.to_numpy_dtype.return_value = np.uint16
    # This mock will be called by dask.delayed, so it should return the data directly
    mock_czi.read_image.return_value = np.ones((5, 3, 64, 64), dtype=np.uint16)

    return mock_czi


def test_czi_reader_multi_scene(mock_czi_file):
    """Verify CZIReader correctly handles a multi-scene file with lazy loading."""
    dummy_path = Path("test.czi")
    # Mock the context manager returned by open_czi
    mock_context = MagicMock()
    mock_context.__enter__.return_value = mock_czi_file
    mock_context.__exit__.return_value = None

    with patch(
        "microimage.readers.pyczi.open_czi", return_value=mock_context
    ) as mock_open_czi:
        with Reader(dummy_path) as reader:
            assert isinstance(reader, CZIReader)
            assert len(reader) == 2  # 2 scenes from 'S' dimension

            scenes = list(reader.iter_scenes())
            assert len(scenes) == 2

            scene_one = scenes[0]
            assert isinstance(scene_one, xr.DataArray)
            assert isinstance(scene_one.data, da.Array)  # Check for Dask array

            # Check dimensions and shape (S is removed)
            assert scene_one.dims == ("Z", "C", "Y", "X")
            assert scene_one.shape == (5, 3, 64, 64)

            # Check coordinates
            np.testing.assert_array_equal(
                scene_one.coords["C"].values, ["DAPI", "GFP", "mCherry"]
            )
            np.testing.assert_allclose(scene_one.coords["Z"].values, np.arange(5) * 0.5)

            # Check attributes
            assert scene_one.attrs["Experiment"] == "Test1"

            # Trigger computation and check data
            computed_data = scene_one.compute()
            assert np.all(computed_data.values == 1)


@pytest.fixture
def mock_nd2_file():
    """Mocks an nd2.ND2File object."""
    mock_nd2 = MagicMock()

    # --- Mock Dimensions ---
    mock_nd2.sizes = {"P": 2, "T": 10, "Z": 5, "C": 3, "Y": 128, "X": 128}

    # --- Mock Data Reading (to_xarray) ---
    # Create a mock xarray that the nd2 mock can return
    mock_data = da.zeros((2, 10, 5, 3, 128, 128), dtype=np.uint16)
    mock_xarray = xr.DataArray(
        mock_data,
        dims=list("PTZCYX"),
        attrs={"description": "A mock ND2 file"},
        coords={"C": ["ch1", "ch2", "ch3"]},
    )
    mock_nd2.to_xarray.return_value = mock_xarray

    return mock_nd2


def test_nd2_reader_multi_position(mock_nd2_file):
    """Verify ND2Reader correctly iterates through positions."""
    dummy_path = Path("../data/WellA01_Seq0000.nd2")
    with patch("nd2.ND2File", return_value=mock_nd2_file):
        with Reader(dummy_path) as reader:
            assert isinstance(reader, ND2Reader)
            assert len(reader) == 2  # From 'P' dimension

            scenes = list(reader.iter_scenes())
            assert len(scenes) == 2

            scene_one = scenes[0]
            assert isinstance(scene_one, xr.DataArray)
            assert isinstance(scene_one.data, da.Array)

            # Check dimensions (P should be sliced away)
            assert scene_one.dims == ("time", "Z", "C", "Y", "X")
            assert scene_one.shape == (10, 5, 3, 128, 128)
            assert scene_one.attrs["description"] == "A mock ND2 file"
            np.testing.assert_array_equal(
                scene_one.coords["C"].values, ["ch1", "ch2", "ch3"]
            )
