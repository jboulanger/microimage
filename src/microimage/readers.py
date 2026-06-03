from abc import ABC, abstractmethod
import itertools
from typing import Dict, Any, List, Generator, Self, Iterator, Tuple
from pathlib import Path
import tifffile
import ome_types
import numpy as np
import xarray as xr
import dask.array as da
import hashlib
import base64
import os
import dask

try:
    from pylibCZIrw import czi as pyczi
except ImportError:
    pyczi = None

try:
    import nd2
except ImportError:
    nd2 = None

try:
    import liffile
except ImportError:
    liffile = None

try:
    import slideio
except ImportError:
    slideio = None

try:
    from imaris_ims_file_reader import ims as ims_reader
except ImportError:
    ims_reader = None


def generate_scene_id(path: Path, scene_index: int) -> str:
    """Generates a unique SHA1 hash ID for a scene.

    The ID is based on the file's name, its last modification time (in
    nanoseconds), and the scene's index within the file.
    """

    if not isinstance(path, Path):
        path = Path(path)

    try:
        mod_time = os.stat(path).st_mtime_ns
    except (FileNotFoundError, OSError):
        # Fallback to 0 if the file doesn't exist (common during testing with mock paths)
        mod_time = 0
    id_string = f"{path.name}-{mod_time}-{scene_index}"
    hasher = hashlib.sha256(id_string.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(hasher).decode("utf-8")[:8]


class BaseReader(ABC):
    """Abstract base class for all image file readers.

    This class defines the common interface that all concrete reader
    implementations must follow. It is designed to be used as a context
    manager.

    Parameters
    ----------
    path : Path
        The file path to the image to be read.

    """

    def __init__(self, path: Path):
        self.path = path

    @abstractmethod
    def __enter__(self) -> Self:
        pass

    @abstractmethod
    def __exit__(self, type, value, traceback):
        pass

    def __iter__(self) -> Iterator[xr.DataArray]:
        return self.iter_scenes_normalized()

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of scenes in the file.

        Returns
        -------
        int
            The total number of scenes.

        """
        pass

    def iter_scenes_normalized(self) -> Generator[xr.DataArray, None, None]:
        """Yield each scene with dimensions normalized to (T, C, Z, Y, X).

        This method wraps `iter_scenes`, reordering and expanding dimensions
        as necessary to ensure a consistent 5D output format (T, C, Z, Y, X).
        Missing dimensions are added as singletons.

        Yields
        ------
        xr.DataArray
            An xarray DataArray with dimensions (T, C, Z, Y, X).
        """
        target_dims = ("T", "C", "Z", "Y", "X")
        for scene in self.iter_scenes():
            # Standardize time dimension naming if 'time' is used instead of 'T'
            if "time" in scene.dims:
                scene = scene.rename({"time": "T"})

            # Identify and add missing dimensions as singletons
            missing_dims = [d for d in target_dims if d not in scene.dims]
            if missing_dims:
                scene = scene.expand_dims({d: 1 for d in missing_dims})

            # Reorder dimensions to the standard (T, C, Z, Y, X)
            # Any unexpected extra dimensions are appended after the target ones.
            order = list(target_dims) + [d for d in scene.dims if d not in target_dims]
            yield scene.transpose(*order)

    @abstractmethod
    def iter_scenes(self) -> Generator[xr.DataArray, None, None]:
        """Yield each scene from the image file as a lazy `dask`-backed DataArray.

        This method is a generator that iterates through all scenes in
        the file, yielding each one as an `xarray.DataArray`.

        Yields
        ------
        xr.DataArray
            An xarray DataArray representing a single scene from the file,
            complete with dimensions, coordinates, and metadata where available.

        """
        pass


READER_REGISTRY = {}


class LSMReader(BaseReader):
    """A reader for Zeiss LSM (Laser Scanning Microscope) files.

    This reader uses `tifffile` to parse `.lsm` files. It can handle both
    single-scene and multi-scene (mosaic) files, where scenes are indexed
    by the 'P' (Position) axis.

    Parameters
    ----------
    path : Path
        The file path to the `.lsm` file.

    Attributes
    ----------
    tif : tifffile.TiffFile
        The underlying `TiffFile` object.
    axes_str : str
        A string representing the axes of the primary image series (e.g., 'PZYXC').
    metadata : dict
        A dictionary containing the LSM-specific metadata.

    """

    def __init__(self, path: Path):
        super().__init__(path)
        self.tif = tifffile.TiffFile(path)
        self.axes_str = self.tif.series[0].axes
        self.metadata = self.tif.lsm_metadata or {}

    def __enter__(self) -> Self:
        return self

    def __exit__(self, type, value, traceback):
        self.tif.close()

    def __len__(self) -> int:
        """Return the number of scenes in the LSM file.

        If a 'P' (Position) axis is present, its length is returned.
        Otherwise, it is assumed to be a single-scene file.

        Returns
        -------
        int
            The number of scenes.

        """
        p_index = self.axes_str.find("P")
        if p_index >= 0:
            return self.tif.series[0].shape[p_index]
        return 1

    def iter_scenes(self) -> Generator[xr.DataArray, None, None]:
        """Yield scenes from the LSM file as xarray DataArrays.

        If the file contains multiple scenes (a 'P' axis), this generator
        iterates through each scene. Otherwise, it yields the single primary
        image. It attempts to parse dimensional coordinates from the LSM metadata.

        Yields
        ------
        xr.DataArray
            A DataArray for each scene, with dimensions, coordinates, and metadata.

        """
        try:
            coords = {
                "T": np.arange(self.metadata["DimensionTime"])
                * self.metadata.get("TimeIntervall", 1.0),
                "Z": [
                    z * self.metadata["VoxelSizeZ"]
                    for z in range(self.metadata["DimensionZ"])
                ],
                "Y": [
                    y * self.metadata["VoxelSizeY"]
                    for y in range(self.metadata["DimensionY"])
                ],
                "X": [
                    x * self.metadata["VoxelSizeX"]
                    for x in range(self.metadata["DimensionX"])
                ],
                "C": self.metadata["ChannelColors"]["ColorNames"],
            }
            # Convert time in seconds to timedelta objects
            if "T" in coords:
                coords["T"] = [np.timedelta64(int(s * 1e9), "ns") for s in coords["T"]]

        except Exception:
            coords = {}
        p_index = self.axes_str.find("P")

        if p_index >= 0:
            num_scenes = self.tif.series[0].shape[p_index]

            for scene_index in range(num_scenes):
                store = self.tif.series[0].aszarr()
                data = dask.array.from_zarr(store)[scene_index]
                scene_id = generate_scene_id(self.path, scene_index)
                scene_attrs = {**self.metadata, "unique_id": scene_id}
                scene_dims = [d for d in self.axes_str if d != "P"]
                scene_coords = {a: coords[a] for a in scene_dims if a in coords}
                yield xr.DataArray(
                    data,
                    dims=scene_dims,
                    coords=scene_coords,
                    attrs=scene_attrs,
                    name=f"scene {scene_index}",
                )
        else:
            store = self.tif.series[0].aszarr()
            data = dask.array.from_zarr(store)
            scene_dims = [d for d in current_axes if d != "P"]
            scene_coords = {a: coords[a] for a in scene_dims if a in coords}
            scene_id = generate_scene_id(self.path, 0)
            scene_attrs = {**self.metadata, "unique_id": scene_id}
            yield xr.DataArray(
                data,
                dims=scene_dims,
                coords=scene_coords,
                attrs=scene_attrs,
                name="scene 0",
            )


READER_REGISTRY[".lsm"] = LSMReader


class TIFReader(BaseReader):
    """A generic reader for TIFF files.

    This reader treats each series within a TIFF file as a separate scene.
    It leverages `tifffile` for reading and `ome-types` to parse OME-XML
    metadata if it is present, creating richly annotated `xarray.DataArray`
    objects.

    Parameters
    ----------
    path : Path
        The file path to the `.tif` or `.tiff` file.

    Attributes
    ----------
    tif : tifffile.TiffFile
        The underlying `TiffFile` object.
    metadata : dict
        A dictionary for storing metadata (currently unused for generic TIFFs).
    ome : ome_types.model.OME, optional
        The parsed OME-XML data structure if available, otherwise None.

    """

    def __init__(self, path: Path):
        super().__init__(path)
        self.tif = tifffile.TiffFile(path)
        self.metadata = {}  # Generic TIFFs have varied metadata
        try:
            self.ome = ome_types.from_xml(self.tif.ome_metadata)
        except Exception:
            self.ome = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, type, value, traceback):
        self.tif.close()

    def __len__(self) -> int:
        return len(self.tif.series)

    def iter_scenes(self) -> Generator[xr.DataArray, None, None]:
        """Yield each series in the TIFF file as a separate scene.

        This generator iterates through all series in the TIFF file. If OME-XML
        metadata is available, it attempts to extract dimensional coordinates
        (like physical size and channel names) and attaches them to the
        `xarray.DataArray`.

        Yields
        ------
        xr.DataArray
            A DataArray for each series, with dimensions corresponding to the
            series axes. Coordinates and attributes are populated from OME
            metadata if present.

        Notes
        -----
        If OME metadata is not present or fails to parse, the resulting DataArray will have dimensions but may lack detailed coordinates.
        """
        for i, series in enumerate(self.tif.series):
            data = series.asarray(dask=True)
            dims = list(series.axes)
            coords = {}
            attrs = {}
            # Try to get coordinates and metadata from OME structure
            if self.ome and i < len(self.ome.images):
                pixels = self.ome.images[i].pixels
                attrs = {"name": self.ome.images[i].name}  # noqa
                coords = self._get_ome_coords(pixels, dims, data.shape)

            scene_id = generate_scene_id(self.path, i)
            yield xr.DataArray(
                data,
                dims=dims,
                coords=coords,
                attrs={**attrs, "unique_id": scene_id},
                name=f"scene {i}",
            )

    def _get_ome_coords(self, pixels, dims, shape) -> Dict:
        """Helper to extract coordinates from OME.pixels."""
        coords = {}
        try:
            possible_coords = {
                "T": None,  # Handle time separately
                "Z": [z * pixels.physical_size_z for z in range(pixels.size_z)]
                if pixels.physical_size_z
                else None,
                "Y": [y * pixels.physical_size_y for y in range(pixels.size_y)]
                if pixels.physical_size_y
                else None,
                "X": [x * pixels.physical_size_x for x in range(pixels.size_x)]
                if pixels.physical_size_x
                else None,
                "C": [
                    c.name or str(ch_idx) for ch_idx, c in enumerate(pixels.channels)
                ],
            }

            # Handle time coordinates with datetime if possible
            if pixels.size_t > 1 and pixels.physical_size_t is not None:
                time_increment_s = pixels.physical_size_t  # Assuming seconds
                # If we have an absolute start time, create datetime objects
                if self.ome.images and self.ome.images[0].acquisition_date:
                    start_time = np.datetime64(self.ome.images[0].acquisition_date)
                    time_deltas = np.arange(pixels.size_t) * time_increment_s
                    possible_coords["T"] = start_time + np.array(
                        [np.timedelta64(int(s * 1e9), "ns") for s in time_deltas]
                    )
                else:  # Otherwise, use timedeltas from zero
                    time_deltas = np.arange(pixels.size_t) * time_increment_s
                    possible_coords["T"] = [
                        np.timedelta64(int(s * 1e9), "ns") for s in time_deltas
                    ]

            # Filter out empty or mismatched coords
            for d, c in possible_coords.items():
                if d in dims and c is not None and len(c) == shape[dims.index(d)]:
                    coords[d] = c
        except Exception:
            return {}  # Fallback on any parsing error
        return coords


READER_REGISTRY[".tif"] = TIFReader
READER_REGISTRY[".tiff"] = TIFReader


class CZIReader(BaseReader):
    """A reader for Zeiss CZI (Carl Zeiss Imaging) files.

    This reader uses the `pylibczirw` library to parse `.czi` files. It
    handles multi-scene mosaic images by iterating over the 'S' dimension.

    Parameters
    ----------
    path : Path
        The file path to the `.czi` file.

    Attributes
    ----------
    czi : pylibczirw.CziFile
        The underlying `CziFile` object.
    metadata : dict
        A dictionary containing the file's metadata.

    Raises
    ------
    ImportError
        If the `pylibczirw` library is not installed.
    """

    def __init__(self, path: Path):
        super().__init__(path)
        if pyczi is None:
            raise ImportError(
                "pylibczirw is not installed. Please run 'pip install pylibczirw'"
            )
        # Manually manage the context from pyczi.open_czi
        self.czi_context = pyczi.open_czi(str(path))
        self.czi = self.czi_context.__enter__()
        self.metadata = self.czi.metadata
        self.to_numpy_dtype = {"Gray8": np.uint8, "Gray16": np.uint16}

    def __enter__(self) -> Self:
        return self

    def __exit__(self, type, value, traceback):
        self.czi_context.__exit__(type, value, traceback)

    def __len__(self) -> int:
        """Return the number of scenes in the CZI file.

        The number of scenes corresponds to the size of the 'S' dimension,
        which represents mosaic positions. If 'S' is not present, it defaults
        to 1.

        Returns
        -------
        int
            The number of scenes.
        """
        return self.czi.get_dims_shape()[0].get("S", (0, 1))[1]

    def iter_scenes(self) -> Generator[xr.DataArray, None, None]:
        """Yield scenes from the CZI file as xarray DataArrays.

        This generator iterates through each scene (mosaic tile) in the file,
        reading its image data and constructing an `xarray.DataArray`. It
        extracts physical scaling for dimensions and channel names to create
        coordinates.

        Yields
        ------
        xr.DataArray
            A DataArray for each scene, with dimensions, coordinates, and metadata.
        """

        num_scenes = len(self.czi.scenes_bounding_rectangle)

        # Get the full shape from total_bounding_box, excluding 'S' (scene)
        full_czi_shape = {
            x: self.czi.total_bounding_box[x][1] - self.czi.total_bounding_box[x][0]
            for x in self.czi.total_bounding_box
        }
        if "S" in full_czi_shape:
            del full_czi_shape["S"]

        # Dimensions for iteration (planes) - these are non-spatial dimensions.
        # We iterate over all non-spatial dimensions, even if they are singletons.
        # This dictionary will be used to generate `planes` for `czi.read`.
        plane_iteration_dims = {
            k: v for k, v in full_czi_shape.items() if k not in ["X", "Y"]
        }

        # Get the dimension keys and their corresponding ranges of indices for planes
        dim_keys = list(plane_iteration_dims.keys())
        index_ranges = [range(size) for size in plane_iteration_dims.values()]

        # Use itertools.product to get all combinations of indices for the planes
        # that will be read by czi.read.
        index_combinations = itertools.product(*index_ranges)

        # Create the list of dictionaries directly from the index combinations
        planes = [dict(zip(dim_keys, indices)) for indices in index_combinations]

        # Compute the coordinates
        distances = self.czi.metadata["ImageDocument"]["Metadata"]["Scaling"]["Items"][
            "Distance"
        ]

        scales = {d["@Id"]: float(d["Value"]) for d in distances}

        if num_scenes == 0:
            data_array_shape = full_czi_shape
            data_array_dims = list(data_array_shape.keys())
            data = da.stack(
                [
                    da.from_delayed(
                        dask.delayed(self.czi.read)(plane, pixel_type="Gray16"),
                        shape=[data_array_shape["Y"], data_array_shape["X"]],
                        dtype=np.uint16,
                    )
                    for plane in planes
                ]
            ).reshape([data_array_shape[k] for k in data_array_dims])

            coords = {}
            for dim in data_array_dims:
                if dim in scales:
                    coords[dim] = np.arange(data_array_shape[dim]) * scales[dim]
                else:
                    coords[dim] = np.arange(data_array_shape[dim])

            attrs = self.czi.metadata["ImageDocument"]["Metadata"]
            attrs["unique_id"] = generate_scene_id(self.path, 0)

            yield xr.DataArray(
                data,
                coords=coords,
                attrs=attrs,
                dims=data_array_dims,
                name="scene 0",
            )

        else:
            for scene_index in range(num_scenes):
                # For each scene, we need its specific Y and X dimensions from scenes_bounding_rectangle
                scene_y_size = self.czi.scenes_bounding_rectangle[scene_index].h
                scene_x_size = self.czi.scenes_bounding_rectangle[scene_index].w

                # The shape for this specific scene's DataArray
                scene_data_array_shape = (
                    plane_iteration_dims.copy()
                )  # Start with non-spatial dims
                scene_data_array_shape["Y"] = scene_y_size
                scene_data_array_shape["X"] = scene_x_size
                scene_data_array_dims = list(scene_data_array_shape.keys())

                data = da.stack(
                    [
                        da.from_delayed(
                            dask.delayed(self.czi.read)(
                                plane,
                                scene=scene_index,
                            ),
                            shape=[scene_y_size, scene_x_size],
                            dtype=np.uint16,
                        )
                        for plane in planes
                    ]
                ).reshape([scene_data_array_shape[k] for k in scene_data_array_dims])

                coords = {}
                for dim in scene_data_array_dims:
                    if dim in scales:
                        coords[dim] = (
                            np.arange(scene_data_array_shape[dim]) * scales[dim]
                        )
                    else:
                        coords[dim] = np.arange(scene_data_array_shape[dim])

                attrs = self.czi.metadata["ImageDocument"]["Metadata"]
                attrs["unique_id"] = generate_scene_id(self.path, scene_index)

                yield xr.DataArray(
                    data,
                    coords=coords,
                    attrs=attrs,
                    dims=scene_data_array_dims,  #
                    name=f"scene {scene_index}",
                )


READER_REGISTRY[".czi"] = CZIReader


class ND2Reader(BaseReader):
    """Reader for ND2 files from Nikon Elements.

    .. warning::
    Parameters
    ----------
    path : Path
        The file path to the `.nd2` file.
    """

    def __init__(self, path: Path):
        super().__init__(path)
        if nd2 is None:
            raise ImportError("nd2 is not installed. Please run 'pip install nd2'")
        self.nd2 = nd2.ND2File(path)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, type, value, traceback):
        self.nd2.close()

    def __len__(self) -> int:
        """Return the number of scenes in the ND2 file.

        The number of scenes corresponds to the size of the 'P' dimension
        (positions/fields of view). If 'P' is not present, it defaults to 1.

        Returns
        -------
        int
            The number of scenes.
        """
        return self.nd2.sizes.get("P", 1)

    def iter_scenes(self) -> Generator[xr.DataArray, None, None]:
        """Yield scenes from the ND2 file as xarray DataArrays.

        This generator uses the `nd2` library's `to_xarray` method to create
        a Dask-backed xarray DataArray for the entire file. If the file contains
        multiple scenes (indicated by a 'P' dimension), it iterates through
        and yields each scene. Otherwise, it yields the single DataArray.

        Yields
        ------
        xr.DataArray
            A DataArray for each scene, with dimensions, coordinates, and metadata.
        """
        # Get the full xarray DataArray, which will be dask-backed
        full_xarray = self.nd2.to_xarray(delayed=True)

        if "P" in full_xarray.dims:
            for p_idx in range(full_xarray.sizes["P"]):
                scene_data = full_xarray.isel(P=p_idx)
                scene_id = generate_scene_id(self.path, p_idx)
                scene_data.attrs["unique_id"] = scene_id
                yield scene_data
        else:
            full_xarray.attrs["unique_id"] = generate_scene_id(self.path, 0)
            yield full_xarray


READER_REGISTRY[".nd2"] = ND2Reader


class LIFReader(BaseReader):
    """Reader for LIF (Leica Image File) format from Leica LAS X.

    This reader uses the `liffile` library to parse `.lif` files. It treats
    each image series within the file as a separate scene.

    Parameters
    ----------
    path : Path
        The file path to the `.lif` file.

    Attributes
    ----------
    lif : liffile.LifFile
        The underlying `LifFile` object.

    Raises
    ------
    ImportError
        If the `liffile` library is not installed.
    """

    def __init__(self, path: Path):
        super().__init__(path)
        if liffile is None:
            raise ImportError(
                "liffile is not installed. Please run 'pip install liffile'"
            )
        self.lif = liffile.LifFile(path)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, type, value, traceback):
        self.lif.close()

    def __len__(self) -> int:
        """Return the number of image series in the LIF file."""
        return len(self.lif.series)

    def iter_scenes(self) -> Generator[xr.DataArray, None, None]:
        """Yield each image series in the LIF file as a separate scene.

        This generator iterates through all series in the LIF file, creating
        a Dask-backed `xarray.DataArray` for each one. It extracts dimensional
        scaling and names from the LIF metadata to populate the coordinates.

        Yields
        ------
        xr.DataArray
            A lazy-loaded DataArray for each series, with dimensions,
            coordinates, and metadata.
        """
        for scene_index, scene in enumerate(self.lif.images):
            name = scene.name
            data = self.lif.images[name].asxarray()

        for i, series in enumerate(self.lif.series):
            # Use dask=True for lazy loading
            data = series.asarray(dask=True)
            dims = list(series.axes)
            if "T" in dims:
                dims[dims.index("T")] = "time"
            attrs = {"name": series.name, **series.info}

            # Create coordinates from scale and unit info
            coords = {
                dim: np.arange(size) * scale
                for dim, size, scale in zip(dims, series.shape, series.scale)
                if scale is not None
            }
            scene_id = generate_scene_id(self.path, i)
            attrs["unique_id"] = scene_id
            yield xr.DataArray(data, dims=dims, coords=coords, attrs=attrs)


READER_REGISTRY[".lif"] = LIFReader


class VSIReader(BaseReader):
    """Reader for Olympus VSI (Virtual Slide Image) files.

    This reader uses the `slideio` library, which is designed for whole-slide
    images. It treats each scene within the VSI file as a separate item.

    Parameters
    ----------
    path : Path
        The file path to the `.vsi` file.
    chunk_size : Tuple[int, int], optional
        The (Y, X) dimensions for each Dask chunk, by default (2048, 2048).

    Attributes
    ----------
    slide : slideio.Slide
        The underlying `Slide` object from the slideio library.

    Raises
    ------
    ImportError
        If the `slideio` library is not installed.
    """

    def __init__(self, path: Path, chunk_size: Tuple[int, int] = (2048, 2048)):
        super().__init__(path)
        if slideio is None:
            raise ImportError(
                "slideio is not installed. Please run 'pip install slideio'"
            )
        self.chunk_size = chunk_size
        self.slide = slideio.open_slide(str(path), "VSI")

    def __enter__(self) -> Self:
        return self

    def __exit__(self, type, value, traceback):
        # slideio objects manage their own resources and do not have a close() method
        pass

    def __len__(self) -> int:
        """Return the number of scenes in the VSI file."""
        return self.slide.num_scenes

    def iter_scenes(self) -> Generator[xr.DataArray, None, None]:
        """Yield each scene from the VSI file as a lazy xarray DataArray.

        This generator iterates through all scenes in the VSI file. The reading
        of pixel data is deferred using `dask.delayed` to ensure lazy loading.
        Metadata such as resolution and channel names are extracted for the
        xarray object.

        Yields
        ------
        xr.DataArray
            A lazy-loaded DataArray for each scene.
        """
        for i in range(self.slide.num_scenes):
            scene = self.slide.get_scene(i)

            # Define shape, dtype, and dims for the dask array
            dims = ["Y", "X", "C"]
            shape = (scene.size[1], scene.size[0], scene.num_channels)
            dtype = scene.channel_data_types[0] if scene.num_channels > 0 else np.uint8

            # --- CHUNKED LAZY LOADING ---
            # Instead of loading the whole image in one go, create a dask array
            # from a grid of lazy-loaded chunks. This is crucial for WSI.
            dask_chunk_size = (*self.chunk_size, shape[2])

            # Create a function that reads a specific block of the image
            def get_chunk(block_info=None):
                loc = block_info[0]["array-location"]
                y_start, y_end = loc[0]
                x_start, x_end = loc[1]
                # slideio uses (x, y, w, h) for rect
                rect = (x_start, y_start, x_end - x_start, y_end - y_start)
                return scene.read_block(rect=rect)

            # Use dask.array.map_blocks to create the chunked array.
            # We start with a dummy array of zeros to define the shape and chunking,
            # then map our reading function over it. The channel dimension is always fully loaded.
            dummy_array = da.zeros(shape, chunks=dask_chunk_size, dtype=dtype)
            data = dummy_array.map_blocks(get_chunk, dtype=dtype)

            # Build coordinates from resolution metadata
            coords = {}
            resolution = scene.resolution
            if resolution[0] > 0:
                # Note: slideio resolution is in meters per pixel
                coords["X"] = np.arange(shape[1]) * resolution[0]
            if resolution[1] > 0:
                coords["Y"] = np.arange(shape[0]) * resolution[1]

            attrs = {"name": scene.name, "magnification": scene.magnification}  # noqa
            scene_id = generate_scene_id(self.path, i)
            attrs["unique_id"] = scene_id
            yield xr.DataArray(data, dims=dims, coords=coords, attrs=attrs)


READER_REGISTRY[".vsi"] = VSIReader


class IMSReader(BaseReader):
    """Reader for Bitplane Imaris IMS files.

    This reader uses the `imaris-ims-file-reader` library to parse `.ims`
    files, which are based on the HDF5 format. An IMS file is treated as a
    single, multi-dimensional scene.

    Parameters
    ----------
    path : Path
        The file path to the `.ims` file.

    Attributes
    ----------
    ims : imaris_ims_file_reader.ims
        The underlying IMS file reader object.

    Raises
    ------
    ImportError
        If the `imaris-ims-file-reader` library is not installed.
    """

    def __init__(self, path: Path):
        super().__init__(path)
        if ims_reader is None:
            raise ImportError(
                "imaris-ims-file-reader is not installed. "
                "Please run 'pip install imaris-ims-file-reader'"
            )
        # The library itself handles lazy loading via dask
        self.ims = ims_reader(path)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, type, value, traceback):
        # The underlying h5py file is managed by the ims object
        pass

    def __len__(self) -> int:
        """Return the number of scenes. IMS files are considered single-scene."""
        return 1

    def iter_scenes(self) -> Generator[xr.DataArray, None, None]:
        """Yield the entire IMS file as a single, lazy xarray DataArray.

        This method wraps the Dask array provided by `imaris-ims-file-reader`
        into an `xarray.DataArray`. It extracts dimensional information and
        metadata to construct the coordinates and attributes.

        Yields
        ------
        xr.DataArray
            A lazy-loaded DataArray representing the entire IMS dataset.
        """
        # The ims object provides a dask array directly. We work with the highest resolution (level 0).
        # The shape from the reader is (t, c, z, y, x)
        data = self.ims.get_dask_array(0)
        dims = ["time", "C", "Z", "Y", "X"]

        # Extract metadata and coordinates
        attrs = self.ims.get_parameters_dict()
        coords = {}

        # Time coordinates
        if "time" in dims and self.ims.TimePoints > 1:
            # Assuming time interval is stored in seconds, but this may need verification
            time_interval = float(attrs.get("TimeInterval", 1.0))
            coords["time"] = [
                np.timedelta64(int(s * 1e9), "ns")
                for s in (np.arange(self.ims.TimePoints) * time_interval)
            ]

        # Channel coordinates
        if "ChannelNames" in attrs:
            coords["C"] = attrs["ChannelNames"]

        # Spatial coordinates (X, Y, Z)
        for i, dim in enumerate(["X", "Y", "Z"]):
            voxel_size = self.ims.get_voxel_size(i)
            coords[dim] = np.arange(data.shape[-(i + 1)]) * voxel_size

        scene_id = generate_scene_id(self.path, 0)
        attrs["unique_id"] = scene_id
        yield xr.DataArray(data, dims=dims, coords=coords, attrs=attrs)


READER_REGISTRY[".ims"] = IMSReader


class Reader:
    """
    A factory that provides the correct reader for a given file path.

    This class uses the `__new__` method to act as a factory. When an
    instance of `Reader` is created, it inspects the file extension and
    returns an instance of the corresponding specialized `BaseReader` subclass
    (e.g., `LSMReader`, `CZIReader`).

    Parameters
    ----------
    path : Path
        The path to the image file to be opened.
    *args
        Variable length argument list passed to the reader's constructor.
    **kwargs
        Arbitrary keyword arguments passed to the reader's constructor.

    Returns
    -------
    BaseReader
        An instance of a `BaseReader` subclass suitable for the given file type.

    Raises
    ------
    ValueError
        If the file extension is not associated with any known reader.
    """

    def __new__(cls, path: Path, *args, **kwargs) -> BaseReader:
        ext = Path(path).suffix.lower()
        reader_class = READER_REGISTRY.get(ext)
        if not reader_class:
            raise ValueError(f"Unsupported file extension: {ext}")
        # Return an instance of the specialized reader class.
        return reader_class(path, *args, **kwargs)


class MetadataParser:
    """A callable class to parse metadata from all scenes in an image file.

    This class uses the `Reader` factory to open an image file and iterates
    through all its scenes. For each scene, it extracts metadata and combines
    it with any extra metadata provided during initialization.

    It is designed to be used as the `multi_file_parser` for the
    `ExperimentManager.scan_data_directory` method.

    Parameters
    ----------
    extra_metadata : Dict[str, Any], optional
        A dictionary of metadata to be merged with the metadata extracted
        from each scene. This can be used to add experiment-wide information.
        Defaults to None.

    """

    def __init__(self, extra_metadata: Dict[str, Any] = None):
        self.extra_metadata = extra_metadata

    def __call__(self, path: Path) -> List[Dict[str, Any]]:
        """Parse the file and return a list of data for each sub-item (scene).

        Parameters
        ----------
        path : Path
            The path to the image file to parse.

        Returns
        -------
        List[Dict[str, Any]]
            A list of dictionaries, where each dictionary represents a scene.
            Each dictionary contains an 'internal_index' (the scene number as a
            string) and a 'metadata' dictionary. This will be renamed to 'scene'.

        """
        items = []
        with Reader(path) as img:
            for k, scene in enumerate(img):
                # scene is now an xarray.DataArray, its metadata is in .attrs
                # Combine scene-specific metadata with any extra metadata
                combined_metadata = {**(self.extra_metadata or {}), **scene.attrs}
                items.append(
                    {
                        "scene": str(k),
                        "metadata": combined_metadata,
                    }
                )
        return items
