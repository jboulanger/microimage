from abc import ABC, abstractmethod
from typing import Dict, Any, List, Generator, Self, Iterator
from pathlib import Path
import tifffile
import ome_types
import numpy as np
import xarray as xr

try:
    import pylibczirw
except ImportError:
    pylibczirw = None


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
        return self.scenes()

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of scenes in the file.

        Returns
        -------
        int
            The total number of scenes.

        """
        pass

    @abstractmethod
    def scenes(self) -> Generator[xr.DataArray, None, None]:
        """Yield scenes from the image file as DataArrays.

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

    def scenes(self, axes="TCZYX") -> Generator[xr.DataArray, None, None]:
        """Yield scenes from the LSM file as xarray DataArrays.

        If the file contains multiple scenes (a 'P' axis), this generator
        iterates through each scene. Otherwise, it yields the single primary
        image. It attempts to parse dimensional coordinates from the LSM metadata.

        Parameters
        ----------
        axes : str, optional
            A string representing the desired order of axes, by default "TCZYX".
            This parameter is currently used for coordinate mapping.

        Yields
        ------
        xr.DataArray
            A DataArray for each scene, with dimensions, coordinates, and metadata.

        """
        try:
            coords = {
                "T": [
                    t * self.metadata["TimeIntervall"]
                    for t in range(self.metadata["DimensionTime"])
                ],
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
        except Exception:
            coords = {}
        p_index = self.axes_str.find("P")
        if p_index >= 0:
            num_scenes = self.tif.series[0].shape[p_index]
            for k in range(num_scenes):
                data = self.tif.series[0].asarray()[k]
                print(data.shape)
                scene_coords = {a: coords[a] for a in self.axes_str if a != "P"}
                print("scene coords", scene_coords)
                yield xr.DataArray(data, coords=scene_coords, attrs=self.metadata)
        else:
            data = self.tif.asarray()
            scene_coords = {a: coords[a] for a in self.axes_str if a != "P"}
            yield xr.DataArray(data, coords=scene_coords, attrs=self.metadata)


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

    def scenes(self) -> Generator[xr.DataArray, None, None]:
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
            data = series.asarray()
            dims = list(series.axes)
            coords = {}
            attrs = {}
            # Try to get coordinates and metadata from OME structure
            if self.ome and i < len(self.ome.images):
                pixels = self.ome.images[i].pixels
                attrs = {"name": self.ome.images[i].name}
                try:
                    coords = {
                        "T": [t * pixels.physical_size_t for t in range(pixels.size_t)]
                        if pixels.physical_size_t
                        else None,
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
                            c.name or str(ch_idx)
                            for ch_idx, c in enumerate(pixels.channels)
                        ],
                    }
                    # Filter out empty or mismatched coords
                    scene_coords = {
                        d: c
                        for d, c in coords.items()
                        if d in dims
                        and c is not None
                        and len(c) == data.shape[dims.index(d)]
                    }
                    coords = scene_coords
                except Exception:
                    coords = {}  # Fallback on parsing error

            yield xr.DataArray(data, dims=dims, coords=coords, attrs=attrs)


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
        if pylibczirw is None:
            raise ImportError(
                "pylibczirw is not installed. Please run 'pip install pylibczirw'"
            )
        self.czi = pylibczirw.CziFile(path)
        self.metadata = self.czi.metadata

    def __enter__(self) -> Self:
        return self

    def __exit__(self, type, value, traceback):
        self.czi.close()

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

    def scenes(self) -> Generator[xr.DataArray, None, None]:
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
        num_scenes = len(self)
        dims_shape = self.czi.get_dims_shape()[0]

        for scene_index in range(num_scenes):
            # Define the dimensions for reading this specific scene
            selector = {"S": scene_index}

            # Read the image data for the scene. The result is a numpy array.
            # The dimensions are ordered according to the file's dimension string.
            data, read_dims = self.czi.read_image(**selector)

            # Squeeze out the scene dimension from the data and the dimension string
            data = data.squeeze()
            dims = [d for d in read_dims if d != "S"]

            # --- Build Coordinates ---
            coords = {}
            # Get scale and unit for each dimension
            for dim in dims:
                if dim in dims_shape:
                    size = dims_shape[dim][1]
                    scale = self.czi.get_dimension_pixel_size(dim)
                    if scale:
                        coords[dim] = np.arange(size) * scale

            # Special handling for channel names
            if "C" in dims and self.czi.channel_names:
                coords["C"] = self.czi.channel_names

            # Create and yield the DataArray
            yield xr.DataArray(data, dims=dims, coords=coords, attrs=self.metadata)


READER_REGISTRY[".czi"] = CZIReader


class ND2Reader(BaseReader):
    """Reader for ND2 files from Nikon Elements.

    .. warning::
        This reader is a placeholder and is not yet implemented.

    Parameters
    ----------
    path : Path
        The file path to the `.nd2` file.
    """

    def __init__(self, path: Path):
        super().__init__(path)
        pass

    def __enter__(self) -> Self:
        return self

    def __exit__(self, type, value, traceback):
        pass

    def __len__(self) -> int:
        return 0

    def scenes(self):
        pass


READER_REGISTRY[".nd2"] = ND2Reader


class LIFReader(BaseReader):
    """Reader for LIF (Leica Image File) format from Leica LAS X.

    .. warning::
        This reader is a placeholder and is not yet implemented.

    Parameters
    ----------
    path : Path
        The file path to the `.lif` file.
    """

    def __init__(self, path: Path):
        super().__init__(path)
        pass

    def __enter__(self) -> Self:
        return self

    def __exit__(self, type, value, traceback):
        pass

    def __len__(self) -> int:
        return 0

    def scenes(self):
        pass


READER_REGISTRY[".lif"] = LIFReader


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
            string) and a 'metadata' dictionary.

        """
        items = []
        with Reader(path) as img:
            for k, scene in enumerate(img):
                # scene is now an xarray.DataArray, its metadata is in .attrs
                # Combine scene-specific metadata with any extra metadata
                combined_metadata = {**(self.extra_metadata or {}), **scene.attrs}
                items.append({"internal_index": str(k), "metadata": combined_metadata})
        return items


# option 1
if __name__ == "__main__":
    filename = "test.lsm"
    with Reader(filename) as file:
        for img in file:
            print(img.shape)

# option 2
if __name__ == "__main__":
    filename = "test.lsm"
    with Reader(filename) as file:
        for scene in file:
            print(scene.metadata)
