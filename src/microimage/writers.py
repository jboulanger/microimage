from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generator, Iterator, Self

import tifffile
import xarray as xr


class BaseWriter(ABC):
    """Abstract base class for all image file writer.

    This class defines the common interface that all concrete writers
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
        return self.iter_scenes()

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


WRITER_REGISTRY = {}


class TIFFWriter(BaseWriter):
    def __init__(self, path: Path):
        super().__init__(path)
        self.tif = tifffile.TiffFile(path)


class Writer:
    """
    A factory that provides the correct writer

    This class uses the `__new__` method to act as a factory. When an
    instance of `Writer` is created, it inspects the file extension and
    returns an instance of the corresponding specialized `BaseWriter` subclass.

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

    Example
    -------

    >>> from microimage import Writer
    >>> with Reader("image.tif) as f:
    >>>     f.write_scene()
    """

    def __new__(cls, path: Path, *args, **kwargs) -> BaseWriter:
        ext = Path(path).suffix.lower()
        writer_class = WRITER_REGISTRY.get(ext)
        if not writer_class:
            raise ValueError(f"Unsupported file extension: {ext}")
        # Return an instance of the specialized writer class.
        return writer_class(path, *args, **kwargs)
