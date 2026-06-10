import csv
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Generator, Iterator, Self

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

    def __enter__(self) -> "BaseWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Close the writer and release resources."""
        pass

    @abstractmethod
    def write(self, data: Any, **kwargs):
        """Write data to the file.

        Parameters
        ----------
        data : Any
            The data to be written (e.g., xarray.DataArray, numpy.ndarray).
        **kwargs : dict
            Additional writer-specific arguments.

        """
        pass


WRITER_REGISTRY = {}


class TIFFWriter(BaseWriter):
    """A writer for TIFF files using tifffile."""

    def write(self, data: Any, **kwargs):
        """Write an image to a TIFF file.

        If data is an xarray.DataArray, it uses its values and metadata.
        """
        if isinstance(data, xr.DataArray):
            # Ensure directory exists
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(str(self.path), data.values, **kwargs)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(str(self.path), data, **kwargs)


WRITER_REGISTRY[".tif"] = TIFFWriter
WRITER_REGISTRY[".tiff"] = TIFFWriter


class CSVWriter(BaseWriter):
    """A writer for CSV files."""

    def write(self, data: Any, **kwargs):
        """Write tabular data to a CSV file.

        Supports pandas.DataFrame or a list of dictionaries.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # Handle pandas DataFrame or similar objects with to_csv method
        if hasattr(data, "to_csv"):
            index = kwargs.pop("index", False)
            data.to_csv(self.path, index=index, **kwargs)
        elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            with open(self.path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=data[0].keys(), **kwargs)
                writer.writeheader()
                writer.writerows(data)
        else:
            raise ValueError(f"Unsupported data type for CSVWriter: {type(data)}")


WRITER_REGISTRY[".csv"] = CSVWriter


class YOLOWriter(BaseWriter):
    """A writer for YOLO segmentation and detection format (.txt)."""

    def write(self, data: Any, **kwargs):
        """Write contours to a text file in YOLO format.

        Parameters
        ----------
        data : List[List[Union[int, float]]]
            A list where each entry is [class_id, x1, y1, x2, y2, ...].
            Coordinates should typically be normalized (0-1).
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            for row in data:
                f.write(" ".join(map(str, row)) + "\n")


WRITER_REGISTRY[".txt"] = YOLOWriter


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
        path = Path(path)
        ext = path.suffix.lower()
        writer_class = WRITER_REGISTRY.get(ext)
        if not writer_class:
            raise ValueError(f"Unsupported file extension: {ext}")
        # Return an instance of the specialized writer class.
        return writer_class(path, *args, **kwargs)
