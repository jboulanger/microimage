"""Data models for processing results."""

from dataclasses import dataclass
from typing import Any
import numpy as np
import xarray as xr
from .utils import mask_to_yolo_contours


@dataclass
class Result:
    """Base class for all processing results."""

    data: Any
    logical_type: str = "data"
    ext: str = ".dat"


class Mask(Result):
    """Represents a binary or label mask (usually saved as .tif)."""

    def __init__(self, data: Any):
        super().__init__(data, "mask", ".tif")


class Table(Result):
    """Represents tabular data (usually saved as .csv)."""

    ext: str = ".tif"

    def __init__(self, data: Any):
        super().__init__(data, "table", ".csv")


class Contours(Result):
    """Represents object contours (usually saved as YOLO .txt)."""

    def __init__(self, data: Any, class_index: int = 0):
        super().__init__(data, "contours", ".txt")
        self.class_index = class_index
        if isinstance(data, (xr.DataArray, np.ndarray)):
            self.data = mask_to_yolo_contours(self.data)


class Thumbnail(Result):
    """Represents a visualization or thumbnail (usually saved as .jpg)."""

    def __init__(self, data: Any):
        super().__init__(data, "thumbnail", ".jpg")
