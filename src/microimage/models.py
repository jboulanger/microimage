import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np


# Type hint for the new parser function: takes a Path and returns a list of sub-item
# data dictionaries.
# Each dictionary MUST contain at least 'internal_index' and 'metadata'.
MultiFileParser = Callable[
    [Path], List[Dict[str, Any]]
]  # Each dict should contain 'scene' and 'metadata'


@dataclass
class DataItem:
    """
    Represents a single processable item, such as an image or a scene.

    This dataclass holds all information related to a single item, including its
    source file, metadata, processing status, and any generated results. It is
    designed to be a self-contained unit that can be serialized to and from
    the database.

    Parameters
    ----------
    relative_path : Path
        The path to the container file, relative to the `ExperimentManager`'s
        `data_dir`.
    scene : str, optional
        A unique identifier for the item within its container file (e.g., a
        scene number). Defaults to "".

    Attributes
    ----------
    metadata : Dict[str, Any]
        A dictionary for storing arbitrary metadata about the item.
    result_files : Dict[str, str]
        A dictionary mapping a result type (str) to its relative path (str).
    processed : bool
        A flag indicating whether the item has been successfully processed.
    error : Optional[str]
        A string containing an error message if processing failed.
    id : str
        The unique identifier for the item, derived from the container name
        and internal index (e.g., "image.czi:0").
    """

    # Required field, points to the container file
    relative_path: Path

    # The unique, hash-based identifier for this item.
    id: str

    # Fields with default values
    scene: str = field(default="")  # The original scene index (e.g., "0", "1")
    metadata: Dict[str, Any] = field(default_factory=dict)  # noqa
    # Result paths are stored relative to ExperimentManager's results_path.
    result_files: Dict[str, Dict[str, str]] = field(default_factory=dict)
    processed: bool = field(default=False)

    # NEW: Fields for basic dimensional metadata
    size_t: Optional[int] = field(default=None)
    size_c: Optional[int] = field(default=None)
    size_z: Optional[int] = field(default=None)
    size_y: Optional[int] = field(default=None)
    size_x: Optional[int] = field(default=None)
    scale_t: Optional[float] = field(default=None)
    scale_z: Optional[float] = field(default=None)
    scale_y: Optional[float] = field(default=None)
    scale_x: Optional[float] = field(default=None)
    channels: Optional[List[str]] = field(default=None)

    # Annotated metadata from 'annotations' table
    annotated_metadata: Optional[Dict] = field(default=None)
    annotated_channels: Optional[Dict] = field(default=None)

    error: Optional[str] = field(default=None)

    def add_metadata(self, metadata_dict: Dict[str, Any]):
        """Add or update key-value pairs in the item's metadata.

        Parameters
        ----------
        metadata_dict : Dict[str, Any]
            A dictionary of metadata to merge into the existing metadata.

        """
        self.metadata.update(metadata_dict)

    def add_result_file(self, result_name: str, relative_path: str, logical_type: str):
        """Register a new result file for this item.

        Parameters
        ----------
        result_name : str
            The unique name/key for this result.
        relative_path : str
            The path to the result file, relative to the project root.
        logical_type : str
            The category of the result (e.g., 'mask', 'table').
        """
        self.result_files[result_name] = {"path": relative_path, "type": logical_type}

    def get_absolute_container_path(self, base_data_dir: Path) -> Path:
        """Return the absolute path to the item's container file.

        Parameters
        ----------
        base_data_dir : Path
            The absolute path to the main data directory, which serves as the
            base for the item's `relative_path`.

        Returns
        -------
        Path
            The resolved, absolute path to the container file.
        """
        return (base_data_dir / self.relative_path).resolve()

    def get_full_id(self) -> str:
        """Return the unique ID for this DataItem.

        Returns
        -------
        str
            The unique identifier string.
        """
        return self.id

    def to_data_row(self) -> Dict[str, Any]:
        """Convert the DataItem object into a dictionary.

        Converts the DataItem object into a dictionary suitable for DB insertion/update
        for the 'data' table (excluding result data).

        Returns
        -------
        Dict[str, Any]
            A dictionary representation of the item for the 'data' table.
        """
        row = {
            "id": self.id,
            "relative_path": str(self.relative_path),
            "scene": self.scene,
            "metadata": json.dumps(self.metadata),
            "processed": int(self.processed),
            "error": self.error,
            "size_t": self.size_t,
            "size_c": self.size_c,
            "size_z": self.size_z,
            "size_y": self.size_y,
            "size_x": self.size_x,
            "scale_t": self.scale_t,
            "scale_z": self.scale_z,
            "scale_y": self.scale_y,
            "scale_x": self.scale_x,
            "channels": json.dumps(self.channels)
            if self.channels is not None
            else None,
            "annotated_channels": json.dumps(self.annotated_channels)
            if self.annotated_channels is not None
            else None,
        }
        if self.annotated_metadata:
            row.update(self.annotated_metadata)
        return row

    @classmethod
    def from_db_row(
        cls, data: Dict[str, Any], result_files: Optional[Dict[str, str]] = None
    ) -> "DataItem":
        """Create a DataItem instance from a database row.

        Parameters
        ----------
        data : Dict[str, Any]
            A dictionary representing a row from the 'data' table.
        result_files : Optional[Dict[str, str]], optional
            A dictionary of result files for this item, by default None.

        Returns
        -------
        DataItem
            A fully hydrated DataItem instance.
        """
        relative_path_str = data.get("relative_path")
        if relative_path_str is None:
            raise ValueError("Database row missing 'relative_path' key.")

        item_id = data.get("id")
        if not item_id:
            raise ValueError("Database row missing 'id' key.")

        scene = data.get("scene", "")
        item = cls(id=item_id, relative_path=Path(relative_path_str), scene=scene)

        # Hydrate the rest of the fields
        item.metadata = json.loads(data.get("metadata", "{}") or "{}")
        item.processed = bool(data.get("processed", 0))
        item.error = data.get("error", None)

        # Hydrate dimensional metadata
        item.size_t = data.get("size_t")
        item.size_c = data.get("size_c")
        item.size_z = data.get("size_z")
        item.size_y = data.get("size_y")
        item.size_x = data.get("size_x")
        item.scale_t = data.get("scale_t")
        item.scale_z = data.get("scale_z")
        item.scale_y = data.get("scale_y")
        item.scale_x = data.get("scale_x")

        # Hydrate channels from JSON
        channels_json = data.get("channels")
        if channels_json:
            item.channels = json.loads(channels_json)

        # Hydrate annotations from JOIN
        ann_metadata_json = data.get("annotated_metadata")
        if ann_metadata_json:
            item.annotated_metadata = json.loads(ann_metadata_json)

        ann_channels_json = data.get("annotated_channels")
        if ann_channels_json:
            item.annotated_channels = json.loads(ann_channels_json)

        if result_files:
            item.result_files = result_files

        return item
