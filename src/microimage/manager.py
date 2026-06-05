"""Manager from microscopy experiments."""

import json
import os
import sqlite3
from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import xarray as xr
import yaml

from .readers import Reader


# Custom connection factory to return dictionary rows instead of tuples
def dict_factory(cursor, row):
    """Convert SQLite tuple row results into a dictionary.

    This function is used as a `row_factory` for an sqlite3 connection
    to make query results more intuitive to access by column name.

    Parameters
    ----------
    cursor : sqlite3.Cursor
        The cursor object from which the row was fetched.
    row : tuple
        The raw tuple row fetched from the database.

    Returns
    -------
    dict
        Dictionnary with content of the row

    """
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d


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
    # This remains a Dict[type: path] in memory, but is stored in a separate DB table ('results').
    # NOTE: Result paths are stored relative to ExperimentManager's project_root (DB parent).
    result_files: Dict[str, str] = field(default_factory=dict)
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
    condition: Optional[str] = field(default=None)
    annotated_channels: Optional[List[str]] = field(default=None)

    error: Optional[str] = field(default=None)

    def add_metadata(self, metadata_dict: Dict[str, Any]):
        """Add or update key-value pairs in the item's metadata.

        Parameters
        ----------
        metadata_dict : Dict[str, Any]
            A dictionary of metadata to merge into the existing metadata.

        """
        self.metadata.update(metadata_dict)

    def add_result_file(self, result_type: str, relative_path: str):
        """Register a new result file for this item.

        Parameters
        ----------
        result_type : str
            A string identifying the type of result (e.g., "thumbnail", "mask").
        relative_path : str
            The path to the result file, relative to the project root.

        """
        self.result_files[result_type] = relative_path

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
        return {
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
            "condition": self.condition,
            "annotated_channels": json.dumps(self.annotated_channels)
            if self.annotated_channels is not None
            else None,
        }

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
        item.condition = data.get("condition")
        ann_channels_json = data.get("annotated_channels")
        if ann_channels_json:
            item.annotated_channels = json.loads(ann_channels_json)

        if result_files:
            item.result_files = result_files

        return item


class ExperimentManager:
    """
    Manages a collection of DataItems using a persistent SQLite database.

    This class is the main entry point for interacting with a dataset. It handles
    directory scanning, database initialization, and the creation, retrieval,
    and updating of `DataItem` objects. It acts as a container, allowing
    dictionary-style access and iteration over the items.

    Parameters
    ----------
    data_dir : Union[str, Path]
        The path to the directory containing the raw data files. This directory
        must exist and be readable.
    results_dir : Union[str, Path]
        The path to the directory where output files will be saved. It will be
        created if it doesn't exist and must be writable.
    db_path : Union[str, Path], optional
        The path to the SQLite database file. Defaults to "experiment.db" in
        the current working directory.

    Raises
    ------
    FileNotFoundError
        If the `data_dir` does not exist.
    PermissionError
        If the `data_dir` is not readable or `results_dir` is not writable.

    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        results_dir: Union[str, Path],
        db_path: Union[str, Path] = "experiment.db",
    ):
        # Store paths
        self.db_path: Path = Path(db_path).resolve()

        # Define the project root as the directory containing the DB file.
        # This acts as the common base for storing relative paths to results.
        self.project_root: Path = self.db_path.parent
        print(f"Project root (Base for result paths) set to: {self.project_root}")

        # --- Enforce Data Directory constraints (Must Exist and be Readable) ---
        self.data_dir: Path = Path(data_dir).resolve()

        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Data directory does not exist: {self.data_dir}. It must be pre-configured and accessible."
            )

        # Check read permissions (R_OK)
        if not os.access(self.data_dir, os.R_OK):
            raise PermissionError(
                f"Data directory is not readable: {self.data_dir}. Check file system permissions."
            )

        # --- Results Directory (Must be Writable) ---
        self.results_dir: Path = Path(results_dir).resolve()

        # Ensure results directory exists and is writable
        self.results_dir.mkdir(parents=True, exist_ok=True)
        if not os.access(self.results_dir, os.W_OK):
            raise PermissionError(
                f"Results directory is not writable: {self.results_dir}. Cannot save output files."
            )

        self._initialize_db()

    def _get_connection(self):
        """Return a new database connection with a dict row factory.

        Returns
        -------
        sqlite3.Connection
            A new connection to the SQLite database.

        """
        conn = sqlite3.connect(
            self.db_path, timeout=10.0
        )  # Set a high timeout for concurrency
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.row_factory = dict_factory
        return conn

    def _initialize_db(self):
        """Create the database tables if they do not already exist."""
        print(f"Initializing database at {self.db_path}...")
        conn = self._get_connection()
        try:
            # Enable WAL mode for better concurrency
            conn.execute("PRAGMA journal_mode=WAL;")

            # Table for main data
            conn.execute("""
                CREATE TABLE IF NOT EXISTS data (
                    id TEXT PRIMARY KEY,
                    relative_path TEXT NOT NULL,
                    scene TEXT NOT NULL,
                    metadata TEXT,
                    processed INTEGER NOT NULL,
                    error TEXT,
                    size_t INTEGER,
                    size_c INTEGER,
                    size_z INTEGER,
                    size_y INTEGER,
                    size_x INTEGER,
                    scale_t REAL,
                    scale_z REAL,
                    scale_y REAL,
                    scale_x REAL,
                    channels TEXT
                )
            """)

            # Table for one-to-many results relationship
            conn.execute("""
                CREATE TABLE IF NOT EXISTS results (
                    data_id TEXT NOT NULL,
                    result_type TEXT NOT NULL,
                    relative_path TEXT NOT NULL,
                    PRIMARY KEY (data_id, result_type),
                    FOREIGN KEY (data_id) REFERENCES data(id) ON DELETE CASCADE
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS annotations (
                    data_id TEXT NOT NULL,
                    channels TEXT NOT NULL,
                    condition TEXT NOT NULL,
                    PRIMARY KEY (data_id),
                    FOREIGN KEY (data_id) REFERENCES data(id) ON DELETE CASCADE
                )
            """)
            conn.commit()
            print("Database structure verified/created.")
        except sqlite3.Error as e:
            print(f"Database Initialization Error: {e}")
        finally:
            conn.close()

    def _fetch_result_files(
        self, conn: sqlite3.Connection, item_id: str
    ) -> Dict[str, str]:
        """Fetch all result files for a given item ID.

        Parameters
        ----------
        conn : sqlite3.Connection
            An active database connection.
        item_id : str
            The unique ID of the item.

        Returns
        -------
        Dict[str, str]
            A dictionary mapping result type to relative file path.
        """
        cursor = conn.execute(
            "SELECT result_type, relative_path FROM results WHERE data_id = ?",
            (item_id,),
        )
        # Convert list of rows (type, path) to a dictionary {type: path}
        return {row["result_type"]: row["relative_path"] for row in cursor.fetchall()}

    def _fetch_item_with_results(
        self, conn: sqlite3.Connection, item_id: str
    ) -> Optional["DataItem"]:
        """Fetch a single item and its results from the database.

        Parameters
        ----------
        conn : sqlite3.Connection
            An active database connection.
        item_id : str
            The unique ID of the item to fetch.

        Returns
        -------
        Optional[DataItem]
            A `DataItem` instance if found, otherwise None.
        """
        cursor = conn.execute(
            """
            SELECT d.*, a.condition, a.channels AS annotated_channels
            FROM data d
            LEFT JOIN annotations a ON d.id = a.data_id
            WHERE d.id = ?
        """,
            (item_id,),
        )
        item_row = cursor.fetchone()

        if not item_row:
            return None

        result_files = self._fetch_result_files(conn, item_id)

        return DataItem.from_db_row(item_row, result_files)

    # --- Python Container Protocol Methods ---

    def __getitem__(self, item_id: str) -> "DataItem":
        """Retrieve a single DataItem by its unique ID.

        Parameters
        ----------
        item_id : str
            The unique ID of the item (e.g., "image.czi:0").

        Returns
        -------
        DataItem
            The requested DataItem instance.

        Raises
        ------
        KeyError
            If no item with the given ID is found.
        """
        conn = self._get_connection()
        try:
            item = self._fetch_item_with_results(conn, item_id)
            if item:
                return item
            raise KeyError(f"DataItem with ID '{item_id}' not found.")
        except sqlite3.Error as e:
            # Re-raise as a standard exception if it's a DB error, not a missing key
            raise RuntimeError(f"Database access error while fetching item: {e}")
        finally:
            conn.close()

    def __len__(self) -> int:
        """Return the total number of items in the database.

        Returns
        -------
        int
            The total count of items.

        """
        conn = self._get_connection()
        try:
            count = conn.execute("SELECT COUNT(id) FROM data").fetchone()["COUNT(id)"]
            return count
        except sqlite3.Error as e:
            print(f"Database Count Error: {e}")
            return 0
        finally:
            conn.close()

    def __iter__(self) -> Iterator["DataItem"]:
        """Return an iterator over all DataItem objects in the database.

        Yields
        ------
        Iterator[DataItem]
            An iterator that yields each `DataItem` in the collection.

        """
        return iter(self.get_all_items())

    # --- Utility Methods ---

    def extract_dims_from_xarray(self, data: xr.DataArray) -> Dict[str, Any]:
        """Extract size and scale information from an xarray.DataArray.

        Parameters
        ----------
        data : xr.DataArray
            The DataArray from which to extract metadata.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing size and scale values for dimensions.
        """
        dims_meta = {}
        # Map xarray dimension names to our database field suffixes
        dim_map = {
            "T": "t",
            "time": "t",
            "C": "c",
            "Z": "z",
            "Y": "y",
            "X": "x",
        }

        for da_dim, db_suffix in dim_map.items():
            # Store size if the dimension exists
            if da_dim in data.sizes:
                dims_meta[f"size_{db_suffix}"] = data.sizes[da_dim]

            # Calculate and store scale if coordinates are available and numeric
            if da_dim in data.coords:
                coords = data.coords[da_dim].values
                if coords.ndim == 1 and len(coords) > 1:
                    # Use the difference between the first two points as the scale
                    try:
                        # Handle numpy datetime/timedelta by converting to seconds
                        if np.issubdtype(coords.dtype, np.datetime64) or np.issubdtype(
                            coords.dtype, np.timedelta64
                        ):
                            scale = (coords[1] - coords[0]) / np.timedelta64(1, "s")
                        else:
                            scale = abs(float(coords[1] - coords[0]))
                        dims_meta[f"scale_{db_suffix}"] = scale
                    except (TypeError, IndexError):
                        continue  # Skip if coordinates are not numeric or too short

        # Extract channel names if they exist
        if "C" in data.coords:
            dims_meta["channels"] = [str(c) for c in data.coords["C"].values]

        return dims_meta

    def add_file(self, file_path: Path | str, metadata: dict[Any]):
        """Add a single image container.

        Note the file as to in the project folder... is this good?

        Parameters
        ----------
        file_path: str | Path
            Path to the
        metadata: dictionnary
            Additional metadata as a dictionnary

        """
        pass

    def scan_data_directory(
        self,
        extensions: List[str] = [".png", ".jpg", ".jpeg", ".tif"],
        multi_file_parser: MultiFileParser = None,
    ):
        """Scan the data directory for new files and register them.

        This method walks through the `data_dir`, identifies files with the
        specified extensions, and adds them to the database if they are not
        already present.

        Parameters
        ----------
        extensions : List[str], optional
            A list of file extensions to look for, by default [".lsm", ".nd2", ".czi", ".lif"].
        multi_file_parser : MultiFileParser, optional
            A callable that takes a file `Path` and returns a list of dicts,
            one for each sub-item (e.g., scene) in the file. If not provided,
            each file is treated as a single item.

        """
        print(f"Scanning {self.data_dir} for new files...")
        conn = self._get_connection()
        found_new_files = 0

        try:
            # Find all potential container files
            glob_iterators = (self.data_dir.glob(f"*{ext}") for ext in extensions)
            all_container_paths = chain.from_iterable(glob_iterators)

            # 2. Iterate through container files
            for container_path in all_container_paths:
                # Skip hidden resource files (e.g., Mac OS ._ files)
                if container_path.name.startswith("._"):
                    print(f"  Skipping hidden resource file: {container_path.name}")
                    continue

                # Delegate processing of each file to a helper method
                new_items_count = self._process_container_file(
                    container_path, conn, multi_file_parser
                )
                found_new_files += new_items_count

            conn.commit()
            if found_new_files > 0:
                print(f"Successfully registered {found_new_files} new items/sub-items.")
            else:
                print("No new files or sub-items found.")

        except sqlite3.Error as e:
            print(f"Database Scan Error: {e}")
            conn.rollback()
        except Exception as e:
            # Catch errors originating from the parser function
            print(f"File Parser Error during scan: {e}")
            conn.rollback()
        finally:
            conn.close()

    def _create_item_in_db(self, conn: sqlite3.Connection, item: DataItem):
        """Inserts a single DataItem into the database."""
        row = item.to_data_row()
        conn.execute(
            """
                        INSERT INTO data (id, relative_path, scene, metadata, processed, error,
                                           size_t, size_c, size_z, size_y, size_x,
                                           scale_t, scale_z, scale_y, scale_x, channels)
                        VALUES (:id, :relative_path, :scene, :metadata, :processed, :error,
                                :size_t, :size_c, :size_z, :size_y, :size_x,
                                :scale_t, :scale_z, :scale_y, :scale_x,
                                :channels)""",
            row,
        )
        print(f"  Registered new item: {item.id}")

    def _process_container_file(
        self,
        container_path: Path,
        conn: sqlite3.Connection,
        multi_file_parser: Optional[MultiFileParser],
    ) -> int:
        """Processes a single container file, creating and adding items to the DB."""
        relative_path = container_path.relative_to(self.data_dir)
        items_data_list = []
        new_items_count = 0

        # Use the provided parser or the default Reader to get scene information
        if multi_file_parser:
            print(f"  Using parser to analyze file: {container_path.name}")
            items_data_list = multi_file_parser(container_path)
        else:
            try:
                with Reader(container_path) as reader:
                    for i, scene in enumerate(reader):
                        print(f"Adding {container_path} scene [{i}]")
                        items_data_list.append(
                            {
                                "scene": str(i),
                                "_scene_xarray": scene,
                            }
                        )
            except Exception as e:
                print(f"  Could not read {container_path.name}: {e}")
                return 0

        # Process each sub-item (scene) from the file
        for item_data in items_data_list:
            item_id = item_data["_scene_xarray"].attrs["unique_id"]

            if not item_id:
                print(f"  Skipping scene in {relative_path} due to missing unique_id.")
                continue

            # Check if item already exists
            if conn.execute("SELECT id FROM data WHERE id = ?", (item_id,)).fetchone():
                continue

            # Extract dimensional metadata
            scene_xarray = item_data["_scene_xarray"]

            if (
                scene_xarray is None
            ):  # Should not happen if ID was found, but for safety
                try:
                    with Reader(container_path) as reader:
                        scene_xarray = next(
                            (
                                s
                                for i, s in enumerate(reader)
                                if str(i) == item_data["scene"]
                            ),
                            None,
                        )
                except Exception:
                    scene_xarray = None

            # Create the DataItem instance now that we have all info
            item = DataItem(
                id=item_id,
                relative_path=relative_path,
                scene=item_data.get("scene", ""),
                # metadata=item_data.get("metadata", {}),
            )
            # add base metadata on image size
            if scene_xarray is not None:
                dims_meta = self.extract_dims_from_xarray(scene_xarray)
                for key, value in dims_meta.items():
                    setattr(item, key, value)

            # Add the fully formed item to the database
            self._create_item_in_db(conn, item)
            new_items_count += 1

        return new_items_count

    def update_item(self, item: DataItem):
        """Update an item's record in the database.

        This method synchronizes the state of a `DataItem` object with the
        database, updating its metadata, processed status, error message,
        and associated result files.

        Parameters
        ----------
        item : DataItem
            The `DataItem` object to update in the database.

        """
        conn = self._get_connection()
        try:
            row = item.to_data_row()
            # 1. Update the main item row
            conn.execute(
                """
                UPDATE data SET
                    scene = :scene,
                    metadata = :metadata,
                    processed = :processed,
                    error = :error,
                    size_t = :size_t, size_c = :size_c, size_z = :size_z, size_y = :size_y, size_x = :size_x,
                    scale_t = :scale_t, scale_z = :scale_z, scale_y = :scale_y, scale_x = :scale_x,
                    channels = :channels
                WHERE id = :id
            """,
                row,
            )

            # 2. Delete existing result files for this item
            conn.execute("DELETE FROM results WHERE data_id = ?", (item.id,))

            # 3. Insert the current set of result files
            if item.result_files:
                result_data = [
                    (item.id, result_type, relative_path)
                    for result_type, relative_path in item.result_files.items()
                ]
                conn.executemany(
                    """
                    INSERT INTO results (data_id, result_type, relative_path)
                    VALUES (?, ?, ?)
                """,
                    result_data,
                )

            # 4. Update annotations
            # Fetch existing annotations if any
            existing_annotation = conn.execute(
                "SELECT condition, channels FROM annotations WHERE data_id = ?",
                (item.id,),
            ).fetchone()

            current_condition = (
                existing_annotation["condition"] if existing_annotation else None
            )
            current_annotated_channels_json = (
                existing_annotation["channels"] if existing_annotation else None
            )
            current_annotated_channels = (
                json.loads(current_annotated_channels_json)
                if current_annotated_channels_json
                else None
            )

            # Determine the values to be used for the update
            # If item has a value, use it. Otherwise, retain the existing value.
            new_condition = (
                item.condition if item.condition is not None else current_condition
            )
            new_annotated_channels = (
                item.annotated_channels
                if item.annotated_channels is not None
                else current_annotated_channels
            )

            # Only proceed if there's something to insert/update for annotations
            if new_condition is not None or new_annotated_channels is not None:
                conn.execute(
                    """
                    INSERT INTO annotations (data_id, condition, channels)
                    VALUES (?, ?, ?)
                    ON CONFLICT(data_id) DO UPDATE SET
                        condition=excluded.condition,
                        channels=excluded.channels
                """,
                    (
                        item.id,
                        new_condition,
                        json.dumps(new_annotated_channels)
                        if new_annotated_channels is not None
                        else None,
                    ),
                )
            conn.commit()
        except sqlite3.Error as e:
            print(f"Database Update Error for {item.id}: {e}")
            conn.rollback()
        finally:
            conn.close()

    def get_absolute_container_path(self, item_id: str) -> Optional[Path]:
        """Get the absolute path to an item's container file.

        Parameters
        ----------
        item_id : str
            The unique ID of the item.

        Returns
        -------
        Optional[Path]
            The absolute `Path` to the container file, or `None` if the item
            is not found.

        """
        try:
            item = self.__getitem__(item_id)
            # FIX: Use self.data_dir as the base for resolving the relative path
            return item.get_absolute_container_path(self.data_dir)
        except KeyError:
            return None

    def get_scene(self, item_id: str) -> Optional[str]:
        """Get the scene identifier for an item.

        Parameters
        ----------
        item_id : str
            The unique ID of the item.

        Returns
        -------
        Optional[str]
            The scene identifier string, or `None` if the item is not found.
        """
        try:
            item = self.__getitem__(item_id)
            return item.scene
        except KeyError:
            return None

    def _bulk_fetch_items(self, query: str) -> List[DataItem]:
        """Fetch a list of items from the database using a custom query.

        This is an efficient helper method that fetches multiple items and all
        their associated result files in a minimal number of database queries.

        Parameters
        ----------
        query : str
            The SQL `SELECT` query to execute to fetch the items.

        Returns
        -------
        List[DataItem]
            A list of hydrated `DataItem` objects.
        """
        conn = self._get_connection()
        try:
            # 1. Fetch all matching item rows
            item_rows = conn.execute(query).fetchall()
            if not item_rows:
                return []

            item_ids = [row["id"] for row in item_rows]

            # 2. Fetch all result files for all fetched item IDs in a single query
            id_placeholders = ",".join(["?"] * len(item_ids))
            result_rows = conn.execute(
                f"""
                SELECT data_id, result_type, relative_path 
                FROM results 
                WHERE data_id IN ({id_placeholders})
            """,
                item_ids,
            ).fetchall()

            # 3. Map results to their item IDs
            all_results: Dict[str, Dict[str, str]] = {
                item_id: {} for item_id in item_ids
            }
            for row in result_rows:
                all_results[row["data_id"]][row["result_type"]] = row["relative_path"]

            # 4. Create and return DataItem objects
            return [
                DataItem.from_db_row(row, all_results[row["id"]]) for row in item_rows
            ]
        except sqlite3.Error as e:
            print(f"Database Query Error: {e}")
            return []
        finally:
            conn.close()

    def get_all_items(self) -> List[DataItem]:
        """Return a list of all DataItem objects from the database.

        Returns
        -------
        List[DataItem]
            A list containing every `DataItem` in the database.
        """
        return self._bulk_fetch_items(
            """
            SELECT d.*, a.condition, a.channels AS annotated_channels
            FROM data d
            LEFT JOIN annotations a ON d.id = a.data_id
        """
        )

    def get_unprocessed_items(self) -> List[DataItem]:
        """Return a list of all items not yet marked as processed.

        Returns
        -------
        List[DataItem]
            A list of all `DataItem` objects where `processed` is False.
        """
        # processed = 0 (False)
        return self._bulk_fetch_items(
            """
            SELECT d.*, a.condition, a.channels AS annotated_channels
            FROM data d
            LEFT JOIN annotations a ON d.id = a.data_id
            WHERE d.processed = 0
        """
        )

    def get_items_with_errors(self) -> List[DataItem]:
        """Return a list of all items that have a recorded error.

        Returns
        -------
        List[DataItem]
            A list of all `DataItem` objects where the `error` field is not NULL.
        """
        # error IS NOT NULL and processed = 0
        return self._bulk_fetch_items(
            """
            SELECT d.*, a.condition, a.channels AS annotated_channels
            FROM data d
            LEFT JOIN annotations a ON d.id = a.data_id
            WHERE d.error IS NOT NULL AND d.processed = 0
        """
        )

    def _update_item_status(self, where_clause: str = "1", message: str = ""):
        """Generic helper to reset the status of items based on a WHERE clause."""
        print(f"Updating item status in DB: {message}")
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                f"UPDATE data SET processed = 0, error = NULL WHERE {where_clause}"
            )
            conn.commit()
            print(f"Reset status for {cursor.rowcount} items.")
        except sqlite3.Error as e:
            print(f"Database Reset Error: {e}")
        finally:
            conn.close()

    def force_reprocess_all(self):
        """Reset the 'processed' and 'error' status for all items.

        This method marks all items in the database as unprocessed, allowing
        them to be picked up again by a processing pipeline.
        """
        self._update_item_status(message="Resetting all items to unprocessed.")

    def reset_errors(self):
        """Reset the status for all items that previously failed.

        This method finds all items with a non-null `error` field and resets
        their `processed` and `error` statuses, allowing a re-attempt at processing.
        """
        self._update_item_status(
            where_clause="error IS NOT NULL", message="Resetting items with errors."
        )

    def report_status(self):
        """Print a summary of the dataset's processing status."""
        total = len(self)
        conn = self._get_connection()
        try:
            processed_count = conn.execute(
                "SELECT COUNT(id) FROM data WHERE processed = 1"
            ).fetchone()["COUNT(id)"]
            error_count = conn.execute(
                "SELECT COUNT(id) FROM data WHERE error IS NOT NULL"
            ).fetchone()["COUNT(id)"]
        except sqlite3.Error as e:
            print(f"Database Report Error: {e}")
            processed_count, error_count = 0, 0
        finally:
            conn.close()

        unprocessed_count = total - processed_count - error_count

        print("\n--- Experiment Status (from SQLite) ---")
        print(f"Total Items:     {total}")
        print(f"Processed:       {processed_count}")
        print(f"Unprocessed:     {unprocessed_count}")
        print(f"With Errors:     {error_count}")
        print("---------------------------------------\n")

    def to_dataframe(self):
        """Export the experiment as a dataframe.

        Returns
        -------
        pd.DataFrame
            The list of files as a dataframe
        """
        import pandas as pd

        return pd.DataFrame.from_records(
            [x.to_data_row() for x in self.get_all_items()]
        )

    def to_annotation(self):
        """Export the list of data to a dictionnary for proof-reading.

        The dataset is split into consistent groups based on the content of the
        "channels" field.
        The list of files also contains the serie index.

        Returns
        -------
        dict
            Dictionnary with {set:{"channels":, "condition":, "files"}}
        """
        df = self.to_dataframe()
        return {
            f"set{k}": {
                "channels": ng[0],
                "condition": "unknown",
                "files": ng[1]["relative_path"].unique().tolist(),
            }
            for k, ng in enumerate(df.groupby("channels"))
        }

    def export_annotation(self, filepath):
        """Save the data table to a yaml for user annotation.

        Parameters
        ----------
        filepath: Path | str
            Path where to save the file
        """
        data = self.to_annotation()
        with open(filepath, "w") as f:
            yaml.dump(data, f)

    def import_annotation(self, filepath):
        """Load user annotations and populate/update the annotation table.

        Parameters
        ----------
        filepath: Path | str
            Path where to save the file

        """
        # Load the annotation from a yml file
        with open(filepath, "r") as f:
            data = yaml.safe_load(f)

        # Group items by their relative path for efficient updates
        all_items = self.get_all_items()
        items_by_path: Dict[str, List[DataItem]] = {}
        for item in all_items:
            path_str = str(item.relative_path)
            items_by_path.setdefault(path_str, []).append(item)

        # Update items using the information from the annotation dictionary
        for s in data:
            condition = data[s]["condition"]
            channels = data[s]["channels"]

            # Ensure channels is handled as a list/object (it may be JSON string in the YAML)
            if isinstance(channels, str):
                try:
                    channels = json.loads(channels)
                except (json.JSONDecodeError, TypeError):
                    pass

            # Update each file found in the set
            for rel_path in data[s].get("files", []):
                if rel_path in items_by_path:
                    affected_items = items_by_path[rel_path]
                    for item in affected_items:
                        item.condition = condition
                        item.annotated_channels = channels
                        self.update_item(item)
                    print(
                        f"Updated annotations for {len(affected_items)} items in {rel_path}."
                    )

    def load_item_data(self, item: DataItem) -> xr.DataArray:
        """Load the image data for a DataItem and apply annotations.

        This method uses the appropriate Reader to load the specific scene,
        and then maps the `annotated_channels` and `condition` to the
        resulting xarray object if they exist.

        Parameters
        ----------
        item : DataItem
            The item to load data for.

        Returns
        -------
        xr.DataArray
            A dask-backed xarray containing the image data and updated metadata.

        Raises
        ------
        ValueError
            If the specified scene cannot be found in the container file.
        """
        abs_path = item.get_absolute_container_path(self.data_dir)

        # Scenes are stored as strings (e.g., "0"), but readers iterate by integer index
        scene_idx = int(item.scene) if item.scene.isdigit() else 0

        # To ensure dask arrays remain functional after this method returns, we avoid
        # using a context manager here. Many microscopy readers (CZI, ND2, LSM)
        # require the underlying file handle to remain open for lazy loading.
        reader = Reader(abs_path)
        data = None
        for i, scene in enumerate(reader):
            if i == scene_idx:
                data = scene
                break

        if data is None:
            raise ValueError(f"Scene {item.scene} not found in {abs_path}")

        # Attach the reader instance to the xarray to keep its lifecycle tied
        # to the data, preventing premature garbage collection and file closing.
        data.attrs["_reader"] = reader

        # Apply annotated channel names if they exist and match the count
        if item.annotated_channels:
            if "C" in data.dims and len(item.annotated_channels) == data.sizes["C"]:
                data.coords["C"] = item.annotated_channels
            else:
                print(f"Warning: Annotation channel count mismatch for {item.id}")

        # Attach condition to attributes
        if item.condition:
            data.attrs["condition"] = item.condition

        return data

    def iter_data(
        self, only_unprocessed: False = False
    ) -> Iterator[Tuple[DataItem, xr.DataArray]]:
        """Yield each DataItem and its associated xarray data.

        Parameters
        ----------
        only_unprocessed: bool
            Indicate whether or not to loop through only unprocessed items

        Yields
        ------
        Iterator[Tuple[DataItem, xr.DataArray]]
            A tuple of (DataItem, xr.DataArray).
        """
        if only_unprocessed:
            for item in self.get_unprocessed_items():
                yield item, self.load_item_data(item)
        else:
            for item in self.get_all_items():
                yield item, self.load_item_data(item)

    def apply(
        self,
        processing_fn: Callable,
        client: Optional[Any] = None,
        items: Optional[List["DataItem"]] = None,
    ):
        """
        Apply a processing function to items (Pandas-like interface).

        This method delegates to the functional `apply` in the processor module.

        Parameters
        ----------
        processing_fn : Callable
            The user-defined function to apply to each item. It receives
            (DataItem, xr.DataArray, results_dir).
        client : Optional[Any], optional
            An active dask.distributed.Client for distributed execution.
        items : Optional[List[DataItem]], optional
            A specific list of items or an iterator to process. If None,
            uses the manager's `iter_data(only_unprocessed=True)`.

        Returns
        -------
        List
            A list of results from the Dask computation.
        """
        from .processor import apply as _apply

        return _apply(self, processing_fn, client=client, items=items)
