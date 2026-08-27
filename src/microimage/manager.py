"""Manager from microscopy experiments."""

import json
import logging
import os
import sqlite3
from contextlib import ExitStack
from itertools import chain
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import xarray as xr
import yaml

from .models import DataItem
from .readers import BaseReader, Reader
from .results import Contours, Mask, Result, Table, Thumbnail
from .utils import inspect_data, show_contours, show_data, show_points, show_scalebar
from .writers import Writer

logger = logging.getLogger(__name__)


# Custom connection factory to return dictionary rows instead of tuples
def dict_factory(cursor, row):
    """Convert SQLite tuple row results into a dictionary.

    This function is used as a `row_factory` for an sqlite3 connection
    to make query results   more intuitive to access by column name.

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
        logger.info(f"Project root (Base for result paths) set to: {self.project_root}")

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

        # Cache for open file readers to support lazy loading and performance
        self._stack = ExitStack()
        self._readers: Dict[Path, BaseReader] = {}

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

    def __enter__(self) -> "ExperimentManager":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Close all open file handles and clean up resources.

        This should be called when the manager is no longer needed to ensure
        that all microscopy file readers are properly closed.
        """
        try:
            self._stack.close()
        except Exception as e:
            logger.error(f"Error while closing readers: {e}")
        finally:
            self._readers.clear()

    def _initialize_db(self):
        """Create the database tables if they do not already exist."""
        logger.info(f"Initializing database at {self.db_path}...")
        conn = self._get_connection()
        try:
            # Ensure we do not use Write ahead logging
            # as WAL does not work over a network filesystem
            # https://www.sqlite.org/wal.html
            conn.execute("PRAGMA journal_mode=DELETE;")

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
                    logical_type TEXT NOT NULL,
                    relative_path TEXT NOT NULL,
                    PRIMARY KEY (data_id, result_type),
                    FOREIGN KEY (data_id) REFERENCES data(id) ON DELETE CASCADE
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS annotations (
                    data_id TEXT NOT NULL,
                    channels TEXT,
                    metadata TEXT,
                    PRIMARY KEY (data_id) ON CONFLICT REPLACE,
                    FOREIGN KEY (data_id) REFERENCES data(id) ON DELETE CASCADE
                )
            """)
            conn.commit()
            logger.info("Database structure verified/created.")
        except sqlite3.Error as e:
            logger.error(f"Database Initialization Error: {e}")
        finally:
            conn.close()

    def _fetch_result_files(
        self, conn: sqlite3.Connection, item_id: str
    ) -> Dict[str, Dict[str, str]]:
        """Fetch all result files for a given item ID.

        Parameters
        ----------
        conn : sqlite3.Connection
            An active database connection.
        item_id : str
            The unique ID of the item.

        Returns
        -------
        Dict[str, Dict[str, str]]
            A dictionary mapping result name to its metadata (path and type).
        """
        cursor = conn.execute(
            "SELECT result_type, logical_type, relative_path FROM results WHERE data_id = ?",
            (item_id,),
        )
        return {
            row["result_type"]: {
                "path": row["relative_path"],
                "type": row["logical_type"],
            }
            for row in cursor.fetchall()
        }

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
            SELECT d.*, a.metadata AS annotated_metadata, a.channels AS annotated_channels
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
            logger.error(f"Database Count Error: {e}")
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

    def add_file(self, container_path: Path | str):
        """Add a single image container.

        Note the file as to in the project folder... is this good?

        Parameters
        ----------
        file_path: str | Path
            Path to the
        metadata: dictionnary
            Additional metadata as a dictionnary

        """
        logger.info(f"Adding scene in file {container_path}")
        try:
            conn = self._get_connection()
            new_items_count = self._process_container_file(Path(container_path), conn)
            conn.commit()
            if new_items_count > 0:
                logger.info(
                    f"Successfully registered {new_items_count} new items/sub-items."
                )
            else:
                logger.info("No new files or sub-items found.")
        except sqlite3.Error as e:
            logger.error(f"Database Scan Error: {e}")
            conn.rollback()
        except Exception as e:
            # Catch errors originating from the parser function
            logger.error(f"File Parser Error during scan: {e}")
            conn.rollback()
        finally:
            conn.close()

    def scan_data_directory(
        self,
        extensions: List[str] = [".png", ".jpg", ".jpeg", ".tif"],
        subfolder: str | None = None,
    ):
        """Scan the data directory for new files and register them.

        This method walks through the `data_dir`, identifies files with the
        specified extensions, and adds them to the database if they are not
        already present.

        Parameters
        ----------
        extensions : List[str], optional
            A list of file extensions to look for, by default [".lsm", ".nd2", ".czi", ".lif"].
        subfolder: str | None
            A path to a subfolder relative to the datadir.

        """
        logger.info(f"Scanning {self.data_dir} for new files...")
        conn = self._get_connection()
        found_new_items = 0

        if subfolder is None:
            folder = Path(self.data_dir)
        else:
            folder = Path(self.data_dir) / subfolder

        try:
            # Find all potential container files
            glob_iterators = (Path(folder).glob(f"**/*{ext}") for ext in extensions)
            all_container_paths = chain.from_iterable(glob_iterators)

            # Iterate through container files
            for container_path in all_container_paths:
                # Skip hidden resource files (e.g., Mac OS ._ files)
                if container_path.name.startswith("._"):
                    logger.debug(
                        f"Skipping hidden resource file: {container_path.name}"
                    )
                    continue

                # Delegate processing of each file to a helper method
                new_items_count = self._process_container_file(container_path, conn)
                found_new_items += new_items_count

            conn.commit()
            if found_new_items > 0:
                logger.info(
                    f"Successfully registered {found_new_items} new items/sub-items."
                )
            else:
                logger.info("No new files or sub-items found.")

        except sqlite3.Error as e:
            logger.error(f"Database Scan Error: {e}")
            conn.rollback()
        except Exception as e:
            # Catch errors originating from the parser function
            logger.error(f"File Parser Error during scan: {e}")
            conn.rollback()
        finally:
            conn.close()

    def _create_item_in_db(self, conn: sqlite3.Connection, item: DataItem):
        """Insert a single DataItem into the database."""
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
        logger.debug(f"Registered new item: {item.id}")

    def _process_container_file(
        self,
        container_path: Path,
        conn: sqlite3.Connection,
    ) -> int:
        """Process a single image container file, adding items to the DB."""
        relative_path = container_path.relative_to(self.data_dir)
        items_data_list = []
        new_items_count = 0

        try:
            with Reader(container_path) as reader:
                for i, scene in enumerate(reader):
                    logger.debug(f"Adding {container_path} scene [{i}]")
                    items_data_list.append(
                        {
                            "scene": str(i),
                            "_scene_xarray": scene,
                        }
                    )
        except Exception as e:
            logger.warning(f"Could not read {container_path.name}: {e}")
            return 0

        # Process each sub-item (scene) from the file
        for item_data in items_data_list:
            item_id = item_data["_scene_xarray"].attrs["unique_id"]

            if not item_id:
                logger.warning(
                    f"Skipping scene in {relative_path} due to missing unique_id."
                )
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
                    size_t = :size_t,
                    size_c = :size_c,
                    size_z = :size_z,
                    size_y = :size_y,
                    size_x = :size_x,
                    scale_t = :scale_t,
                    scale_z = :scale_z,
                    scale_y = :scale_y,
                    scale_x = :scale_x,
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
                    (item.id, name, info["type"], info["path"])
                    for name, info in item.result_files.items()
                ]
                conn.executemany(
                    """
                    INSERT INTO results (data_id, result_type, logical_type, relative_path)
                    VALUES (?, ?, ?, ?)
                """,
                    result_data,
                )

            # 4. Update annotations
            # Fetch existing annotations if any
            existing_annotation = conn.execute(
                "SELECT metadata, channels FROM annotations WHERE data_id = ?",
                (item.id,),
            ).fetchone()

            current_annotated_metadata_json = (
                existing_annotation["metadata"] if existing_annotation else None
            )
            current_annotated_metadata = (
                json.loads(current_annotated_metadata_json)
                if current_annotated_metadata_json
                else None
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
            new_annotated_metadata = (
                item.annotated_metadata
                if item.annotated_metadata is not None
                else current_annotated_metadata
            )
            new_annotated_channels = (
                item.annotated_channels
                if item.annotated_channels is not None
                else current_annotated_channels
            )

            # Only proceed if there's something to insert/update for annotations
            if new_annotated_metadata is not None or new_annotated_channels is not None:
                conn.execute(
                    """
                    INSERT INTO annotations (data_id, metadata, channels)
                    VALUES (?, ?, ?)
                    ON CONFLICT(data_id) DO UPDATE SET
                        metadata=excluded.metadata,
                        channels=excluded.channels
                """,
                    (
                        item.id,
                        json.dumps(new_annotated_metadata)
                        if new_annotated_metadata is not None
                        else None,
                        json.dumps(new_annotated_channels)
                        if new_annotated_channels is not None
                        else json.dumps(item.channels),
                    ),
                )
            conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Database Update Error for {item.id}: {e}")
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
                SELECT data_id, result_type, logical_type, relative_path
                FROM results
                WHERE data_id IN ({id_placeholders})
            """,
                item_ids,
            ).fetchall()

            # 3. Map results to their item IDs
            all_results: Dict[str, Dict[str, Dict[str, str]]] = {
                item_id: {} for item_id in item_ids
            }
            for row in result_rows:
                all_results[row["data_id"]][row["result_type"]] = {
                    "path": row["relative_path"],
                    "type": row["logical_type"],
                }

            # 4. Create and return DataItem objects
            return [
                DataItem.from_db_row(row, all_results[row["id"]]) for row in item_rows
            ]
        except sqlite3.Error as e:
            logger.error(f"Database Query Error: {e}")
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
            SELECT d.*, a.metadata AS annotated_metadata, a.channels AS annotated_channels
            FROM data d
            LEFT JOIN annotations a ON d.id = a.data_id
        """
        )

    def get_item_by_id(self, id) -> List[DataItem]:
        """Return the item.

        Returns
        -------
        List[DataItem]
            A list of all `DataItem` objects where the `error` field is not NULL.
        """
        # error IS NOT NULL and processed = 0
        return self._bulk_fetch_items(
            f"""
            SELECT d.*, a.metadata AS annotated_metadata, a.channels AS annotated_channels
            FROM data d
            LEFT JOIN annotations a ON d.id = a.data_id
            WHERE d.id is '{id}'
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
            SELECT d.*, a.metadata AS annotated_metadata, a.channels AS annotated_channels
            FROM data d
            LEFT JOIN annotations a ON d.id = a.data_id
            WHERE d.processed = 0
        """
        )

    def get_processed_items(self) -> List[DataItem]:
        """Return a list of all items marked as processed.

        Returns
        -------
        List[DataItem]
            A list of all `DataItem` objects where `processed` is False.
        """
        # processed =  (False)
        return self._bulk_fetch_items(
            """
            SELECT d.*, a.metadata AS annotated_metadata, a.channels AS annotated_channels
            FROM data d
            LEFT JOIN annotations a ON d.id = a.data_id
            WHERE d.processed = 1
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
            SELECT d.*, a.metadata AS annotated_metadata, a.channels AS annotated_channels
            FROM data d
            LEFT JOIN annotations a ON d.id = a.data_id
            WHERE d.error IS NOT NULL AND d.processed = 0
        """
        )

    def _update_item_status(self, where_clause: str = "1", message: str = ""):
        """Reset the status of items based on a WHERE clause."""
        logger.info(f"Updating item status in DB: {message}")
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                f"UPDATE data SET processed = 0, error = NULL WHERE {where_clause}"
            )
            conn.commit()
            logger.info(f"Reset status for {cursor.rowcount} items.")
        except sqlite3.Error as e:
            logger.error(f"Database Reset Error: {e}")
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
            logger.error(f"Database Report Error: {e}")
            processed_count, error_count = 0, 0
        finally:
            conn.close()

        unprocessed_count = total - processed_count - error_count

        logger.info("--- Experiment Status (from SQLite) ---")
        logger.info(f"Total Items:     {total}")
        logger.info(f"Processed:       {processed_count}")
        logger.info(f"Unprocessed:     {unprocessed_count}")
        logger.info(f"With Errors:     {error_count}")
        logger.info("---------------------------------------")

    def to_dataframe(self):
        """Export the experiment as a dataframe.

        Annotated metadata will be added as new columns in the dataframe.

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

        Returns
        -------
        dict
            Dictionnary with {set:{"channels":, "metadata":, "files"}}
        """
        groups = {}
        for item in self.get_all_items():
            # Group by the JSON string of channels (since lists aren't hashable)
            channels_key = (
                json.dumps(item.channels) if item.channels is not None else "null"
            )
            groups.setdefault(channels_key, {"paths": set(), "items": []})
            groups[channels_key]["paths"].add(str(item.relative_path))
            groups[channels_key]["items"].append(item)

        result_dict = {}
        for k, (key, group_data) in enumerate(groups.items()):
            paths = group_data["paths"]
            items = group_data["items"]

            channels_dict = {}
            orig_channels = json.loads(key) if key != "null" else None

            if orig_channels:
                channels_dict = {c: c for c in orig_channels}
                for item in items:
                    if item.annotated_channels:
                        if isinstance(item.annotated_channels, dict):
                            channels_dict.update(item.annotated_channels)
                            break
                        elif isinstance(item.annotated_channels, list):
                            for orig, ann in zip(
                                orig_channels, item.annotated_channels
                            ):
                                channels_dict[orig] = ann
                            break
            else:
                channels_dict = None

            result_dict[f"set{k}"] = {
                "channels": channels_dict,
                "metadata": "unknown",
                "files": sorted(list(paths)),
            }
        return result_dict

    def convert_annotation_to_dataframe(self):
        """Convert the annotation to a dataframe for export.

        One might prefer a csv file for annotation which is more rigid than the YAML.

        Returns
        -------
        pd.Dataframe:
            Table with a row per scene with id, relative_path, channels and annotated
            metadata as individual columns.
        """
        import pandas as pd

        df = []
        for item in self:
            d = {
                "id": item.id,
                "relative_path": item.relative_path,
                "scene": item.scene,
            }

            # Insert metadata
            if item.annotated_metadata is not None:
                for m in item.annotated_metadata:
                    d[m] = item.annotated_metadata[m]

            # Insert annotated channels
            if item.annotated_channels is not None:
                if isinstance(item.annotated_channels, dict):
                    ann_dict = {
                        c: item.annotated_channels.get(c, c) for c in item.channels
                    }
                else:
                    ann_dict = {k: v in zip(item.channels, item.annotated_channels)}
                for k, v in ann_dict:
                    d[k] = v
            else:
                # Insert original channels
                for v in item.channels:
                    d[v] = v
            df.append(d)

        return pd.DataFrame.from_records(df, index="id")

    def update_annotation_from_dataframe(self, df):
        """Update annotations from a dataframe.

        Parameter
        ---------
        df: pd.DataFrame
            Dataframe with relative_path, scene, channel_k and extra columns used to add
            annotation.
        """
        # 1. Map all items by their ID for O(1) batch lookup
        items_by_id = {item.id: item for item in self.get_all_items()}

        # Static columns to exclude from extra metadata
        base_exclude = {"relative_path", "scene"}

        # 2. Iterate using to_dict for fast row access
        for row_id, row_data in df.to_dict(orient="index").items():
            item = items_by_id.get(row_id)
            if item is None:
                continue

            # 3. Handle annotated_channels as a list
            channels_count = len(item.channels)
            channel_keys = [f"channel_{k}" for k in range(channels_count)]

            # Populate or overwrite list values
            ann_list = [row_data[key] for key in channel_keys if key in row_data]
            item.annotated_channels = {
                orig: ann for orig, ann in zip(item.channels, ann_list)
            }

            # 4. Handle remaining metadata
            if item.annotated_metadata is None:
                item.annotated_metadata = {}

            exclude_cols = base_exclude.union(channel_keys)

            # Assign remaining columns to metadata
            for col, val in row_data.items():
                if col not in exclude_cols:
                    item.annotated_metadata[col] = val

            self.update_item(item)

    def export_annotation(self, filepath):
        """Save the data table to a yaml for user annotation.

        The yaml file group the data in set of consistent channels.
        Create new groups if required (eg by condition) and update
        the name of the channel to a semantic that will be used for
        processing the data later on.

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
            metadata = data[s]["metadata"]
            channels = data[s]["channels"]

            # Ensure channels is handled as a list/object (it may be JSON string)
            if isinstance(channels, str):
                try:
                    channels = json.loads(channels)
                except (json.JSONDecodeError, TypeError):
                    pass
            # Ensure metadata is handled as a list/object (it may be JSON string)
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except (json.JSONDecodeError, TypeError):
                    pass

            # Update each file found in the set
            for rel_path in data[s].get("files", []):
                if rel_path in items_by_path:
                    affected_items = items_by_path[rel_path]
                    for item in affected_items:
                        if isinstance(metadata, Dict):
                            item.annotated_metadata = metadata
                        if isinstance(channels, (list, dict)):
                            item.annotated_channels = channels
                        self.update_item(item)
                    logger.info(
                        f"Updated annotations for {len(affected_items)} items "
                        f"in {rel_path}."
                    )

    def load_item_data(self, item: DataItem) -> xr.DataArray:
        """Load the image data for a DataItem and apply annotations.

        This method uses the appropriate Reader to load the specific scene,
        and then maps the `annotated_channels` and `annotated_metadata` to the
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

        # Retrieve from cache or open new reader
        if abs_path not in self._readers:
            reader = Reader(abs_path)
            self._stack.enter_context(reader)
            self._readers[abs_path] = reader

        reader = self._readers[abs_path]
        try:
            data = reader.get_scene(scene_idx)
        except IndexError:
            raise ValueError(f"Scene {item.scene} not found in {abs_path}")

        # Apply annotated channel names if they exist and match the count
        if item.annotated_channels:
            if isinstance(item.annotated_channels, dict):
                ann_list = [item.annotated_channels.get(c, c) for c in item.channels]
            else:
                ann_list = item.annotated_channels
            if "C" in data.dims and len(ann_list) == data.sizes["C"]:
                data.coords["C"] = ann_list
            else:
                logger.warning(f"Annotation channel count mismatch for {item.id}")

        # Attach annotated metadata to attributes
        if item.annotated_metadata:
            for k, v in item.annotated_metadata.items():
                data.attrs[k] = v

        return data

    def save_result(self, item: "DataItem", result_name: str, data: Result):
        """Save a result object to the results directory and track it.

        The file path is automatically generated based on the result_name and logical_type.

        Parameters
        ----------
        item : DataItem
            The item the result belongs to.
        result_name : str
            The unique identifier for this result (used as filename and DB key).
        data : Result
            The data to save.
        """
        # Path relative to results_dir: item_id/result_name.ext
        rel_result_path = Path(item.id) / f"{result_name}{data.ext}"
        abs_result_path = self.results_dir / rel_result_path
        abs_result_path.parent.mkdir(parents=True, exist_ok=True)

        # 2. Use Writer to save
        with Writer(abs_result_path) as writer:
            writer.write(data.data)

        # 3. Update DataItem
        # Result paths in DB are relative to results_dir
        path_to_store = abs_result_path.relative_to(self.results_dir)
        item.add_result_file(result_name, str(path_to_store), data.logical_type)

    def load_result(self, item: "DataItem", result_name: str) -> Result:
        """Load a result by name and return it as the appropriate Result type.

        Parameters
        ----------
        item : DataItem
            The item whose result to load.
        result_name : str
            The name of the result to load.

        Returns
        -------
        Result
            The loaded result wrapped in its specific Result class.
        """
        if result_name not in item.result_files:
            raise KeyError(f"Result '{result_name}' not found for item {item.id}")

        res_info = item.result_files[result_name]
        logical_type = res_info["type"]
        abs_path = self.results_dir / res_info["path"]

        if logical_type == "mask":
            import tifffile

            return Mask(tifffile.imread(abs_path))

        if logical_type == "table":
            import pandas as pd

            return Table(pd.read_csv(abs_path))

        if logical_type == "contours":
            contours_data = []
            with open(abs_path, "r") as f:
                for line in f:
                    parts = line.strip().split(" ")
                    if parts:
                        # Format: class_id x1 y1 x2 y2 ...
                        row = [int(parts[0])] + [float(x) for x in parts[1:]]
                        contours_data.append(row)
            return Contours(contours_data)

        if logical_type == "thumbnail":
            import tifffile

            return Thumbnail(tifffile.imread(abs_path))

        return Result(data=None, logical_type=logical_type)

    def iter_results(self, result_name) -> Iterator[Tuple[DataItem, Result]]:
        """Yield each Result.

        Parameters
        ----------
        result_name : str
            The name of the result to load.

        Returns
        -------
        Iterator[Tuple[DataItem, Result]]
            A tuple of (DataItem, Result).
        """
        for item in self.get_processed_items():
            yield item, self.load_result(item, result_name)

    def iter_data(
        self, only_unprocessed: bool = False
    ) -> Iterator[Tuple[DataItem, xr.DataArray]]:
        """Yield each DataItem and its associated xarray data.

        Parameters
        ----------
        only_unprocessed : bool, optional
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

    def show_data(self, item, channels, cmap, percentiles=[0, 100], gamma=1.0, ax=None):
        """Load and show the data item as maximum intensity projection at frame 0.

        Parameters
        ----------
        item: DataItem
            Item to show
        channels: list[str]
            Channels name to use.
        cmap:list[str]
        """
        data = self.load_item_data(item)
        show_data(data, channels, cmap, percentiles, gamma, ax)

    def inspect_data(self, item, ax):
        """Display each channel for the item.

        Parameters
        ----------
        item: DataItem
            Item to display
        ax: matplotlib.axes
            Axes to use
        """
        data = self.load_item_data(item)
        inspect_data(data, ax)

    def show_contours(self, item, name, ax=None, **kwargs):
        """Show segmentation contours.

        Parameters
        ----------
        item: DataItem
            Item to show
        """
        contours = self.load_result(item, name)
        show_contours(
            [float(item.size_y) * item.scale_y, float(item.size_x) * item.scale_x],
            contours.data,
            ax,
            **kwargs,
        )

    def show_points(self, item, name, columns=None, ax=None, **kwargs):
        """Load coordinates of points from a table and add then to the figure.

        Parameters
        ----------
        item: DataItem
        name: str
            Name of the table to load
        columns: list[str]
            Name of the 2 columns to use for x and y coordinates.

        """
        df = self.load_result(item, name).data
        # Check that the requested columns are in the table columns
        if columns is None:
            if item.size_z == 1:
                columns = ["centroid-0", "centroid-1"]
            else:
                columns = ["centroid-2", "centroid-1"]
        for x in columns:
            if x not in df.columns:
                raise KeyError(f"Column {x} in not in table {name}")
        if "frame index" in df.columns:
            df = df[df["frame index"] == 0]
        points = np.array(df[columns])
        show_points(
            points[:, 0] * item.scale_x, points[:, 1] * item.scale_y, ax=ax, **kwargs
        )

    def show_scalebar(self, item, color="white", ax=None):
        """Show a scalebar."""
        data = self.load_item_data(item)
        xmin = data["X"][0].item()
        xmax = data["X"][-1].item()
        ymin = data["Y"][0].item()
        ymax = data["Y"][-1].item()

        show_scalebar(xmin, xmax, ymin, ymax, color, ax)
