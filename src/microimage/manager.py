import sqlite3
import json
import os
from typing import List, Dict, Any, Optional, Union, Iterator, Tuple, Callable
from pathlib import Path
from dataclasses import dataclass, field
from itertools import chain


# Custom connection factory to return dictionary rows instead of tuples
def dict_factory(cursor, row):
    """Converts SQLite tuple row results into a dictionary.

    This function is used as a `row_factory` for an sqlite3 connection
    to make query results more intuitive to access by column name.

    Parameters
    ----------
    cursor : sqlite3.Cursor
        The cursor object from which the row was fetched.
    row : tuple
        The raw tuple row fetched from the database.

    """
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d


# Type hint for the new parser function: takes a Path and returns a list of sub-item data dictionaries.
# Each dictionary MUST contain at least 'internal_index' and 'metadata'.
MultiFileParser = Callable[[Path], List[Dict[str, Any]]]


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
    relative_image_path : Path
        The path to the container file, relative to the `ExperimentManager`'s
        `data_dir`.
    internal_index : str, optional
        A unique identifier for the item within its container file (e.g., a
        scene number or series index). Defaults to "".

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
    relative_image_path: Path

    # NEW: Index/identifier for the specific image within the container file
    internal_index: str = ""

    # Fields with default values
    metadata: Dict[str, Any] = field(default_factory=dict)
    # This remains a Dict[type: path] in memory, but is stored in a separate DB table.
    # NOTE: Result paths are stored relative to ExperimentManager's project_root (DB parent).
    result_files: Dict[str, str] = field(default_factory=dict)
    processed: bool = field(default=False)
    error: Optional[str] = field(default=None)

    # ID is derived from the path and index and is not passed to __init__
    id: str = field(init=False)

    def __post_init__(self):
        """Called automatically after __init__ to set derived fields."""
        container_name = self.relative_image_path.name
        if self.internal_index:
            # Compound ID: Filename:Index (ensuring uniqueness)
            self.id = f"{container_name}:{self.internal_index}"
        else:
            # Fallback for single-image files (ID is just the filename)
            self.id = container_name

    def add_metadata(self, metadata_dict: Dict[str, Any]):
        """Add or update key-value pairs in the item's metadata.

        Parameters
        ----------
        metadata_dict : Dict[str, Any]
            A dictionary of metadata to merge into the existing metadata.

        """
        self.metadata.update(metadata_dict)

    def add_result_file(self, result_type: str, relative_file_path: str):
        """Register a new result file for this item.

        Parameters
        ----------
        result_type : str
            A string identifying the type of result (e.g., "thumbnail", "mask").
        relative_file_path : str
            The path to the result file, relative to the project root.

        """
        self.result_files[result_type] = relative_file_path

    def get_absolute_container_path(self, base_data_dir: Path) -> Path:
        """Return the absolute path to the item's container file.

        Parameters
        ----------
        base_data_dir : Path
            The absolute path to the main data directory, which serves as the
            base for the item's `relative_image_path`.

        Returns
        -------
        Path
            The resolved, absolute path to the container file.

        """
        return (base_data_dir / self.relative_image_path).resolve()

    def get_full_id(self) -> str:
        """Return the unique ID for this DataItem.

        Returns
        -------
        str
            The unique identifier string.
        """
        return self.id

    def to_db_row(self) -> Dict[str, Any]:
        """
        Converts the DataItem object into a dictionary suitable for DB insertion/update
        for the 'items' table (excluding result_files).
        """
        """Convert the DataItem to a dictionary for database storage.

        Returns
        -------
        Dict[str, Any]
            A dictionary representation of the item for the 'items' table.
        """
        return {
            "id": self.id,
            "relative_image_path": str(self.relative_image_path),
            "internal_index": self.internal_index,
            "metadata": json.dumps(self.metadata),
            "processed": int(self.processed),
            "error": self.error,
        }

    @classmethod
    def from_db_row(
        cls, data: Dict[str, Any], result_files: Optional[Dict[str, str]] = None
    ) -> "DataItem":
        """Create a DataItem instance from a database row.

        Parameters
        ----------
        data : Dict[str, Any]
            A dictionary representing a row from the 'items' table.
        result_files : Optional[Dict[str, str]], optional
            A dictionary of result files for this item, by default None.

        Returns
        -------
        DataItem
            A fully hydrated DataItem instance.

        """
        relative_path_str = data.get("relative_image_path")
        if relative_path_str is None:
            raise ValueError("Database row missing 'relative_image_path' key.")

        # Extract the new field
        internal_index = data.get("internal_index", "")

        # Instantiate the dataclass with the required argument
        item = cls(Path(relative_path_str), internal_index=internal_index)

        # Hydrate the rest of the fields
        item.metadata = json.loads(data.get("metadata", "{}") or "{}")
        item.processed = bool(data.get("processed", 0))
        item.error = data.get("error", None)

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
        conn.row_factory = dict_factory
        return conn

    def _initialize_db(self):
        """Create the database tables if they do not already exist."""
        print(f"Initializing database at {self.db_path}...")
        conn = self._get_connection()
        try:
            # Table for main item data
            conn.execute("""
                CREATE TABLE IF NOT EXISTS items (
                    id TEXT PRIMARY KEY,
                    relative_image_path TEXT NOT NULL,
                    internal_index TEXT NOT NULL,
                    metadata TEXT,
                    processed INTEGER NOT NULL,
                    error TEXT
                )
            """)

            # Table for one-to-many result files relationship
            conn.execute("""
                CREATE TABLE IF NOT EXISTS result_files (
                    item_id TEXT NOT NULL,
                    result_type TEXT NOT NULL,
                    relative_file_path TEXT NOT NULL,
                    PRIMARY KEY (item_id, result_type),
                    FOREIGN KEY (item_id) REFERENCES items(id) ON DELETE CASCADE
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
            "SELECT result_type, relative_file_path FROM result_files WHERE item_id = ?",
            (item_id,),
        )
        # Convert list of rows (type, path) to a dictionary {type: path}
        return {
            row["result_type"]: row["relative_file_path"] for row in cursor.fetchall()
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
        cursor = conn.execute("SELECT * FROM items WHERE id = ?", (item_id,))
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
            count = conn.execute("SELECT COUNT(id) FROM items").fetchone()["COUNT(id)"]
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
            A list of file extensions to look for, by default [".png", ".jpg", ".jpeg", ".tif"].
        multi_file_parser : MultiFileParser, optional
            A callable that takes a file `Path` and returns a list of dicts,
            one for each sub-item (e.g., scene) in the file. If not provided,
            each file is treated as a single item.

        """
        print(f"Scanning {self.data_dir} for new files...")
        conn = self._get_connection()
        cursor = conn.cursor()
        found_new_files = 0

        try:
            # 1. Find all potential container files
            glob_iterators = (self.data_dir.glob(f"*{ext}") for ext in extensions)
            all_container_paths = chain.from_iterable(glob_iterators)

            # 2. Iterate through container files
            for container_path in all_container_paths:
                # Skip hidden resource files (e.g., Mac OS ._ files)
                if container_path.name.startswith("._"):
                    print(f"  Skipping hidden resource file: {container_path.name}")
                    continue

                # FIX: Calculate relative path from the *data directory*
                relative_path = container_path.relative_to(self.data_dir)

                items_data_list = []

                # If a parser is provided, it handles the logic for all files
                if multi_file_parser:
                    print(f"  Using parser to analyze file: {container_path.name}")
                    # The parser decides if it's a single or multi-image container
                    items_data_list = multi_file_parser(container_path)

                # If no parser was used OR the parser returned an empty list (e.g., unrecognized file),
                # default to treating the file as a single DataItem.
                if not items_data_list:
                    # Treat as a single-image file
                    items_data_list.append(
                        {
                            "internal_index": "",  # Empty index for the file name only
                            "metadata": {},
                        }
                    )

                # 3. Process each sub-item extracted from the container/file
                for item_data in items_data_list:
                    internal_index = item_data.get("internal_index", "")

                    # Create the DataItem and calculate the unique ID
                    item = DataItem(relative_path, internal_index=internal_index)

                    # Check if item (by its compound ID) already exists
                    cursor.execute("SELECT id FROM items WHERE id = ?", (item.id,))
                    if cursor.fetchone():
                        continue

                    # Add parsed metadata
                    item.add_metadata(item_data.get("metadata", {}))

                    # Insert new item into ITEMS table
                    row = item.to_db_row()
                    cursor.execute(
                        """
                        INSERT INTO items (id, relative_image_path, internal_index, metadata, processed, error)
                        VALUES (:id, :relative_image_path, :internal_index, :metadata, :processed, :error)
                    """,
                        row,
                    )

                    found_new_files += 1
                    print(f"  Registered new item: {item.id}")

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
            row = item.to_db_row()
            # 1. Update the main item row
            conn.execute(
                """
                UPDATE items SET 
                    internal_index = :internal_index,
                    metadata = :metadata, 
                    processed = :processed, 
                    error = :error
                WHERE id = :id
            """,
                row,
            )

            # 2. Delete existing result files for this item
            conn.execute("DELETE FROM result_files WHERE item_id = ?", (item.id,))

            # 3. Insert the current set of result files
            if item.result_files:
                result_data = [
                    (item.id, result_type, relative_path)
                    for result_type, relative_path in item.result_files.items()
                ]
                conn.executemany(
                    """
                    INSERT INTO result_files (item_id, result_type, relative_file_path)
                    VALUES (?, ?, ?)
                """,
                    result_data,
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

    def get_internal_index(self, item_id: str) -> Optional[str]:
        """Get the internal index for an item.

        Parameters
        ----------
        item_id : str
            The unique ID of the item.

        Returns
        -------
        Optional[str]
            The internal index string, or `None` if the item is not found.
        """
        try:
            item = self.__getitem__(item_id)
            return item.internal_index
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
                SELECT item_id, result_type, relative_file_path 
                FROM result_files 
                WHERE item_id IN ({id_placeholders})
            """,
                item_ids,
            ).fetchall()

            # 3. Map results to their item IDs
            all_results: Dict[str, Dict[str, str]] = {
                item_id: {} for item_id in item_ids
            }
            for row in result_rows:
                all_results[row["item_id"]][row["result_type"]] = row[
                    "relative_file_path"
                ]

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
        return self._bulk_fetch_items("SELECT * FROM items")

    def get_unprocessed_items(self) -> List[DataItem]:
        """Return a list of all items not yet marked as processed.

        Returns
        -------
        List[DataItem]
            A list of all `DataItem` objects where `processed` is False.
        """
        # processed = 0 (False)
        return self._bulk_fetch_items("SELECT * FROM items WHERE processed = 0")

    def get_items_with_errors(self) -> List[DataItem]:
        """Return a list of all items that have a recorded error.

        Returns
        -------
        List[DataItem]
            A list of all `DataItem` objects where the `error` field is not NULL.
        """
        # error IS NOT NULL and processed = 0
        return self._bulk_fetch_items(
            "SELECT * FROM items WHERE error IS NOT NULL AND processed = 0"
        )

    def force_reprocess_all(self):
        """Reset the 'processed' and 'error' status for all items.

        This method marks all items in the database as unprocessed, allowing
        them to be picked up again by a processing pipeline.
        """
        print("Resetting 'processed' status for all items in the database.")
        conn = self._get_connection()
        try:
            conn.execute("UPDATE items SET processed = 0, error = NULL")
            conn.commit()
            print(f"Reset status for {conn.total_changes} items.")
        except sqlite3.Error as e:
            print(f"Database Reset Error: {e}")
        finally:
            conn.close()

    def reset_errors(self):
        """Reset the status for all items that previously failed.

        This method finds all items with a non-null `error` field and resets
        their `processed` and `error` statuses, allowing a re-attempt at processing.
        """
        print("Resetting status for all items with errors in the database.")
        conn = self._get_connection()
        try:
            # Resets all items where error is present (and processed is False)
            conn.execute(
                "UPDATE items SET processed = 0, error = NULL WHERE error IS NOT NULL"
            )
            conn.commit()
            print(f"Reset status for {conn.total_changes} items.")
        except sqlite3.Error as e:
            print(f"Database Reset Error: {e}")
        finally:
            conn.close()

    def report_status(self):
        """Print a summary of the dataset's processing status."""
        total = len(self)
        conn = self._get_connection()
        try:
            processed_count = conn.execute(
                "SELECT COUNT(id) FROM items WHERE processed = 1"
            ).fetchone()["COUNT(id)"]
            error_count = conn.execute(
                "SELECT COUNT(id) FROM items WHERE error IS NOT NULL"
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
