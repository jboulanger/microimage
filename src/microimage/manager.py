import sqlite3
import json
import os
from typing import List, Dict, Any, Optional, Union, Iterator, Tuple, Callable
from pathlib import Path
from dataclasses import dataclass, field
from itertools import chain


# Custom connection factory to return dictionary rows instead of tuples
def dict_factory(cursor, row):
    """Converts SQLite tuple row results into a dictionary."""
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
    Represents a single item (image) in your dataset.

    relative_image_path now refers to the container file, and is stored
    relative to the ExperimentManager's data_dir.
    internal_index identifies the specific image within that container file.
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
        """Adds or updates key-value pairs in the metadata."""
        self.metadata.update(metadata_dict)

    def add_result_file(self, result_type: str, relative_file_path: str):
        """
        Registers a new result file for this item, categorized by type.
        """
        self.result_files[result_type] = relative_file_path

    def get_absolute_container_path(self, base_data_dir: Path) -> Path:
        """
        Returns the full, absolute path to the container file, resolving relative_image_path
        against the provided base_data_dir (which should be the ExperimentManager's data_dir).
        """
        return (base_data_dir / self.relative_image_path).resolve()

    def get_full_id(self) -> str:
        """Returns the unique ID for this DataItem."""
        return self.id

    def to_db_row(self) -> Dict[str, Any]:
        """
        Converts the DataItem object into a dictionary suitable for DB insertion/update
        for the 'items' table (excluding result_files).
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
        """
        Creates a DataItem object from a database row dictionary, optionally
        including result files fetched from the separate table.
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
    Manages the entire collection of DataItems using an SQLite database

    We want to store in it some extra metadata on the dataset collection.
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
        """Returns a new database connection."""
        conn = sqlite3.connect(
            self.db_path, timeout=10.0
        )  # Set a high timeout for concurrency
        conn.row_factory = dict_factory
        return conn

    def _initialize_db(self):
        """
        Creates the necessary tables if they don't exist.
        The 'items' table now includes 'internal_index'.
        """
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
        """Helper to fetch result files for a single item from the dedicated table."""
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
        """Helper to fetch an item and its associated results, returning a DataItem."""
        cursor = conn.execute("SELECT * FROM items WHERE id = ?", (item_id,))
        item_row = cursor.fetchone()

        if not item_row:
            return None

        result_files = self._fetch_result_files(conn, item_id)

        return DataItem.from_db_row(item_row, result_files)

    # --- Python Container Protocol Methods ---

    def __getitem__(self, item_id: str) -> "DataItem":
        """
        Retrieves a single DataItem by its ID (filename:index) from the database.
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
        """Returns the total number of items in the dataset, enabling len(manager)."""
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
        """
        Returns an iterator over all DataItem objects in the database, enabling 'for item in manager:'.
        """
        return iter(self.get_all_items())

    # --- Utility Methods ---

    def scan_data_directory(
        self,
        extensions: List[str] = [".png", ".jpg", ".jpeg", ".tif"],
        multi_file_parser: MultiFileParser = None,
    ):
        """
        Scans the data directory for new files. If multi_file_parser is provided,
        it handles all file types and returns one or more items per file.
        If not provided, every file is treated as a single DataItem.
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
        """
        Updates the status of a single item in the database, including
        synchronizing its associated result files.
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
        """
        Gets the full, absolute path for an item's container file by resolving
        its relative path against the ExperimentManager's data directory.
        """
        try:
            item = self.__getitem__(item_id)
            # FIX: Use self.data_dir as the base for resolving the relative path
            return item.get_absolute_container_path(self.data_dir)
        except KeyError:
            return None

    def get_internal_index(self, item_id: str) -> Optional[str]:
        """Gets the internal index for an item."""
        try:
            item = self.__getitem__(item_id)
            return item.internal_index
        except KeyError:
            return None

    def _bulk_fetch_items(self, query: str) -> List[DataItem]:
        """Generic helper for fetching multiple items."""
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
        """Returns a list of all DataItem objects from the database."""
        return self._bulk_fetch_items("SELECT * FROM items")

    def get_unprocessed_items(self) -> List[DataItem]:
        """Returns a list of all items that are not yet marked as processed."""
        # processed = 0 (False)
        return self._bulk_fetch_items("SELECT * FROM items WHERE processed = 0")

    def get_items_with_errors(self) -> List[DataItem]:
        """Returns a list of all items that have an error message set."""
        # error IS NOT NULL and processed = 0
        return self._bulk_fetch_items(
            "SELECT * FROM items WHERE error IS NOT NULL AND processed = 0"
        )

    def force_reprocess_all(self):
        """Resets the 'processed' status for ALL items in the database."""
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
        """Resets the 'processed' and 'error' status for all items that previously failed."""
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
        """Prints a summary of the dataset status using fast SQL queries."""
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
