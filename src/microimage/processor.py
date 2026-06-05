"""Processing helpers to apply functions to experiment DataItems.

This module provides utilities to run a user-defined processing function
against DataItem objects, either locally or via a dask.distributed client.
"""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import xarray as xr

from . import DataItem, ExperimentManager

logger = logging.getLogger(__name__)


def _apply_to_item(
    item: Union[DataItem, Tuple[DataItem, xr.DataArray]],
    processing_fn: Callable[[DataItem, xr.DataArray, Path], Dict[str, Path]],
    data_dir: Path,
    results_dir: Path,
    db_path: Path,
) -> Tuple[str, bool, Optional[str]]:
    """Apply processing_fn to an item.

    This function handles the lifecycle of a single task on a remote worker,
    including re-initializing the database connection and committing results.

    Parameters
    ----------
    item : Union[DataItem, Tuple[DataItem, xr.DataArray]]
        The item to be processed. Can be a DataItem object or a tuple
        containing a DataItem and its pre-loaded xr.DataArray.
    processing_fn : Callable[[DataItem, xr.DataArray, Path], Dict[str, Path]]
        A user-defined function that accepts a DataItem, the loaded image data,
        and the results directory, returning a dictionary mapping result
        types to output file paths.
    data_dir : Path
        The absolute path to the raw data directory.
    results_dir : Path
        The absolute path to the results directory.
    db_path : Path
        The absolute path to the SQLite database.

    Returns
    -------
    Tuple[str, bool, Optional[str]]
        A tuple containing (item_id, success_flag, error_message).

    """
    # CRITICAL: Re-initialize ExperimentManager locally on the worker process
    # using the paths passed from the main function.
    try:
        worker_manager = ExperimentManager(
            data_dir=data_dir, results_dir=results_dir, db_path=db_path
        )
    except Exception as e:
        # item may be a tuple (DataItem, xr.DataArray)
        item_id = getattr(item[0] if isinstance(item, tuple) else item, "id", None)
        logger.error(f"Worker failed to initialize ExperimentManager: {e}")
        return item_id, False, f"Worker initialization failed: {e}"

    # Normalize item and data: item may be (DataItem, xr.DataArray)
    target_item = item[0] if isinstance(item, tuple) else item
    logger.info(f"Starting work on ID: {target_item.id}")

    try:
        if isinstance(item, tuple):
            # pre-loaded data provided by the caller
            data = item[1]
        else:
            # Load the data using the worker's manager (this handles channel mapping)
            data = worker_manager.load_item_data(target_item)

        # 1. Run the core logic using the loaded data.
        # Note: processing_fn can also update item.metadata or item.condition directly.
        generated_files: Dict[str, Path] = processing_fn(
            target_item, data, worker_manager.results_dir
        )

        # 2. Update the DataItem object
        for result_type, f_path in generated_files.items():
            # Ensure the path is relative to results directory for DB consistency
            try:
                # Check if it's already relative or subpath of project_root
                p = Path(f_path)
                if p.is_absolute():
                    rel_path = p.relative_to(worker_manager.results_dir)
                else:
                    rel_path = p
                target_item.add_result_file(result_type, str(rel_path))
            except Exception:
                target_item.add_result_file(result_type, str(f_path))

        target_item.processed = True
        target_item.error = None

    except Exception as e:
        logger.error(f"Error processing {target_item.id} on worker: {e}", exc_info=True)
        target_item.processed = False
        target_item.error = str(e)

    finally:
        # 3. CRITICAL: Commit the result to the shared DB
        try:
            worker_manager.update_item(target_item)
            logger.info(f"Finished and committed status for ID: {target_item.id}")
        except Exception:
            logger.exception("Failed to commit item status to DB")

    return target_item.id, target_item.processed, target_item.error


def apply(
    manager: ExperimentManager,
    processing_fn: Callable[[DataItem, xr.DataArray, Path], Dict[str, Path]],
    client: Optional[Any] = None,
    items: Optional[List[DataItem]] = None,
) -> List:
    """
    Apply a processing function to items using Dask.

    Parameters
    ----------
    manager : ExperimentManager
        The manager instance providing access to the experiment database and paths.
    processing_fn : Callable[[DataItem, xr.DataArray, Path], Dict[str, Path]]
        The core logic to apply to each item. It receives the DataItem and
        the loaded xarray data (with annotations applied), and the results directory.
    client : Optional[Any], optional
        An active dask.distributed.Client. If None, execution occurs locally
        using dask.bag. Defaults to None.
    items : Optional[List[DataItem]], optional
        A specific list of items to process. If None, fetches all
        unprocessed items from the database. Defaults to None.

    Returns
    -------
    List
        A list of results from the Dask computation.

    Example
    -------
    >>> def process(item, data, results_dir):
    >>>     print(item)
    >>>     return {"labels":"labels.tif","measurements":"measurement.csv"}
    >>>
    >>> microimage.apply(exp, process)
    """
    if items is None:
        items = manager.iter_data(only_unprocessed=True)

    if not items:
        logger.info("No items to process.")
        return []

    # Prepare the arguments for the workers
    task_kwargs = {
        "processing_fn": processing_fn,
        "data_dir": manager.data_dir,
        "results_dir": manager.results_dir,
        "db_path": manager.db_path,
    }

    if client:
        # Distributed execution (Asynchronous)
        return client.map(_apply_to_item, items, **task_kwargs)
    else:
        # Local execution
        import dask.bag

        return (
            dask.bag.from_sequence(items).map(_apply_to_item, **task_kwargs).compute()
        )
