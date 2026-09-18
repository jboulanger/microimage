"""Processing helpers to apply functions to experiment DataItems.

This module provides utilities to run a user-defined processing function
against DataItem objects, either locally or via a dask.distributed client.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import xarray as xr

from .manager import ExperimentManager
from .models import DataItem
from .results import Parameters, Result

logger = logging.getLogger(__name__)


class Processor(ABC):
    """Analysis base class for processing DataItems.

    Subclasses implement `process(item, data)` to perform custom
    analysis on each image and metadata item.
    """

    def __call__(
        self, item: DataItem, data: xr.DataArray
    ) -> Dict[str, Result]:
        """Process an item and its image data."""
        return self.process(item, data)

    @abstractmethod
    def process(
        self, item: DataItem, data: xr.DataArray
    ) -> Dict[str, Result]:
        """Process an item and its image data.

        Returns
        -------
        Dict[str, Result]
            Mapping of result names to microimage Result instances.
        """
        ...

    def validate(
        self, item: DataItem, data: xr.DataArray
    ) -> Tuple[bool, Optional[str]]:
        """Validate whether the item and data are suitable for processing.

        Parameters
        ----------
        item : DataItem
            The metadata item to validate.
        data : xr.DataArray
            The image data to validate.

        Returns
        -------
        Tuple[bool, Optional[str]]
            A tuple of (is_valid, error_or_skip_message).
        """
        return True, None

    def get_parameters(self) -> Optional[Dict[str, Any]]:
        """Return parameters dictionary for provenance tracking.

        Returns
        -------
        Optional[Dict[str, Any]]
            Parameters dictionary to be saved alongside the results.
        """
        return None


def _apply_to_item(
    item: Union[DataItem, Tuple[DataItem, xr.DataArray]],
    processing_fn: Union[Processor, Callable[[DataItem, xr.DataArray], Dict[str, Any]]],
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
    processing_fn : Union[Processor, Callable[[DataItem, xr.DataArray], Dict[str, Any]]]
        A Processor instance or a user-defined function that accepts a DataItem
        and loaded image data, returning a dictionary mapping result names to Result objects.
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
    # Normalize item and data: item may be (DataItem, xr.DataArray)
    target_item = item[0] if isinstance(item, tuple) else item
    item_id = target_item.id

    try:
        # Using context manager ensures files are closed even if processing fails
        with ExperimentManager(
            data_dir=data_dir, results_dir=results_dir, db_path=db_path
        ) as worker_manager:
            logger.info(f"Starting work on ID: {item_id}")

            try:
                if isinstance(item, tuple):
                    data = item[1]
                else:
                    data = worker_manager.load_item_data(target_item)

                # 1. Validation hook (if processing_fn provides validate)
                if hasattr(processing_fn, "validate") and callable(processing_fn.validate):
                    is_valid, msg = processing_fn.validate(target_item, data)
                    if not is_valid:
                        err_msg = f"Validation failed: {msg}" if msg else "Validation failed."
                        logger.warning(f"Item {item_id} validation failed: {err_msg}")
                        target_item.processed = False
                        target_item.error = err_msg
                        return target_item.id, False, target_item.error

                # 2. Process data
                results = processing_fn(target_item, data)

                if results is None:
                    err_msg = "Processor returned None."
                    logger.warning(f"Item {item_id}: {err_msg}")
                    target_item.processed = False
                    target_item.error = err_msg
                    return target_item.id, False, target_item.error

                # 3. Parameter provenance hook (if processing_fn provides get_parameters)
                if hasattr(processing_fn, "get_parameters") and callable(
                    processing_fn.get_parameters
                ):
                    params = processing_fn.get_parameters()
                    if params and "parameters" not in results:
                        results["parameters"] = Parameters(params)

                # 4. Store results
                for res_name, res_obj in results.items():
                    if not isinstance(res_obj, Result):
                        raise TypeError(
                            f"Result '{res_name}' must be an instance of Result, "
                            f"got {type(res_obj)} instead."
                        )
                    worker_manager.save_result(target_item, res_name, res_obj)

                target_item.processed = True
                target_item.error = None

            except Exception as e:
                logger.error(f"Error processing {item_id}: {e}", exc_info=True)
                target_item.processed = False
                target_item.error = str(e)
            finally:
                # Commit result to DB before the manager closes its connection
                worker_manager.update_item(target_item)
                logger.info(f"Finished and committed status for ID: {item_id}")

    except Exception as e:
        logger.error(f"Worker failed to initialize ExperimentManager: {e}")
        return item_id, False, f"Worker initialization failed: {e}"

    return target_item.id, target_item.processed, target_item.error


def apply(
    manager: ExperimentManager,
    processing_fn: Union[Processor, Callable[[DataItem, xr.DataArray], Dict[str, Any]]],
    client: Optional[Any] = None,
    items: Optional[List[DataItem]] = None,
) -> List:
    """
    Apply a processing function to items using Dask.

    Parameters
    ----------
    manager : ExperimentManager
        The manager instance providing access to the experiment database and paths.
    processing_fn : Callable[[DataItem, xr.DataArray], Dict[str, Any]]
        The core logic to apply to each item. It receives the DataItem and
        the loaded xarray data (with annotations applied).
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
    >>> def process(item, data):
    >>>     print(item)
    >>>     return {"mask": some_array, "table1": some_df}
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
        # Sequential Local execution
        import dask.bag

        return (
            dask.bag.from_sequence(items)
            .map(_apply_to_item, **task_kwargs)
            .compute(scheduler="synchronous")
        )


def summarize_run(result: list[tuple[str, bool, str]]):
    """Summarize the output of running apply in a readable format.

    Parameter
    ---------
    result: list[tuple[str,bool,str]

    """
    nsuccess = sum([elem[1] for elem in result])

    print(f"Pass...{nsuccess}")
    print(f"Fail...{len(result) - nsuccess}")
    print(f"Total..{len(result)}")
