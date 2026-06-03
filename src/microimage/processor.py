from pathlib import Path
from typing import Dict, Tuple, Optional, Callable  # Added Callable


from . import ExperimentManager, DataItem


def dask_worker_task(
    item: DataItem,
    processing_fn: Callable,
    data_dir: Path,
    results_dir: Path,
    db_path: Path,
) -> Tuple[str, bool, Optional[str]]:
    """
    Generic wrapper function executed by Dask worker.
    It manages the database connection and delegates the actual work to the injected processing_fn.
    """
    # CRITICAL: Re-initialize ExperimentManager locally on the worker process
    # using the paths passed from the main function.
    try:
        worker_manager = ExperimentManager(
            data_dir=data_dir, results_dir=results_dir, db_path=db_path
        )
    except Exception as e:
        print(f"Worker failed to initialize ExperimentManager: {e}")
        return item.id, False, f"Worker initialization failed: {e}"

    print(f"Starting work on ID: {item.id}")

    try:
        # 1. Run the core logic using the dynamically passed function
        generated_files: Dict[str, Path] = processing_fn(
            item, worker_manager.results_dir, worker_manager.data_dir
        )

        # 2. Update the DataItem object
        for result_type, f_path in generated_files.items():
            rel_result_path = f_path.relative_to(worker_manager.project_root)
            item.add_result_file(result_type, str(rel_result_path))

        item.processed = True
        item.error = None

    except Exception as e:
        print(f"!! ERROR processing {item.id} on worker: {e}")
        item.processed = False
        item.error = str(e)

    finally:
        # 3. CRITICAL: Commit the result to the shared DB
        worker_manager.update_item(item)
        print(f"Finished and committed status for ID: {item.id}")

    return item.id, item.processed, item.error
