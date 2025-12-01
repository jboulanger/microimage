# Micro image

This project aims at factoring code that is used repetitively when developing workflows for the analysis of light microscopy image.

It is composed of a dataset management, unified image reader that decode basic metadata to xarray and basic batch processing.

## Installation

```bash
pip install ...
```

## Data organization

```bash
.
├── data
│   ├── image001.lsm
│   ├── image002.lsm
│   └── image003.lsm
└── project
    └── project.db
```

## Usage



```python
from microimage import ExperimentManager, MetadataParser
from dask.distributed import Client, LocalCluster, Future
from functools import partial

data_dir = "data/"
results_dir = "results/"
db_path = "project.db"
m = ExperimentManager(data_dir, results_dir, db_path)
p = MetadataParser({"dapi": 1, "gfp": 2, "rfp": 0})
m.scan_data_directory(["lsm","czi"], p)


cluster = LocalCluster(n_workers=4, threads_per_worker=1)
client = Client(cluster)
print(f"Dask Dashboard link: {client.dashboard_link}")

configured_worker_task = partial(
    dask_worker_task, 
    data_dir=data_dir, 
    results_dir=results_dir, 
    db_path=db_path)

```


