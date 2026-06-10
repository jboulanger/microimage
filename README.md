# Micro image

This project aims at factoring code that is used repetitively when developing workflows for the analysis of light microscopy image.

It is composed of a dataset management, unified image readers and writers working with xarray and basic batch processing.

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

- The reader load the data and metadata from various formats as xarrays
- The metadataparser add extra metadata to the database.
- A tool help curate these metadata.
- The ExperimentManager store list of files path with extra metadata and result files.


```python
from microimage import ExperimentManager, MetadataParser
from dask.distributed import Client, LocalCluster, Future
from functools import partial

data_dir = "data/"
results_dir = "results/"
db_path = "project.db"
exp = ExperimentManager(data_dir, results_dir, db_path)
exp.export_annotation("annotation.yml")

# verify the annotations
exp.import_annotation("annotation.yml")

# process all files
def process(item, data):
    return {"table":Table(),"labels":Contours()}

exp.apply(process)


cluster = LocalCluster(n_workers=4, threads_per_worker=1)
client = Client(cluster)
print(f"Dask Dashboard link: {client.dashboard_link}")

microimage.apply(process,)

configured_worker_task = partial(
    dask_worker_task, 
    data_dir=data_dir, 
    results_dir=results_dir, 
    db_path=db_path)

```

## Supported file formats

| Format | Read | Write |
|--------|------|-------|
| TIF    | yes  |       |
| LSM    | yes  |       |
| CZI    | yes  |       |
| ND2    | yes  |       |
| LIF    | yes  |       |
| VSI    | yes  |       |
| IMS    | yes  |       |
| NGFF   |      |       |     


