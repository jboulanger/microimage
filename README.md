# Microimage

Microimage reduces boilerplate when developing analysis workflows for light microscopy. It provides dataset management, unified image readers and writers built on [xarray](https://docs.xarray.dev/), and a batch processing framework.

The fundamental unit of the library is a **scene**: a single physical position within a dataset to which results and annotations are attached.

## Installation

```bash
pip install microimage
# or using uv
uv add microimage
```

## Data Organization

```bash
.
├── data
│   ├── image001.lsm
│   ├── image002.lsm
│   └── image003.lsm
├── results
│   ├── WEDF-ERPT
│   │   ├── contours.txt
│   │   └── measurements.csv
│   ├── OXDF-BFJP
│   └── QQAP-FGOP
└── project
    ├── annotation.yml
    └── project.db
```

- The `results` folder contains one subfolder per scene, named after the scene's unique ID (`DataItem`) in the database.
- The annotation file can be edited to enrich scenes with semantic channel labels and experimental metadata.

## Database Schema

The project uses a SQLite database to track images, annotations, and processing results across three tables, each populated at a different stage of the workflow.

### `data` Table

Populated during directory scanning. Contains one row per scene (physical position).

| Column | Type | Description |
| :--- | :--- | :--- |
| `id` | TEXT | Primary key. SHA1 hash derived from the file name, modification time, and scene index. |
| `relative_path` | TEXT | Path to the container file, relative to the data directory. |
| `scene` | TEXT | Index or name of the scene within the container file. |
| `metadata` | TEXT | JSON string of arbitrary metadata extracted from the file. |
| `processed` | INTEGER | Flag (0 or 1) indicating whether the scene has been successfully processed. |
| `error` | TEXT | Error message recorded if processing failed, otherwise null. |
| `size_t, size_c, ...` | INTEGER | Dimension sizes (T, C, Z, Y, X). |
| `scale_t, scale_z, ...` | REAL | Physical scale/calibration values for each dimension. |
| `channels` | TEXT | JSON list of channel names as extracted from the file. |

> **Note:** The id is computed at scan time from the file name, modification time, and scene index. If a file is renamed or modified and the directory is rescanned, a new ID will be generated, breaking the link to any existing results. Avoid rescanning after files have been processed.

### `annotation` Table

Populated by importing a user-edited YAML file. Overrides or extends the raw metadata from the `data` table with semantic channel names and experimental context (condition, well, run, etc.).

| Column | Type | Description |
| :--- | :--- | :--- |
| `data_id` | TEXT | Foreign key referencing `data.id`. |
| `channels` | TEXT | JSON list of user-defined semantic channel names, overriding those in `data.channels`. |
| `metadata` | TEXT | JSON string of arbitrary user-defined metadata. |

*Primary key: `(data_id)`.*

### `results` Table

Populated during processing. Tracks output files generated for each scene, one row per result type.

| Column | Type | Description |
| :--- | :--- | :--- |
| `data_id` | TEXT | Foreign key referencing `data.id`. |
| `result_type` | TEXT | Type identifier for the result (e.g., `"thumbnail"`, `"mask"`). |
| `relative_path` | TEXT | Path to the result file, relative to the project root. |

*Primary key: `(data_id, result_type)`. Each scene can hold at most one result per type.*

## Usage

- **Readers** load data and metadata from various file formats as xarrays with labelled axes.
- The **ExperimentManager** organizes the dataset and provides unified access using semantic channel names. It is responsible for scanning the data folder and populating the data table. When files are multi-scene containers, each scene in the container is handled as a single object and its properties stored in the data table as a single row. In short the atomic unit is the scene and not the file.
- **Annotations** are applied in bulk by editing and re-importing the YAML annotation file.
- **Batch processing** is done via `apply(exp, process, client)`, where `process` has the signature `def process(item: DataItem, data: xr.DataArray) -> Dict[str, Result]`. Any exception raised inside `process` is caught, logged to the `error` column, and the scene is left as unprocessed — enabling safe partial reruns. On success, the scene is flagged as `processed=1`. Results are written to disk automatically based on their type: `Table` is saved as CSV, `Contours` as a text file, etc.

```python
from microimage import ExperimentManager, Table, Contours
from dask.distributed import Client, LocalCluster

# Initialize the experiment
data_dir = "data/"
results_dir = "results/"
db_path = "project.db"
exp = ExperimentManager(data_dir, results_dir, db_path)

# Scan the data directory and register all scenes
exp.scan_data_directory([".czi", ".lsm", ".nd2"])

# Export a template annotation file, edit it, then re-import
exp.export_annotation("annotation.yml")
exp.import_annotation("annotation.yml")

# Visualize a scene in a notebook
item = exp.get_all_items()[0]
exp.show_image(item,["NUCLEI","MEMBRANE"],["blue","green"])
exp.show_scalebar()

# Define a processing function
def process(item, data):
    return {"table": Table(), "labels": Contours()}

# Run processing in parallel using Dask
cluster = LocalCluster(n_workers=4, threads_per_worker=1)
client = Client(cluster)
print(f"Dask dashboard: {client.dashboard_link}")

microimage.apply(exp, process, client)

# Inspect results
item = exp.get_all_items()[0]
exp.show_image(item,["NUCLEI","MEMBRANE"],["blue","green"])
exp.show_contours(item)
exp.show_scalebar()
```

## Supported File Formats

## Image file formats
A few microscopy format are supported. New readers can be registered if required.

| Format  | Read | Write |
|---------|------|-------|
| TIF     | ✓    |       |
| LSM     | ✓    |       |
| CZI     | ✓    |       |
| ND2     | ✓    |       |
| LIF     | ✓    |       |
| VSI     | ✓    |       |
| IMS     | ✓    |       |
| OME-ZARR|      |       |

## Results format
Results are serialized by default formats depending on their type when using `apply`. 

| Type     | Format | Read | Write |
|----------|--------|------|-------|
| Table    | CSV    | ✓    | ✓     |
| Contours | YOLO   | ✓    | ✓     |
| Labels   | TIF    | ✓    | ✓     |