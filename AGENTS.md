## Project Overview
Python file management tool high-throughput microscopy pipelines where standardizing metadata and handling large multi-dimensional images are critical.

## Code Conventions
- numpy docstring documentation
- src based project

## Directory Structure
- src/microimage/ — python module
- test — shared UI components
- examples — use cases notebooks and example scripts

## Important Constraints
- Write code that is straightforward to read and understand, prioritizing clarity
- Only build features that are required right now 
- Break the code into decoupled, cohesive components

## Tooling
- use ruff for code formating
- use uv for managing environment
- use pytest for testing

## Use cases
1. Basic usage
    - Define a data folder (containing image files) a result folder and a database location. 
    - List files in the data folder 
    - Add user defined metadata: conditions and channel semantic mapping is important
2. Processing
    - Load an experiment database
    - Process all items with a user defined processing callable.

## Database Schema
The project uses a SQLite database to track images, metadata, and processing results.

### `data` Table
Stores core information and dimensional metadata for each scene.

| Column | Type | Description |
| :--- | :--- | :--- |
| `id` | TEXT | Primary Key. Unique SHA1 hash based on file name, modification time, and scene index. |
| `relative_path` | TEXT | Path to the container file relative to the data directory. |
| `scene` | TEXT | Index or name of the scene within the container file. |
| `metadata` | TEXT | JSON string of arbitrary metadata extracted from the file. |
| `processed` | INTEGER | Boolean flag (0 or 1) indicating if the item has been processed. |
| `error` | TEXT | Error message if processing failed. |
| `size_t, size_c, ...` | INTEGER | Dimensional sizes (T, C, Z, Y, X). |
| `scale_t, scale_z, ...` | REAL | Physical scaling/calibration for dimensions. |
| `channels` | TEXT | JSON list of channel names. |

### `results` Table
Tracks output files generated during processing.

| Column | Type | Description |
| :--- | :--- | :--- |
| `data_id` | TEXT | Foreign Key referencing `data.id`. |
| `result_type` | TEXT | Identifier for the type of result (e.g., "thumbnail", "mask"). |
| `relative_path` | TEXT | Path to the result file relative to the project root. |

*Primary Key is a composite of `(data_id, result_type)`.*
