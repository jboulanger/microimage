# Project Progress - microimage

## Overview
Status of the `microimage` library implementation according to the specifications in `AGENTS.md`.

## Core Features

### 1. Unified Image Reading (`src/microimage/readers.py`)
- [x] Abstract `BaseReader` interface.
- [x] Reader factory (`Reader`) with auto-detection.
- [x] Supported Formats:
    - [x] LSM (Zeiss)
    - [x] TIF/TIFF (Generic/OME)
    - [x] CZI (Zeiss)
    - [x] ND2 (Nikon)
    - [x] LIF (Leica)
    - [x] VSI (Olympus)
    - [x] IMS (Imaris)
- [x] Dask-backed lazy loading.
- [x] Metadata extraction (XYZ, time, channels).

### 2. Experiment Management (`src/microimage/manager.py`)
- [x] SQLite-based persistent storage.
- [x] `DataItem` for individual scene tracking.
- [x] Directory scanning for automatic file registration.
- [x] Metadata updates and retrieval.
- [x] Status reporting (processed, errors, unprocessed).

### 3. Processing Framework (`src/microimage/processor.py`)
- [x] Dask-compatible worker task wrapper.
- [x] Database synchronization from worker processes.
- [x] High-level pipeline orchestration in `ExperimentManager`.
- [x] Refactor `Processor` base class and `_apply_to_item`:
    - [x] Correct return type annotations on `Processor.process` to `Dict[str, Result]`.
    - [x] Add `validate(item, data)` hook to `Processor` for early data/channel validation.
    - [x] Add `get_parameters()` hook to `Processor` for automated provenance logging.
    - [x] Refactor `_apply_to_item` to exploit `Processor` validation, parameters persistence, and structured error reporting.
    - [x] Add unit tests covering `Processor` hooks and `_apply_to_item` refactor.

### 4. Writing (`src/microimage/writers.py`)
- [x] Abstract `BaseWriter` interface (Partially implemented).
- [x] Writer factory (`Writer`).
- [x] Concrete implementations (e.g., `TIFFWriter`, `CSVWriter`, `YOLOWriter`).

## Directory Structure & Tooling
- [x] `src/` based layout.
- [x] `uv` for environment management (`uv.lock` present).
- [x] `pytest` for testing (tests present in `test/`).
- [x] `examples/` directory for notebooks and scripts.
- [x] `ruff` for formatting (assumed usage per constraints).

## Use Cases Implementation Status
1. **Basic Usage**: **Completed**. `ExperimentManager` handles data/result folders, DB location, scanning, and metadata.
2. **Processing**: **In Progress**. Core worker logic exists, but high-level execution API is pending.

## Recent Changes
- Initial project structure setup.
- Comprehensive reader support for major microscopy formats.
- SQLite-backed experiment manager implementation.
- Basic Dask processing wrapper.

## Database schema
- [ ] **A1** data table metadata and channels columns: Storing these as JSON strings offers flexibility but makes direct SQL querying difficult. If there are common metadata fields or channel properties that users might want to filter or group by, consider promoting them to dedicated columns in the data table.

- [ ] **A2** Provenance tracking (Section 2) is a good principle but currently has no implementation counterpart anywhere in the project documentation. It's worth either sketching how it would work in microimage (e.g. storing software version in the metadata column) or flagging it as a future concern, otherwise it risks being aspirational rather than actionable. Should this be implemented as a separate table such as processing log or directly in the results table. We could record processing time stamps and function signature. The processing function are external to the package. whose responsability is it (microimage or other)?

- [x] **A3** Change the annotated channels to a dictionary as defined in the model

## Processing
- [ ] **B1** in processor.apply, add a summary of the number of sucessfuly processed items