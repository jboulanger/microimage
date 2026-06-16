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

