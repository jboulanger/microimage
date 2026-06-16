# Software Development Principles

Welcome to the core development manifesto for the microimage project. Research Software Engineering (RSE) requires balancing rigorous engineering practices with the fluid, exploratory nature of scientific discovery. 

This document defines the core principles, environment constraints, and technical stack ensuring our software is scientifically sound, reproducible, and maintainable.

---

## 1. Project Stack
All AI assistants and developers must strictly adhere to the project's defined tech stack and version constraints. Do not introduce unlisted core dependencies without explicit confirmation.

### Tech Stack Matrix
| Component | Technology | Target Version | Purpose / Scope |
| :--- | :--- | :--- | :--- |
| **Runtime & Tooling** | Python via `uv` | `3.12.x` | Unified runtime, environment, and lockfile manager. |
| **Array** | xarray/dask | `^2025.11` | Multi-dimensional array |
| **Data Storage** | sqlite | `3` | Local database. |
| **Testing** | pytest | `^7.4` | Automated unit, integration, and regression tests. |

### Environment Setup & Agent Commands
AI agents must use `uv` syntax to execute project tasks without manual environment activation loops:

* **Initialize/Sync Project:** `uv sync`
* **Add a New Dependency:** `uv add <package>`
* **Run Test Suite:** `uv run pytest tests/`
* **Run Code Linter:** `uv run ruff check src/`

---

## 2. Reproducibility & Verifiability
In scientific software, correctness and reproducibility are paramount. A result that cannot be replicated is not a scientific result.

* **Deterministic Execution:** Avoid or carefully control sources of non-determinism (e.g., random number generator seeds, parallel race conditions). Always explicitly set seeds in numerical modules.
* **Environment As Code:** The exact execution environment must be versioned. 
* **Provenance Tracking:** Ensure output data products are automatically tagged with the software version, configuration parameters, and raw data inputs used to generate them.

---

## 3. Computational Correctness & Testing
Unlike commercial software where a bug might crash a page, a bug in scientific software can quietly corrupt data and lead to false scientific conclusions.

### Testing Scientific Code
* **Analytical Verification:** Test your algorithms against known analytical or exact solutions for simplified baseline cases.
* **Regression Testing for Data:** Maintain a suite of "golden data" inputs and outputs to ensure optimizations or refactors do not unexpectedly alter scientific results.
* **Edge-Case Validation:** Actively test bounds, zero-cases, and invalid physical parameters (e.g., negative mass or absolute temperature below $0\text{ K}$).
* **Test data strategy** Original test data in the `data` folder are used for testing.

| Test Category | Target | Objective |
| :--- | :--- | :--- |
| **Integration** | Data processing pipelines | Ensure smooth data flow from ingestion to output. |
| **Numerical Stability** | Floating-point operations | Detect precision loss, underflow, or overflow. |

---

## 4. Data Integrity & Interoperability
Data is the lifeblood of science. Our software must treat data with strict stewardship.

* **Open Standard Formats:** Favor widely accepted scientific formats (e.g., HDF5, NetCDF, Parquet, JSON) over proprietary or custom binary formats to avoid vendor lock-in.
* **Immutability:** Raw data inputs must be treated as strictly read-only. Intermediate and final outputs should be written to separate directories.
* **Metadata Richness:** Self-documenting datasets are preferred. We use xarray to insure that data axes are labelled.
* **Image I/O conventions** Microscopy data are normalized to TCZYX axes order. 
* **Database access** The access is coordinated by the experiment manager only.

---

## 5. Code Accessibility & Open Science (FAIR)
We align our development practices with the FAIR principles: **F**indable, **A**ccessible, **I**nteroperable, and **R**eusable.

* **Documentation is a Deliverable:** Code without documentation is unusable. Every core algorithm must document its assumptions, physical limitations, and citations to the original scientific literature.
* **Permissive Licensing:** Default to open-source licenses (e.g., MIT, BSD-3-Clause, or Apache 2.0) to encourage community collaboration and academic reuse.

---

## 6. Pragmatic Performance Optimization
Scientific computing often demands high performance, but premature optimization can introduce bugs and hurt readability.

* **Profile Before Optimizing:** Never optimize based on intuition. Use profiling tools (like `cProfile` or `line_profiler`) to identify actual computational bottlenecks.
* **Prototype First, Optimize Later:** Write clean, high-level code to verify scientific validity first. Delegate performance-critical bottlenecks to compiled languages or highly optimized libraries only when proven necessary.
* **Maintainability > Micro-optimizations:** Do not trade code clarity for negligible performance gains.

