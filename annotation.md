# Plan: CSV-based Semantic Channel & Condition Annotation

## TL;DR: Add CSV export/import functionality to allow users to:

Export all items to a CSV file with columns: id, relative_image_path, scene, condition, channels_json
Annotate channels (rename, add stain/marker/target metadata) and conditions in the CSV
Import the CSV back to update the database with these annotations
Store channel metadata as nested JSON in item metadata (flexible, no schema migration needed)
Steps

## Phase 1: Data Model

Create ChannelAnnotation dataclass with fields: name, stain, marker, target
Update DataItem to store channel annotations in metadata as: {"channels_meta": [{"name": "GFP", "stain": "GFP", "marker": "actin", "target": "..."}, ...]}
Update DataItem.to_db_row() and from_db_row() to handle channel metadata serialization

##  Phase 2: Export Functionality
4. Add ExperimentManager.export_to_csv(output_path: Path) that:

Writes each item as one row with: id, relative_image_path, scene, condition, channels_json
Serializes channel annotations to JSON string (or use semicolon-delimited format if preferred)
Example row: image.czi:0 | data/image.czi | 0 | control | [{"name":"GFP","stain":"GFP","marker":"actin"},{"name":"DAPI","stain":"DAPI","marker":"dna"}]

##  Phase 3: Import Functionality
5. Add ExperimentManager.import_from_csv(csv_path: Path) that:

Reads CSV rows
For each row: parse channels JSON, deserialize to ChannelAnnotation objects
Update item's metadata with condition and channels_meta
Skip/warn on invalid rows (missing id, bad JSON)
Commit all updates to database

## Phase 4: Testing
6. Test export: verify CSV structure, valid JSON serialization
7. Test import: verify DB updates, error handling for malformed rows
8. Test round-trip: export → modify → import → verify state matches


## Phase 5: Example & Documentation
9. Create example notebook showing full workflow: scan → export → annotate CSV → import