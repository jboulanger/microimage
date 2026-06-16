TODO

## Remove multi-file parser
The multifile parser seems now redundant.


## Improve _process_container_file Logic:
In src/microimage/manager.py, the block:

```
if (
    scene_xarray is None
):  # Should not happen if ID was found, but for safety
    try:
        with Reader(container_path) as reader:
            scene_xarray = next(
                (
                    s
                    for i, s in enumerate(reader)
                    if str(i) == item_data["scene"]
                ),
                None,
            )
    except Exception:
        scene_xarray = None
```

This re-opens the reader and re-iterates through scenes if scene_xarray is None. Given that items_data_list is populated earlier with _scene_xarray directly from the reader, this if block seems redundant and potentially inefficient. The _scene_xarray should already be available from item_data. It might indicate an issue where _scene_xarray is not always correctly populated by the multi_file_parser or the default Reader path.

## ExperimentManager.add_file Method: 

- [x] The add_file method in src/microimage/manager.py is currently a pass statement. If this method is not intended for direct use or is a placeholder, it should either be implemented, removed, or clearly marked as a future feature.


## Database schema
- [ ] data table metadata and channels columns: Storing these as JSON strings offers flexibility but makes direct SQL querying difficult. If there are common metadata fields or channel properties that users might want to filter or group by, consider promoting them to dedicated columns in the data table.

- [x] annotation table PRIMARY KEY: The README.md states Primary key: (data_id). While manager.py's update_item uses ON CONFLICT(data_id) DO UPDATE, which implies data_id is unique, the CREATE TABLE IF NOT EXISTS annotations statement in _initialize_db does not explicitly define data_id as a PRIMARY KEY. It should be added for clarity and to enforce uniqueness at the database level.

- [] Provenance tracking (Section 2) is a good principle but currently has no implementation counterpart anywhere in the project documentation. It's worth either sketching how it would work in microimage (e.g. storing software version in the metadata column) or flagging it as a future concern, otherwise it risks being aspirational rather than actionable.