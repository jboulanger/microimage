# TODO

## Database schema
- [ ] data table metadata and channels columns: Storing these as JSON strings offers flexibility but makes direct SQL querying difficult. If there are common metadata fields or channel properties that users might want to filter or group by, consider promoting them to dedicated columns in the data table.

- [] Provenance tracking (Section 2) is a good principle but currently has no implementation counterpart anywhere in the project documentation. It's worth either sketching how it would work in microimage (e.g. storing software version in the metadata column) or flagging it as a future concern, otherwise it risks being aspirational rather than actionable. Should this be implemented as a separate table such as processing log or directly in the results table. We could record processing time stamps and function signature. The processing function are external to the package. whose responsability is it (microimage or other)?

- [] Change the annotated channels to a dictionary as defined in the model
    