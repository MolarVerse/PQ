- introduce new class `ExternalQMSettings` for the external QM script (`qm_script`, `qm_script_full_path`); the script path is resolved once in `finalize()` and the build-time script directory/singularity/static flags now live in the `settings` library.
- introduce non-static `NoseHooverSettings` class
- remove static access of some `FileSettings` members
- `SelectionTag` carries an unresolved `expression`; `HybridSettings::finalize(const FileSettings&)` (called from `Settings::finalize()`) resolves it once the start/moldescriptor files are known.
- `FileSettings` start file and ring-polymer start file are now non-static `Setting` members (`getStartFileName()`, `setStartFileNameByUser()`, `isRingPolymerStartFileNameSetByUser()`, ...); the static ring-polymer start file flag is removed.

