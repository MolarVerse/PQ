find_path(ZSTD_INCLUDE_DIR zstd.h REQUIRED)
find_library(ZSTD_LIBRARY NAMES zstd REQUIRED)
message(STATUS "zstd ${ZSTD_LIBRARY} for Zarr output")

add_library(zstd::libzstd UNKNOWN IMPORTED)
set_target_properties(zstd::libzstd PROPERTIES
    IMPORTED_LOCATION "${ZSTD_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${ZSTD_INCLUDE_DIR}"
)
