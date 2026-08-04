
add_library(ringPolymerRestartFileReader
    ringPolymerRestartFileReader.cpp
)

target_include_directories(ringPolymerRestartFileReader
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/input/ringPolymerRestartFileReader
)

target_link_libraries(ringPolymerRestartFileReader
    PUBLIC
    engine
    settings
    utilities
    exceptions
)

install(TARGETS ringPolymerRestartFileReader
    DESTINATION lib
)

target_precompile_headers(ringPolymerRestartFileReader REUSE_FROM pq_pch)
