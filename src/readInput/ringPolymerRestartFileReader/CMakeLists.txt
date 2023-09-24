
add_library(ringPolymerRestartFileReader
    ringPolymerRestartFileReader.cpp
)

target_include_directories(ringPolymerRestartFileReader
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/readInput/ringPolymerRestartFileReader
)

target_link_libraries(ringPolymerRestartFileReader
    PUBLIC
    engine
    settings
    utilities
    exceptions
)

install(TARGETS ringPolymerRestartFileReader
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)