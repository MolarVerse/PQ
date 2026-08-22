add_library(config_headers 
    INTERFACE
)

target_link_libraries(config_headers
    INTERFACE
    utilities
)

target_include_directories(config_headers
    INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include/config
)
