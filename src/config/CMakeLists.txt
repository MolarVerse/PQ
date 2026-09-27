add_library(config_headers 
    enums/manostat.cpp
)

target_link_libraries(config_headers
    PUBLIC
    utilities
)

target_include_directories(config_headers
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include/config
)
