add_library(output
    output.cpp
    outputData.cpp
)

target_include_directories(output
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include/output
)

target_link_libraries(output
    PUBLIC
    config
    utilities
)