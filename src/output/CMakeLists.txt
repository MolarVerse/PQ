add_library(output
    output.cpp
)

target_include_directories(output
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/include/output
)