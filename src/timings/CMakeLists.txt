add_library(timings
    timings.cpp
)

target_include_directories(timings
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include/timings
)

target_link_libraries(timings
    PUBLIC
    config
)