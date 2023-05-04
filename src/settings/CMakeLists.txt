add_library(settings
    settings.cpp
    timings.cpp
    jobtype.cpp
)

target_include_directories(settings
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/include/settings
)

target_link_libraries(settings
    PUBLIC
        output
)