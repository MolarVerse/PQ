add_library(settings
    settings.cpp
)

target_include_directories(settings
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/include/settings
)