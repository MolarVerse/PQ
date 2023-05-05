add_library(config
    commandLineArgs.cpp
    exceptions.cpp
)

target_include_directories(config
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/include/config
)