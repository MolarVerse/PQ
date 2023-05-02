add_library(setup
    rstFileReader.cpp
    rstFileSection.cpp
)

target_include_directories(setup
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/include/setup
)

target_link_libraries(setup
    PUBLIC
        settings
        simulationBox
        utilities
)