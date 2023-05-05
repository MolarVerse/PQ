add_library(engine
    settings.cpp
    timings.cpp
    jobtype.cpp
    engine.cpp
)

target_include_directories(engine
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/include/engine
)

target_link_libraries(engine
    PUBLIC
        output
        simulationBox
)