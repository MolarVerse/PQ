add_library(potential
    potential.cpp
)

target_include_directories(potential
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include/potential
)

target_link_libraries(potential
    PUBLIC
    output
    simulationBox
    MPI_C
)