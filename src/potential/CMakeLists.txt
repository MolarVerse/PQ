set(potential_source_files
    potential.cpp
    potential.tpp.cpp
    potentialBruteForce.cpp
    potentialCellList.cpp
)

add_library(potential
    ${potential_source_files}
)

target_include_directories(potential
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/potential
)

target_link_libraries(potential
    PUBLIC
    physicalData
    simulationBox
    coulombPotential
    nonCoulombPotential
    timings
)

install(TARGETS potential
    DESTINATION lib
)

add_subdirectory(coulomb)
add_subdirectory(nonCoulomb)
