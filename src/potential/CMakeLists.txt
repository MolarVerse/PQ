set(potential_source_files
    potential.cpp
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
    linearAlgebra
)

install(TARGETS potential
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)

add_subdirectory(coulomb)
add_subdirectory(nonCoulomb)