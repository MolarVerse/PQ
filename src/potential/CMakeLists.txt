add_library(potential
    potential.cpp
    potentialBruteForce.cpp
    potentialCellList.cpp
    potentialKokkos.cpp
)

add_dependencies(potential
    kokkos
)

target_include_directories(potential
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/potential
    ${Kokkos_INCLUDE_DIRS_RET}
)

target_link_libraries(potential
    PUBLIC
    physicalData
    simulationBox
    coulombPotential
    nonCoulombPotential
    kokkos
    Kokkos::kokkoskernels
)
install(TARGETS potential
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)

add_subdirectory(coulomb)
add_subdirectory(nonCoulomb)