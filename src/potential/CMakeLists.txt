add_library(potential
    potential.cpp
    potentialBruteForce.cpp
    potentialCellList.cpp
    potentialKokkos.cpp
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
)

if(BUILD_WITH_KOKKOS)
    # add_dependencies(potential
    # kokkos
    # )
    target_include_directories(potential
        PUBLIC
        ${Kokkos_INCLUDE_DIRS_RET}
    )

    target_link_libraries(potential
        PUBLIC
        Kokkos::kokkos
        Kokkos::kokkoskernels
    )
endif()

install(TARGETS potential
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)

add_subdirectory(coulomb)
add_subdirectory(nonCoulomb)

if(BUILD_WITH_KOKKOS)
    add_subdirectory(nonCoulomb_kokkos)
endif()