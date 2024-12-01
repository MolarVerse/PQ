set(potential_source_files
    potential.cpp
    potentialCellList.cpp
)

if(BUILD_LEGACY)
    set(potential_source_files
        ${potential_source_files}
        potential_legacy.cpp
        potentialBruteForce_legacy.cpp
    )
else()
    set(potential_source_files
        ${potential_source_files}
        potential_legacy.cpp # remove this one - atm needed for celllist
        potential_nonLegacy.cpp
        potentialBruteForce_nonLegacy.cpp
        # potentialBruteForce_legacy.cpp
    )
endif()


if(BUILD_WITH_KOKKOS)
    set(potential_source_files
        ${potential_source_files}
        potential_kokkos.cpp
    )
endif()

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
        nonCoulombPotential_kokkos
        coulombPotential_kokkos
        Kokkos::kokkos

        # Kokkos::kokkoskernels
    )
endif()

install(TARGETS potential
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)

add_subdirectory(coulomb)
add_subdirectory(nonCoulomb)

if(BUILD_WITH_KOKKOS)
    add_subdirectory(nonCoulomb_kokkos)
    add_subdirectory(coulomb_kokkos)
endif()