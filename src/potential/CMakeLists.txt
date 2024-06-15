set(potential_source_files
    potential.cpp
    potentialBruteForce.cpp
    potentialCellList.cpp
)

if(BUILD_WITH_KOKKOS)
    set(potential_source_files
        ${potential_source_files}
        potential_kokkos.cpp
    )
endif()

if(BUILD_WITH_CUDA)
    set(potential_source_files
        ${potential_source_files}
        potential_cuda.cu
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
    target_include_directories(potential
        PUBLIC
        ${Kokkos_INCLUDE_DIRS_RET}
    )

    target_link_libraries(potential
        PUBLIC
        nonCoulombPotential_kokkos
        coulombPotential_kokkos
        Kokkos::kokkos
    )
endif()

if(BUILD_WITH_CUDA)
    target_link_libraries(potential
        PUBLIC
        ${CUDA_LIBRARIES})
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