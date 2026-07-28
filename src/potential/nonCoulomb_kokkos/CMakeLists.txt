add_library(nonCoulombPotential_kokkos
    lennardJones_kokkos.cpp
)

target_include_directories(nonCoulombPotential_kokkos
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/potential/nonCoulomb_kokkos
    ${PROJECT_SOURCE_DIR}/include/potential/nonCoulomb
    ${Kokkos_INCLUDE_DIRS_RET}
)

target_link_libraries(nonCoulombPotential_kokkos
    PUBLIC
    linearAlgebra

    Kokkos::kokkos

    # Kokkos::kokkoskernels
)

install(TARGETS nonCoulombPotential_kokkos
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)