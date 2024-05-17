add_library(coulombPotential_kokkos
    coulombWolf_kokkos.cpp
)

target_include_directories(coulombPotential_kokkos
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/potential/coulomb_kokkos
    ${PROJECT_SOURCE_DIR}/include/potential/coulomb
    ${Kokkos_INCLUDE_DIRS_RET}
)

target_link_libraries(coulombPotential_kokkos
    PUBLIC
    linearAlgebra
    Kokkos::kokkos
    Kokkos::kokkoskernels
)

install(TARGETS coulombPotential_kokkos
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)