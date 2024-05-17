add_library(nonCoulombPotential_kokkos
    lennardJones_kokkos.cpp
)

target_include_directories(nonCoulombPotential_kokkos
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/potential/nonCoulomb_kokkos
)

target_link_libraries(nonCoulombPotential_kokkos
    PUBLIC
)

install(TARGETS nonCoulombPotential_kokkos
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)