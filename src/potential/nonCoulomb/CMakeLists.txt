add_library(nonCoulombPotential
    nonCoulombPair.cpp
    lennardJonesPair.cpp
    buckinghamPair.cpp
    morsePair.cpp
    guffPair.cpp

    nonCoulombPotential.cpp
    guffNonCoulomb.cpp
    forceFieldNonCoulomb.cpp
)

target_include_directories(nonCoulombPotential
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include/potential/nonCoulomb
)

target_link_libraries(nonCoulombPotential
    PUBLIC
    config
    utilities
    linearAlgebra
)

install(TARGETS nonCoulombPotential
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)