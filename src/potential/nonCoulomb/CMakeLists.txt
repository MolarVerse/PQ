set(nonCoulombPotentialSources
    nonCoulombPair.cpp
    lennardJonesPair.cpp
    buckinghamPair.cpp
    morsePair.cpp
    guffPair.cpp

    nonCoulombPotential.cpp
    guffNonCoulomb.cpp
    forceFieldNonCoulomb.cpp
)

if(NOT BUILD_LEGACY)
    list(APPEND nonCoulombPotentialSources
        lennardJones.cpp
    )
endif()

add_library(nonCoulombPotential
    ${nonCoulombPotentialSources}
)

target_include_directories(nonCoulombPotential
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/potential/nonCoulomb
    ${PROJECT_SOURCE_DIR}/include/linearAlgebra
)

target_link_libraries(nonCoulombPotential
    PUBLIC
    utilities
    linearAlgebra
    settings
)

install(TARGETS nonCoulombPotential
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)