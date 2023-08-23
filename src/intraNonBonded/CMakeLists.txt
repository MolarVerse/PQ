add_library(intraNonBonded
    intraNonBonded.cpp
    intraNonBondedMap.cpp
    intraNonBondedContainer.cpp
)

target_include_directories(intraNonBonded
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/intraNonBonded
)

target_link_libraries(intraNonBonded
    PUBLIC
    simulationBox
    physicalData
    forceField
    potential
    nonCoulombPotential
    coulombPotential
)

install(TARGETS intraNonBonded
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)