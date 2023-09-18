add_library(setup
    setup.cpp
    simulationBoxSetup.cpp
    celllistSetup.cpp
    thermostatSetup.cpp
    manostatSetup.cpp
    resetKineticsSetup.cpp
    potentialSetup.cpp
    integratorSetup.cpp
    constraintsSetup.cpp
    forceFieldSetup.cpp
    intraNonBondedSetup.cpp
    qmSetup.cpp
    ringPolymerSetup.cpp
)

target_include_directories(setup
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/setup
)

target_link_libraries(setup
    PUBLIC
    readInput
    topologyFileReader
    restartFileReader
    parameterFileReader
    ringPolymerRestartFileReader

    engine
    utilities
    output
    gtest
    potential
    coulombPotential
    nonCoulombPotential
    maxwellBoltzmann
)

install(TARGETS setup
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)