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
    outputFilesSetup.cpp
)

target_include_directories(setup
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/setup
)

target_link_libraries(setup
    PUBLIC
    PQ_input
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

if(BUILD_WITH_MPI)
    target_link_libraries(setup
        PRIVATE
        PQ_mpi
    )
endif()

install(TARGETS setup
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)