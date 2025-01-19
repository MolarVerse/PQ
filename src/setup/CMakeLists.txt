set(setup_source_files
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
    hybridSetup.cpp
    ringPolymerSetup.cpp
    outputFilesSetup.cpp
    optimizerSetup.cpp
)

if(NOT BUILD_LEGACY)
    set(setup_source_files
        ${setup_source_files}
        setup_nonLegacy.cpp
    )
endif()

add_library(setup
    ${setup_source_files}
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
    potential
    coulombPotential
    nonCoulombPotential
    maxwellBoltzmann
    optimization
)

if(BUILD_WITH_TESTS)
    target_link_libraries(setup
        PUBLIC
        gtest
    )
endif()

if(BUILD_WITH_MPI)
    target_link_libraries(setup
        PRIVATE
        PQ_mpi
    )
endif()

if(BUILD_WITH_PYTHON_BINDINGS)
    add_definitions(-DPYTHON_ENABLED)

    target_link_libraries(setup
        PUBLIC
        pq_python
    )
endif()

install(TARGETS setup
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)
