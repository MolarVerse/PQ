set(engine_source_files
    engine.cpp

    optEngine.cpp
    hessianEngine.cpp
    mdEngine.cpp

    mmmdEngine.cpp
    qmmdEngine.cpp
    qmCapableEngine.cpp
    qmRunnerManager.cpp
    ringPolymerEngine.cpp
    ringPolymerqmmdEngine.cpp

    hybridMDEngine.cpp
    qmmmMDEngine.cpp

    engineOutput.cpp
)

add_library(engine
    ${engine_source_files}
)

target_include_directories(engine
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/engine
    ${PROJECT_SOURCE_DIR}/include/waterModels/intra
    ${PROJECT_SOURCE_DIR}/include/waterModels/inter
    PRIVATE
    ${PROJECT_SOURCE_DIR}/external/progressbar/include
)

if(BUILD_WITH_MPI)
    target_link_libraries(engine
        PRIVATE
        MPI::MPI_C
        PQ_mpi
    )
endif()

target_link_libraries(engine
    PUBLIC
    constraints
    integrator
    manostat
    potential
    resetKinetics
    settings
    thermostat
    output
    simulationBox
    timings
    physicalData
    virial
    forceField
    intraNonBonded
    interWater
    references
    hybridConfigurator

    optimization
    QM
    externalQM
)

if(BUILD_WITH_ASE)
    target_link_libraries(engine
        PUBLIC
        aseQM
    )
endif()

if(BUILD_WITH_PYBIND11)
    target_include_directories(engine
        PRIVATE
        ${pybind11_INCLUDE_DIR}
    )
    target_link_libraries(engine
        PRIVATE
        pybind11::embed
    )
endif()

install(TARGETS engine
    DESTINATION lib
)

target_precompile_headers(engine REUSE_FROM pq_pch)
