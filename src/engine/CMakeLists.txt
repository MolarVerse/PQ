add_library(engine
    engine.cpp

    mmmdEngine.cpp
    qmmdEngine.cpp
    ringPolymerEngine.cpp
    ringPolymerqmmdEngine.cpp

    engineOutput.cpp
)

target_include_directories(engine
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/engine
    PRIVATE
    ${PROJECT_SOURCE_DIR}/external/progressbar/include
)

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
    QM
)

install(TARGETS engine
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)