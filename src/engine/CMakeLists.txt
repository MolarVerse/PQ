add_library(engine
    engine.cpp
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
    potential_new

    resetKinetics
    settings
    thermostat
    output
    simulationBox
    config
    timings
    physicalData
    virial
    forceField
    intraNonBonded
)

install(TARGETS engine
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)