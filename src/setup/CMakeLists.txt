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
)

target_include_directories(setup
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include/setup
)

target_link_libraries(setup
    PUBLIC
    readInput
    engine
    utilities
    output
    config
    gtest
    math
)

install(TARGETS setup
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)