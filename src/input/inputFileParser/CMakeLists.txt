
add_library(inputFileParser
    inputFileParser.cpp

    cellListInputParser.cpp
    constraintsInputParser.cpp
    convergenceInputParser.cpp
    coulombLongRangeInputParser.cpp
    filesInputParser.cpp
    forceFieldInputParser.cpp
    generalInputParser.cpp
    hybridInputParser.cpp
    integratorInputParser.cpp
    manostatInputParser.cpp
    nonCoulombInputParser.cpp
    outputInputParser.cpp
    optInputParser.cpp
    QMInputParser.cpp
    resetKineticsInputParser.cpp
    ringPolymerInputParser.cpp
    simulationBoxInputParser.cpp
    thermostatInputParser.cpp
    timingsInputParser.cpp
    virialInputParser.cpp
)

target_include_directories(inputFileParser
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/input
    ${PROJECT_SOURCE_DIR}/include/input/inputFileParser/
)

target_link_libraries(inputFileParser
    PUBLIC
    engine
    utilities
    output
    forceField
    settings
    references
)

if(BUILD_WITH_TESTS)
    target_link_libraries(inputFileParser
        PUBLIC
        gtest
    )
endif()

if(BUILD_WITH_PYTHON_BINDINGS)
    add_definitions(-DPYTHON_ENABLED)

    target_link_libraries(setup
        PUBLIC
        pq_python
    )
endif()

install(TARGETS inputFileParser
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)