
add_library(inputFileParser
    inputFileParser.cpp
    inputFileParserCelllist.cpp
    inputFileParserConstraints.cpp
    inputFileParserCoulombLongRange.cpp
    inputFileParserForceField.cpp
    inputFileParserIntegrator.cpp
    inputFileParserManostat.cpp
    inputFileParserNonCoulomb.cpp
    inputFileParserOutput.cpp
    inputFileParserQMMM.cpp
    inputFileParserResetKinetics.cpp
    inputFileParserRingPolymer.cpp
    inputFileParserSimulationBox.cpp
    inputFileParserThermostat.cpp
    inputFileParserTimings.cpp
    inputFileParserVirial.cpp
    inputFileParserFiles.cpp

    convergenceInputParser.cpp
    generalInputParser.cpp
    optInputParser.cpp
    QMInputParser.cpp
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

install(TARGETS inputFileParser
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)