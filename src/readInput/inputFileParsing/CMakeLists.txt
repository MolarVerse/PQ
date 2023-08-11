
add_library(inputFileParser
    inputFileParser.cpp
    inputFileParserCelllist.cpp
    inputFileParserConstraints.cpp
    inputFileParserCoulombLongRange.cpp
    inputFileParserForceField.cpp
    inputFileParserGeneral.cpp
    inputFileParserIntegrator.cpp
    inputFileParserManostat.cpp
    inputFileParserNonCoulomb.cpp
    inputFileParserOutput.cpp
    inputFileParserParameterFile.cpp
    inputFileParserResetKinetics.cpp
    inputFileParserSimulationBox.cpp
    inputFileParserThermostat.cpp
    inputFileParserTimings.cpp
    inputFileParserTopology.cpp
    inputFileParserVirial.cpp
)

target_include_directories(inputFileParser
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/readInput
    ${PROJECT_SOURCE_DIR}/include/readInput/inputFileParser/
)

target_link_libraries(inputFileParser
    PUBLIC
    engine
    utilities
    output
    config
    gtest
    forceField
)

install(TARGETS inputFileParser
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)