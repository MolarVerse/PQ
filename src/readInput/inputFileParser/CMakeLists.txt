
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
    inputFileParserQM.cpp
    inputFileParserResetKinetics.cpp
    inputFileParserRingPolymer.cpp
    inputFileParserSimulationBox.cpp
    inputFileParserThermostat.cpp
    inputFileParserTimings.cpp
    inputFileParserVirial.cpp
    inputFileParserFiles.cpp
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
    gtest
    forceField
    settings
)

install(TARGETS inputFileParser
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)