add_library(setup
    setup.cpp

    rstFileReader.cpp
    rstFileSection.cpp

    inputFileReader.cpp

    inputFileParsing/inputFileParser.cpp
    inputFileParsing/inputFileParserOutput.cpp
    inputFileParsing/inputFileParserTimings.cpp
    inputFileParsing/inputFileParserGeneral.cpp
    inputFileParsing/inputFileParserIntegrator.cpp
    inputFileParsing/inputFileParserBox.cpp
    inputFileParsing/inputFileParserVirial.cpp
    inputFileParsing/inputFileParserSimulationBox.cpp
    inputFileParsing/inputFileParserCelllist.cpp
    inputFileParsing/inputFileParserThermostat.cpp
    inputFileParsing/inputFileParserManostat.cpp
    inputFileParsing/inputFileParserReset.cpp

    moldescriptorReader.cpp
    postProcessSetup.cpp

    guffDatReader.cpp
)

target_include_directories(setup
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include/setup
)

target_link_libraries(setup
    PUBLIC
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