file(GLOB InputFileParserFiles ${CMAKE_CURRENT_SOURCE_DIR}/inputFileParsing/inputFileParser*.cpp)

add_library(readInput
    rstFileReader.cpp
    rstFileSection.cpp
    moldescriptorReader.cpp
    guffDatReader.cpp
    topologyReader.cpp
    topologySection.cpp
    parameterFileReader.cpp
    parameterFileSection.cpp

    inputFileReader.cpp
    ${InputFileParserFiles}
)

target_include_directories(readInput
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/readInput
)

target_link_libraries(readInput
    PUBLIC
    engine
    utilities
    output
    config
    gtest
    forceField
)

install(TARGETS readInput
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)