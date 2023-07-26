file(GLOB InputFileParserFiles ${CMAKE_CURRENT_SOURCE_DIR}/inputFileParsing/inputFileParser*.cpp)

add_library(setup
    setup.cpp
    rstFileReader.cpp
    rstFileSection.cpp
    moldescriptorReader.cpp
    postProcessSetup.cpp
    guffDatReader.cpp
    topologyReader.cpp
    topologySection.cpp

    inputFileReader.cpp
    ${InputFileParserFiles}
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