
add_library(readInput
    guffDatReader.cpp
    inputFileReader.cpp
    intraNonBondedReader.cpp
    moldescriptorReader.cpp
    parameterFileReader.cpp
    parameterFileSection.cpp
    rstFileReader.cpp
    rstFileSection.cpp
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
    inputFileParser
    potential
    nonCoulombPotential
)

install(TARGETS readInput
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)

add_subdirectory(inputFileParsing)
add_subdirectory(topologyFileReader)