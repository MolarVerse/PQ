
add_library(readInput
    guffDatReader.cpp
    inputFileReader.cpp
    intraNonBondedReader.cpp
    moldescriptorReader.cpp
    commandLineArgs.cpp
)

target_include_directories(readInput
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/readInput
)

target_link_libraries(readInput
    PUBLIC
    engine
    inputFileParser
)

install(TARGETS readInput
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)

add_subdirectory(inputFileParser)
add_subdirectory(topologyFileReader)
add_subdirectory(restartFileReader)
add_subdirectory(parameterFileReader)