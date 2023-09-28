
add_library(input
    guffDatReader.cpp
    inputFileReader.cpp
    intraNonBondedReader.cpp
    moldescriptorReader.cpp
    commandLineArgs.cpp
)

target_include_directories(input
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/input
)

target_link_libraries(input
    PUBLIC
    engine
    inputFileParser
)

install(TARGETS input
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)

add_subdirectory(inputFileParser)
add_subdirectory(topologyFileReader)
add_subdirectory(restartFileReader)
add_subdirectory(parameterFileReader)
add_subdirectory(ringPolymerRestartFileReader)