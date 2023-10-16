
add_library(pimd_qmcf_input
    guffDatReader.cpp
    inputFileReader.cpp
    intraNonBondedReader.cpp
    moldescriptorReader.cpp
    commandLineArgs.cpp
)

target_include_directories(pimd_qmcf_input
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/input
)

target_link_libraries(pimd_qmcf_input
    PUBLIC
    engine
    inputFileParser
)

install(TARGETS pimd_qmcf_input
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)

add_subdirectory(inputFileParser)
add_subdirectory(topologyFileReader)
add_subdirectory(restartFileReader)
add_subdirectory(parameterFileReader)
add_subdirectory(ringPolymerRestartFileReader)