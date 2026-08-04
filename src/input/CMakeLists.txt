
add_library(PQ_input
    guffDatReader.cpp
    inputFileReader.cpp
    inputValidation.cpp
    intraNonBondedReader.cpp
    moldescriptorReader.cpp
    mShakeReader.cpp
    commandLineArgs.cpp
)

target_include_directories(PQ_input
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/input
)

target_link_libraries(PQ_input
    PUBLIC
    engine
    inputFileParser
)

install(TARGETS PQ_input
    DESTINATION lib
)
target_precompile_headers(PQ_input REUSE_FROM pq_pch)

add_subdirectory(inputFileParser)
add_subdirectory(topologyFileReader)
add_subdirectory(restartFileReader)
add_subdirectory(parameterFileReader)
add_subdirectory(ringPolymerRestartFileReader)
