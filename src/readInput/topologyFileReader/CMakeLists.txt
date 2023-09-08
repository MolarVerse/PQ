
add_library(topologyFileReader
    topologyReader.cpp
    topologySection.cpp
    shakeSection.cpp
    bondSection.cpp
    angleSection.cpp
    dihedralSection.cpp
    improperDihedralSection.cpp
)

target_include_directories(topologyFileReader
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/readInput/topologyFileReader
)

target_link_libraries(topologyFileReader
    PUBLIC
    forceField
    constraints
    engine
)

install(TARGETS topologyFileReader
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)