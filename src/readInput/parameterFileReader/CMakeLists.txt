
add_library(parameterFileReader
    parameterFileReader.cpp
    parameterFileSection.cpp

    typesSection.cpp
    bondSection.cpp
    angleSection.cpp
    dihedralSection.cpp
    improperDihedralSection.cpp
    nonCoulombicsSection.cpp
)

target_include_directories(parameterFileReader
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/readInput/parameterFileReader
)

target_link_libraries(parameterFileReader
    PUBLIC
    engine
)

install(TARGETS parameterFileReader
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)