
add_library(restartFileReader
    restartFileReader.cpp
    restartFileSection.cpp

    boxSection.cpp
    atomSection.cpp
    noseHooverSection.cpp
    stepCountSection.cpp
)

target_include_directories(restartFileReader
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/input/restartFileReader
)

target_link_libraries(restartFileReader
    PUBLIC
    engine
    linearAlgebra
)

install(TARGETS restartFileReader
    DESTINATION lib
)
