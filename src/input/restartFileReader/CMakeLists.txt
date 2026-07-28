
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

if(BUILD_WITH_TESTS)
    target_link_libraries(restartFileReader
        PUBLIC
        gtest
    )
endif()

install(TARGETS restartFileReader
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)