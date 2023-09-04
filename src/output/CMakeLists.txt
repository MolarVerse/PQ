add_library(output
    output.cpp

    energyOutput.cpp
    infoOutput.cpp
    trajectoryOutput.cpp
    logOutput.cpp
    stdoutOutput.cpp
    rstFileOutput.cpp
)

target_include_directories(output
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/output
)

target_link_libraries(output
    PUBLIC
    utilities
    physicalData
    simulationBox
)

install(TARGETS output
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)