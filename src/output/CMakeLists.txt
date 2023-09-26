add_library(output
    output.cpp
    outputMessages.cpp

    energyOutput.cpp
    infoOutput.cpp
    trajectoryOutput.cpp
    logOutput.cpp
    stdoutOutput.cpp
    rstFileOutput.cpp
    ringPolymerRestartFileOutput.cpp
    ringPolymerTrajectoryOutput.cpp
    momentumOutput.cpp
)

target_include_directories(output
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/output
    ${PROJECT_SOURCE_DIR}/include/config
)

target_link_libraries(output
    PUBLIC
    utilities
    physicalData
    simulationBox
    settings
)

install(TARGETS output
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)