add_library(output
    output.cpp
    outputMessages.cpp

    energyOutput.cpp
    infoOutput.cpp
    trajectoryOutput.cpp
    logOutput.cpp
    stdoutOutput.cpp
    rstFileOutput.cpp
    momentumOutput.cpp

    virialOutput.cpp
    stressOutput.cpp
    boxOutput.cpp

    ringPolymerRestartFileOutput.cpp
    ringPolymerTrajectoryOutput.cpp
    ringPolymerEnergyOutput.cpp

    optOutput.cpp

    timingsOutput.cpp
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
    optimization
    settings
    thermostat
)

install(TARGETS output
    DESTINATION lib
)

add_subdirectory(references)
