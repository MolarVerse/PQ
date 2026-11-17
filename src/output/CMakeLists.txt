add_library(output
    output.cpp
    frame.cpp
    textBuffer.cpp
    extxyzOutput.cpp
    h5mdOutput.cpp
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
)

target_link_libraries(output
    PUBLIC
    config_headers
    utilities
    hybridConfigurator
    physicalData
    molsys
    optimization
    settings
    thermostat
)

target_precompile_headers(output REUSE_FROM pq_pch)

# h5mdOutput.cpp alone sees HDF5: no PCH or unity batch, as its macro and
# helper names must not leak into the other sources
set_source_files_properties(h5mdOutput.cpp PROPERTIES
    SKIP_PRECOMPILE_HEADERS ON
    SKIP_UNITY_BUILD_INCLUSION ON
)

if(BUILD_WITH_HDF5)
    target_link_libraries(output PRIVATE HDF5::HDF5)
    set_source_files_properties(h5mdOutput.cpp PROPERTIES
        COMPILE_DEFINITIONS WITH_HDF5
    )
endif()

install(TARGETS output
    DESTINATION lib
)

add_subdirectory(references)
