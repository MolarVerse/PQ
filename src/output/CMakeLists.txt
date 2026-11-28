add_library(output
    output.cpp
    frame.cpp
    textBuffer.cpp
    extxyzOutput.cpp
    h5mdOutput.cpp
    aseTrajOutput.cpp
    zarrOutput.cpp
    zarrWriter.cpp
    atomicNumber.cpp
    ulmWriter.cpp
    outputMessages.cpp

    energyColumns.cpp
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

# h5mdOutput.cpp and hdf5Wrapper.cpp alone see HDF5: no PCH or unity batch, as its macro and
# helper names must not leak into the other sources
set_source_files_properties(h5mdOutput.cpp PROPERTIES
    SKIP_PRECOMPILE_HEADERS ON
    SKIP_UNITY_BUILD_INCLUSION ON
)

if(BUILD_WITH_HDF5)
    target_sources(output PRIVATE hdf5Wrapper.cpp)
    target_link_libraries(output PRIVATE HDF5::HDF5)
    set_source_files_properties(h5mdOutput.cpp PROPERTIES
        COMPILE_DEFINITIONS WITH_HDF5
    )
    set_source_files_properties(hdf5Wrapper.cpp PROPERTIES
        SKIP_PRECOMPILE_HEADERS ON
        SKIP_UNITY_BUILD_INCLUSION ON
    )
endif()

# zarrWriter.cpp alone sees zstd: no PCH or unity batch, like the HDF5 sources
if(BUILD_WITH_ZSTD)
    target_link_libraries(output PRIVATE zstd::libzstd)
    set_source_files_properties(zarrWriter.cpp PROPERTIES
        COMPILE_DEFINITIONS WITH_ZSTD
        SKIP_PRECOMPILE_HEADERS ON
        SKIP_UNITY_BUILD_INCLUSION ON
    )
endif()

install(TARGETS output
    DESTINATION lib
)

add_subdirectory(references)
