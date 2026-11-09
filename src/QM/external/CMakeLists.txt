add_library(externalQM
    externalQMRunner.cpp

    dftbplusRunner.cpp
    turbomoleRunner.cpp
    orcaRunner.cpp
    pyscfRunner.cpp
)

target_include_directories(externalQM
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/QM
    ${PROJECT_SOURCE_DIR}/include/QM/external
)

target_link_libraries(externalQM
    PUBLIC
    molsys
    settings
    utilities
    physicalData
    exceptions
    hybridConfigurator
    QM
    config_headers
)

if(BUILD_SHARED_LIBS)
    set(BUILD_STATIC "OFF")
else()
    set(BUILD_STATIC "ON")
endif()

target_compile_definitions(externalQM
    PUBLIC
    _SCRIPT_PATH_=\"${CMAKE_CURRENT_BINARY_DIR}/../scripts/\"
    _SINGULARITY_=\"${BUILD_WITH_SINGULARITY}\"
    _STATIC_BUILD_=\"${BUILD_STATIC}\"
)

# Refresh Python entry points and their helpers even on script-only changes.
add_custom_target(copyExternalQMScripts
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/src/QM/scripts $<TARGET_FILE_DIR:QM>/scripts
    COMMENT "Copy external QM scripts"
)
add_dependencies(externalQM copyExternalQMScripts)

install(TARGETS externalQM
    DESTINATION lib
)

install(
    DIRECTORY ${CMAKE_SOURCE_DIR}/src/QM/scripts/
    DESTINATION share/PQ/scripts
    USE_SOURCE_PERMISSIONS
    COMPONENT qm-scripts
)
