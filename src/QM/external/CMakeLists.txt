add_library(externalQM
    externalQMRunner.cpp

    dftbplusRunner.cpp
    turbomoleRunner.cpp
    pyscfRunner.cpp
)

target_include_directories(externalQM
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/QM
    ${PROJECT_SOURCE_DIR}/include/QM/external
    ${PROJECT_SOURCE_DIR}/include/config
)

target_link_libraries(externalQM
    PUBLIC
    simulationBox
    settings
    utilities
    physicalData
    exceptions
    hybridConfigurator
    QM
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

add_custom_command(TARGET externalQM POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/src/QM/scripts $<TARGET_FILE_DIR:QM>/scripts
)

install(TARGETS externalQM
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)