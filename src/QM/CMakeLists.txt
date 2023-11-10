add_library(QM
    qmRunner.cpp

    dftbplusRunner.cpp
    turbomoleRunner.cpp
    pyscfRunner.cpp
)

target_include_directories(QM
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/QM
    ${PROJECT_SOURCE_DIR}/include/config
)

target_link_libraries(QM
    PUBLIC
    simulationBox
    settings
    utilities
    physicalData
)

set(BUILD_STATIC NOT BUILD_SHARED_LIBS)

target_compile_definitions(QM
    PUBLIC
    _SCRIPT_PATH_=\"${CMAKE_CURRENT_BINARY_DIR}/scripts/\"
    _SINGULARITY_=\"${BUILD_WITH_SINGULARITY}\"
    _STATIC_BUILD_=\"${BUILD_STATIC}\"
)

add_custom_command(TARGET QM POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/src/QM/scripts $<TARGET_FILE_DIR:QM>/scripts
)

install(TARGETS QM
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)