set(source_files
    qmRunner.cpp

    externalQMRunner.cpp
    internalQMRunner.cpp

    dftbplusRunner.cpp
    turbomoleRunner.cpp
    pyscfRunner.cpp
)

if(BUILD_WITH_MACE)
    set(source_files
        ${source_files}
        maceRunner.cpp
    )
endif()

add_library(QM
    ${source_files}
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
    exceptions
)

if(BUILD_WITH_MACE)
    target_include_directories(QM
        PUBLIC
        ${pybind11_INCLUDE_DIRS}
    )

    target_link_libraries(QM
        PUBLIC
        pybind11::embed
        pybind11::module
    )
endif()

if(BUILD_SHARED_LIBS)
    set(BUILD_STATIC "OFF")
else()
    set(BUILD_STATIC "ON")
endif()

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