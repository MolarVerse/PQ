set(SRC_FILES
    qmRunner.cpp
    qmRunnerFactory.cpp

    external/externalQMRunner.cpp
    external/pyscfRunner.cpp
    external/turbomoleRunner.cpp
    external/dftbplusRunner.cpp
)

if(BUILD_WITH_ASE)
    list(APPEND SRC_FILES
        ase/aseQMRunner.cpp
        ase/aseFennolRunner.cpp
        ase/aseXtbRunner.cpp
        ase/aseMaceRunner.cpp
        ase/aseDftbRunner.cpp
    )
endif()

add_library(QM
    ${SRC_FILES}
)

target_include_directories(QM
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/QM
    ${PROJECT_SOURCE_DIR}/include/QM/external
    ${PROJECT_SOURCE_DIR}/include/QM/ase
)

target_link_libraries(QM
    PUBLIC
    config_headers
    molsys
    settings
    utilities
    physicalData
    exceptions
    hybridConfigurator
)

if(BUILD_WITH_ASE)
    target_link_libraries(QM
        PUBLIC
        pybind11::embed
        pybind11::module
    )

    target_include_directories(QM
        PUBLIC
        ${pybind11_INCLUDE_DIRS}
    )

    target_precompile_headers(QM PRIVATE
        <pybind11/embed.h>
        <pybind11/numpy.h>
        <pybind11/pybind11.h>
        <pybind11/stl.h>
    )
endif()

add_custom_command(TARGET QM POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/src/QM/scripts $<TARGET_FILE_DIR:QM>/scripts
)

install(TARGETS QM
    DESTINATION lib
)

install(
    DIRECTORY ${CMAKE_SOURCE_DIR}/src/QM/scripts/
    DESTINATION share/PQ/scripts
    USE_SOURCE_PERMISSIONS
    COMPONENT qm-scripts
)
