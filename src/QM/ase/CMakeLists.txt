add_library(aseQM
    aseQMRunner.cpp
    maceRunner.cpp
    fairchemRunner.cpp
    aseDftbRunner.cpp
)

target_include_directories(aseQM
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/QM
    ${PROJECT_SOURCE_DIR}/include/QM/ase
    ${PROJECT_SOURCE_DIR}/include/config

    ${pybind11_INCLUDE_DIRS}
)

target_link_libraries(aseQM
    PUBLIC
    simulationBox
    settings
    utilities
    physicalData
    exceptions

    QM

    pybind11::embed
    pybind11::module
)

target_include_directories(aseQM
    PUBLIC
    ${pybind11_INCLUDE_DIRS}
)

install(TARGETS aseQM
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)