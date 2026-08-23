add_library(aseQM
    aseQMRunner.cpp
    aseMaceRunner.cpp
    aseDftbRunner.cpp
    aseXtbRunner.cpp
    aseFennolRunner.cpp
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
    molsys
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

target_precompile_headers(aseQM PRIVATE
    <pybind11/embed.h>
    <pybind11/numpy.h>
    <pybind11/pybind11.h>
    <pybind11/stl.h>
)

install(TARGETS aseQM
    DESTINATION lib
)
