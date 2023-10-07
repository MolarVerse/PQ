add_library(simulationBox
    simulationBox.cpp
    molecule.cpp
    moleculeType.cpp
    celllist.cpp
    cell.cpp
    atom.cpp
)

target_include_directories(simulationBox
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/simulationBox
    ${PROJECT_SOURCE_DIR}/include/config
)

target_link_libraries(simulationBox
    PUBLIC
    linearAlgebra
    box
    exceptions
    settings
)

install(TARGETS simulationBox
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)