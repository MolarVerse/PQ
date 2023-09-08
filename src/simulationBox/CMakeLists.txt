add_library(simulationBox
    box.cpp
    simulationBox.cpp
    molecule.cpp
    celllist.cpp
    cell.cpp
)

target_include_directories(simulationBox
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/simulationBox
    ${PROJECT_SOURCE_DIR}/include/config
)

target_link_libraries(simulationBox
    PUBLIC
    linearAlgebra
    exceptions
)

install(TARGETS simulationBox
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)