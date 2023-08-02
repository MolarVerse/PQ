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
)

target_link_libraries(simulationBox
    PUBLIC
    vector3d
    config
)

install(TARGETS simulationBox
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)