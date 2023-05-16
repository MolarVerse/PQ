add_library(simulationBox
    box.cpp
    simulationBox.cpp
    molecule.cpp
    celllist.cpp
    cell.cpp
)

target_include_directories(simulationBox
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/include/simulationBox
)

target_link_libraries(simulationBox
    PUBLIC
        config
)