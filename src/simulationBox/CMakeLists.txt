set(simulationBox_source_files
    simulationBox.cpp
    simulationBox_standardMethods.cpp
    simulationBox_API.cpp
    simulationBox_base.cpp
    simulationBox_SoA.cpp
    coordinates.cpp

    molecule.cpp
    moleculeType.cpp
    celllist.cpp
    cell.cpp
    atom.cpp
)

if(NOT BUILD_LEGACY)
    set(simulationBox_source_files
        ${simulationBox_source_files}
        simulationBox_nonLegacy.cpp
    )
else()
    set(simulationBox_source_files
        ${simulationBox_source_files}
        simulationBox_legacy.cpp
    )
endif()

if(BUILD_WITH_MPI)
    set(simulationBox_source_files
        ${simulationBox_source_files}
        simulationBox_MPI.cpp
    )
endif()

if(BUILD_WITH_GPU)
    set(simulationBox_source_files
        ${simulationBox_source_files}
        simulationBox_GPU.cpp
        simulationBox_SoA_GPU.cpp
        coordinates_GPU.cpp
    )
endif()

add_library(simulationBox
    ${simulationBox_source_files}
)

target_include_directories(simulationBox
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/simulationBox
    ${PROJECT_SOURCE_DIR}/include/config
    ${PROJECT_SOURCE_DIR}/include/device
)

target_link_libraries(simulationBox
    PUBLIC
    linearAlgebra
    box
    exceptions
    settings
    timings
)

if(BUILD_WITH_GPU)
    target_link_libraries(simulationBox
        PUBLIC
        device
    )
endif()

install(TARGETS simulationBox
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)