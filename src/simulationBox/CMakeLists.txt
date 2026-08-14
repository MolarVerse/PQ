set(simulationBox_source_files
    simulationBox.cpp
    simulationBox_standardMethods.cpp

    molecule.cpp
    moleculeType.cpp
    celllist.cpp
    cell.cpp
    atom.cpp
)

if(BUILD_WITH_MPI)
    set(simulationBox_source_files
        ${simulationBox_source_files}
        simulationBox_MPI.cpp
    )
endif()

add_library(simulationBox
    ${simulationBox_source_files}
)

target_include_directories(simulationBox
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/simulationBox
    ${PROJECT_SOURCE_DIR}/include/config
    ${PROJECT_SOURCE_DIR}/include/views
    PRIVATE
    ${PROJECT_SOURCE_DIR}/include/utilities
)

target_link_libraries(simulationBox
    PUBLIC
    linearAlgebra
    box
    exceptions
    settings
    timings
    randomNumberGenerator
)

install(TARGETS simulationBox
    DESTINATION lib
)
