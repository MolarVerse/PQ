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
        simulationBox_parallel.cpp
    )
endif()

if(BUILD_WITH_CUDA)
    set(simulationBox_source_files
        ${simulationBox_source_files}
        simulationBox_parallel.cpp
        simulationBox_cuda.cu
    )
endif()

add_library(simulationBox
    ${simulationBox_source_files}
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
    timings
)

install(TARGETS simulationBox
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)

if(BUILD_WITH_CUDA)
    target_include_directories(simulationBox
        PUBLIC
        ${CUDA_INCLUDE_DIRS}
    )
endif()