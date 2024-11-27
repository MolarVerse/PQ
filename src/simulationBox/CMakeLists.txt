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

if(BUILD_WITH_KOKKOS)
    set(simulationBox_source_files
        ${simulationBox_source_files}
        simulationBox_kokkos.cpp
    )
endif()

if(BUILD_WITH_GPU)
    set(simulationBox_source_files
        ${simulationBox_source_files}
        simulationBox_GPU.cpp
    )
endif()

add_library(simulationBox
    ${simulationBox_source_files}
)

if(BUILD_WITH_KOKKOS)
    target_link_libraries(simulationBox
        PUBLIC
        Kokkos::kokkos

        # Kokkos::kokkoskernels
    )

    target_include_directories(simulationBox
        PUBLIC
        ${Kokkos_INCLUDE_DIRS_RET}
    )
endif()

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

install(TARGETS simulationBox
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)