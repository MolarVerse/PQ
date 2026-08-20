set(molsys_source_files
    simulationBox.cpp
    simulationBox_standardMethods.cpp

    molecule.cpp
    moleculeType.cpp
    celllist.cpp
    cell.cpp
    atom.cpp
)

if(BUILD_WITH_MPI)
    set(molsys_source_files
        ${molsys_source_files}
        simulationBox_MPI.cpp
    )
endif()

add_library(molsys
    ${molsys_source_files}
)

target_include_directories(molsys
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/molsys
    ${PROJECT_SOURCE_DIR}/include/config
    ${PROJECT_SOURCE_DIR}/include/views
    PRIVATE
    ${PROJECT_SOURCE_DIR}/include/utilities
)

target_link_libraries(molsys
    PUBLIC
    linearAlgebra
    box
    exceptions
    settings
    timings
    randomNumberGenerator
)

install(TARGETS molsys
    DESTINATION lib
)
