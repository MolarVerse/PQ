add_library(maxwellBoltzmann
    maxwellBoltzmann.cpp
)

target_include_directories(maxwellBoltzmann
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/maxwellBoltzmann
)

if(BUILD_WITH_MPI)
    target_link_libraries(maxwellBoltzmann
        PRIVATE
        MPI::MPI_CXX
        pimd_qmcf_mpi
    )
endif()

target_link_libraries(maxwellBoltzmann
    PUBLIC
    simulationBox
    settings
    resetKinetics
)

install(TARGETS maxwellBoltzmann
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)