set(potential_source_files
    potential.cpp
    potentialBruteForce.cpp
    potentialCellList.cpp
)

if(BUILD_WITH_CUDA)
    set(potential_source_files
        ${potential_source_files}
        potential_cuda.cu
    )
endif()

add_library(potential
    ${potential_source_files}
)

target_include_directories(potential
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/potential
)

target_link_libraries(potential
    PUBLIC
    physicalData
    simulationBox
    coulombPotential
    nonCoulombPotential
    timings
)

if (BUILD_WITH_CUDA)
    set_target_properties(potential 
        PROPERTIES 
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    
    target_link_libraries(potential
        PUBLIC
        coulombPotential_cuda
        nonCoulombPotential_cuda
        CUDA::cudart
    )
endif()

install(TARGETS potential
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)

add_subdirectory(coulomb)
add_subdirectory(nonCoulomb)

if(BUILD_WITH_CUDA)
    add_subdirectory(nonCoulomb_cuda)
    add_subdirectory(coulomb_cuda)
endif()