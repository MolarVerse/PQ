add_library(nonCoulombPotential_cuda
    STATIC
    lennardJones_cuda.cu
)

set_target_properties(nonCoulombPotential_cuda
    PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS OFF
)

target_include_directories(nonCoulombPotential_cuda
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/potential/nonCoulomb_cuda
    ${PROJECT_SOURCE_DIR}/include/potential/nonCoulomb
)

target_link_libraries(nonCoulombPotential_cuda
    PUBLIC
    linearAlgebra
)

install(TARGETS nonCoulombPotential_cuda
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)