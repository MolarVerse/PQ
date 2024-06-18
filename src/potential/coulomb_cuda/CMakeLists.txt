add_library(coulombPotential_cuda
    coulombWolf_cuda.cu
)

set_target_properties(coulombPotential_cuda
    PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS OFF
)

target_include_directories(coulombPotential_cuda
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/potential/coulomb_cuda
    ${PROJECT_SOURCE_DIR}/include/potential/coulomb
)

install(TARGETS coulombPotential_cuda
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)