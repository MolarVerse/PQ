add_library(coulombPotential_cuda
    coulombWolf_cuda.cu
)

target_include_directories(coulombPotential_cuda
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/potential/coulomb_cuda
    ${PROJECT_SOURCE_DIR}/include/potential/coulomb
    ${CUDA_INCLUDE_DIRS}
)

install(TARGETS coulombPotential_cuda
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)