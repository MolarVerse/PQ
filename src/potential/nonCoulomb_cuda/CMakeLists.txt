add_library(nonCoulombPotential_cuda
    lennardJones_cuda.cu
)

target_include_directories(nonCoulombPotential_cuda
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/potential/nonCoulomb_cuda
    ${PROJECT_SOURCE_DIR}/include/potential/nonCoulomb
    ${CUDA_INCLUDE_DIRS}
)

target_link_libraries(nonCoulombPotential_cuda
    PUBLIC
    linearAlgebra
)

install(TARGETS nonCoulombPotential_cuda
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)