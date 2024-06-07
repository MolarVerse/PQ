add_library(kernels
    distanceKernels.cpp
)

target_include_directories(kernels
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/kernels
    ${PROJECT_SOURCE_DIR}/include/linearAlgebra
)

target_link_libraries(kernels
    PUBLIC
    lienarAlgebra
    simulationBox
)

install(TARGETS kernels
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)