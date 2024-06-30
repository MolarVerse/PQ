add_library(constraints
    constraints.cpp
    bondConstraint.cpp
    distanceConstraint.cpp
    mShakeReference.cpp
    mShake.cpp
)

target_include_directories(constraints
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/constraints
)

target_link_libraries(constraints
    PUBLIC
    physicalData

    PRIVATE
    connectivity
    simulationBox
    linearAlgebra
    settings
    timings
    kernels

    Eigen3::Eigen
)

install(TARGETS constraints
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)
