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
    Eigen3::Eigen

    PRIVATE
    connectivity
    simulationBox
    linearAlgebra
    settings
    timings
    kernels

)

install(TARGETS constraints
    DESTINATION lib
)
