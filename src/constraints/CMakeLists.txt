add_library(constraints
    constraints.cpp
    bondConstraint.cpp
    distanceConstraint.cpp
    mShakeReference.cpp
)

target_include_directories(constraints
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/constraints
)

target_link_libraries(constraints
    PUBLIC
    connectivity
    simulationBox
    linearAlgebra
    settings
    timings
    physicalData

    PRIVATE
    Eigen3::Eigen
)

install(TARGETS constraints
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)
