add_library(constraints
    constraints.cpp
    bondConstraint.cpp
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
)

install(TARGETS constraints
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)