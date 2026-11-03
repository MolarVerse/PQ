add_library(optimizer
    optimizer.cpp

    steepestDescent.cpp
    adam.cpp
)

target_include_directories(optimizer
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/opt/optimizer
    ${PROJECT_SOURCE_DIR}/include/opt/convergence
    ${PROJECT_SOURCE_DIR}/include/concepts
)

target_link_libraries(optimizer
    PUBLIC
    convergence
    config_headers

    PRIVATE
    linearAlgebra
    PUBLIC
    molsys
    physicalData
)

install(TARGETS optimizer
    DESTINATION lib
)
