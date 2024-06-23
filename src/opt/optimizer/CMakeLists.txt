add_library(optimizer
    optimizer.cpp

    steepestDescent.cpp
)

target_include_directories(optimizer
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/opt/optimizer
    ${PROJECT_SOURCE_DIR}/include/opt/convergence
    ${PROJECT_SOURCE_DIR}/include/config
    ${PROJECT_SOURCE_DIR}/include/concepts
)

target_link_libraries(optimizer
    PRIVATE
    linearAlgebra
    simulationBox
    physicalData
)

install(TARGETS optimizer
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)