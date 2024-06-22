add_library(convergence
    convergence.cpp
)

target_include_directories(convergence
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/opt/convergence
    ${PROJECT_SOURCE_DIR}/include/config
    ${PROJECT_SOURCE_DIR}/include/concepts
)

target_link_libraries(convergence
    PRIVATE
    linearAlgebra
    simulationBox
    physicalData
)

install(TARGETS convergence
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)