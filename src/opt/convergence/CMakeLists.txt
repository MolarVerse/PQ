add_library(convergence
    convergence.cpp
)

target_include_directories(convergence
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/opt/convergence
    ${PROJECT_SOURCE_DIR}/include/concepts
)

target_link_libraries(convergence
    PUBLIC
    config_headers
    PRIVATE
    linearAlgebra
    molsys
    physicalData
)

install(TARGETS convergence
    DESTINATION lib
)
