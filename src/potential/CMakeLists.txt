add_library(potential
    potential.cpp
)

target_include_directories(potential
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include/potential
    ${PROJECT_SOURCE_DIR}/include/potential/coulomb
)

target_link_libraries(potential
    PUBLIC
    physicalData
    simulationBox
    coulombPotential
    nonCoulombPotential
)

install(TARGETS potential
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)

add_subdirectory(coulomb)
add_subdirectory(nonCoulomb)