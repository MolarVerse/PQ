add_library(potential_new
    potential_new.cpp
    nonCoulombPair.cpp
    coulombPotential.cpp
    nonCoulombPotential.cpp
)

target_include_directories(potential_new
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include/potential_new
)

target_link_libraries(potential_new
    PUBLIC
    physicalData
    simulationBox
)

install(TARGETS potential_new
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)