add_library(potential_new
    potential.cpp
    nonCoulombicPair.cpp

    guffLJ.cpp
    guffBuckingham.cpp
    guffNonCoulomb.cpp

    guffCoulomb.cpp
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