add_library(coulombPotential
    coulombPotential.cpp
    coulombShiftedPotential.cpp
    coulombWolf.cpp
)

target_include_directories(coulombPotential
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/potential/coulomb
    ${PROJECT_SOURCE_DIR}/include/config
)

target_link_libraries(coulombPotential
    PUBLIC
    linearAlgebra
)

install(TARGETS coulombPotential
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)