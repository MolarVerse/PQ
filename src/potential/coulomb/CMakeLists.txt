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

install(TARGETS coulombPotential
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)