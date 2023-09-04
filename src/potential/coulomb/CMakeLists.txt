add_library(coulombPotential
    coulombPotential.cpp
    coulombShiftedPotential.cpp
    coulombWolf.cpp
)

target_include_directories(coulombPotential
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include/potential/coulomb
    ${PROJECT_SOURCE_DIR}/include/config
)

install(TARGETS coulombPotential
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)