add_library(maxwellBoltzmann
    maxwellBoltzmann.cpp
)

target_include_directories(maxwellBoltzmann
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/maxwellBoltzmann
)

target_link_libraries(maxwellBoltzmann
    PUBLIC
    simulationBox
    settings
    resetKinetics
)

install(TARGETS maxwellBoltzmann
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)