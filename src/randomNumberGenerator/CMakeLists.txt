add_library(randomNumberGenerator
    randomNumberGenerator.cpp
)

target_include_directories(randomNumberGenerator
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/randomNumberGenerator
)

target_link_libraries(randomNumberGenerator
    PUBLIC
    simulationBox
    thermostat
    manostat
    maxwellBoltzmann
)

install(TARGETS randomNumberGenerator
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)