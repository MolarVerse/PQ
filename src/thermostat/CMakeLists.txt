add_library(thermostat
    thermostat.cpp
    berendsenThermostat.cpp
    velocityRescalingThermostat.cpp
    langevinThermostat.cpp
    noseHooverThermostat.cpp
)

target_include_directories(thermostat
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include/thermostat
)

target_link_libraries(thermostat
    PUBLIC
    simulationBox
    physicalData
    settings
)

install(TARGETS thermostat
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)