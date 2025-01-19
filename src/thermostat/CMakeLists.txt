set(thermostat_source_files
    thermostat.cpp
    berendsenThermostat.cpp
    velocityRescalingThermostat.cpp
    langevinThermostat.cpp
    noseHooverThermostat.cpp
)

if(BUILD_LEGACY)
    set(thermostat_source_files
        ${thermostat_source_files}
        langevin_legacy.cpp
    )
else()
    set(thermostat_source_files
        ${thermostat_source_files}
        langevin_nonLegacy.cpp
    )
endif()

add_library(thermostat
    ${thermostat_source_files}
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
    timings
)

install(TARGETS thermostat
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)