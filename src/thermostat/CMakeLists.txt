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
        berendsen_legacy.cpp
        bussi_legacy.cpp
        langevin_legacy.cpp
        noseHoover_legacy.cpp
    )
else()
    set(thermostat_source_files
        ${thermostat_source_files}
        berendsen_nonLegacy.cpp
        bussi_nonLegacy.cpp
        langevin_nonLegacy.cpp
        noseHoover_nonLegacy.cpp
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