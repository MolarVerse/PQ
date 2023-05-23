add_library(thermostat
    thermostat.cpp
)

target_include_directories(thermostat
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include/thermostat
)

target_link_libraries(thermostat
    PUBLIC
    config
    simulationBox
    output
)