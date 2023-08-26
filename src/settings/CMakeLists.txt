add_library(settings
    settings.cpp
    simulationBoxSettings.cpp
    manostatSettings.cpp
    thermostatSettings.cpp
)

target_include_directories(settings
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/settings
)

target_link_libraries(settings
    PUBLIC
    config
    timings
)

install(TARGETS settings
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)