add_library(settings
    settings.cpp
    simulationBoxSettings.cpp
    manostatSettings.cpp
    thermostatSettings.cpp
    constraintSettings.cpp
    potentialSettings.cpp
    fileSettings.cpp
    resetKineticsSettings.cpp
    timingsSettings.cpp
    forceFieldSettings.cpp
    qmSettings.cpp
    ringPolymerSettings.cpp
    outputFileSettings.cpp
    optimizerSettings.cpp
)

target_include_directories(settings
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/settings
    ${PROJECT_SOURCE_DIR}/include/config
)

target_link_libraries(settings
    PUBLIC
    utilities
)

install(TARGETS settings
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)