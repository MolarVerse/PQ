add_library(settings
    generalSettings.cpp
    simulationBoxSettings.cpp
    manostatSettings.cpp
    thermostatSettings.cpp
    constraintSettings.cpp
    potentialSettings.cpp
    fileSettings.cpp
    hybridSettings.cpp
    timingsSettings.cpp
    forceFieldSettings.cpp
    qmSettings.cpp
    ringPolymerSettings.cpp
    outputFileSettings.cpp
    optimizerSettings.cpp
    convergenceSettings.cpp
    waterModelSettings.cpp
    hessianSettings.cpp

    settings.cpp
    resetKineticsSettings.cpp
)

target_include_directories(settings
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/settings
)

target_link_libraries(settings
    PUBLIC
    utilities
    config_headers
)

install(TARGETS settings
    DESTINATION lib
)

if(BUILD_SHARED_LIBS)
    set(BUILD_STATIC "OFF")
else()
    set(BUILD_STATIC "ON")
endif()

target_compile_definitions(settings
    PUBLIC
    _SCRIPT_PATH_=\"${PROJECT_BINARY_DIR}/src/QM/scripts/\"
    _SINGULARITY_=\"${BUILD_WITH_SINGULARITY}\"
    _STATIC_BUILD_=\"${BUILD_STATIC}\"
)
