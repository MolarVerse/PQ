set(integrator_source_files
    integrator.cpp
    velocityVerlet.cpp
)

add_library(integrator
    ${integrator_source_files}
)

target_include_directories(integrator
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include/integrator
)

target_link_libraries(integrator
    PUBLIC
    simulationBox
    settings
    timings
)

install(TARGETS integrator
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)