add_library(integrator
    integrator.cpp
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
)

install(TARGETS integrator
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)