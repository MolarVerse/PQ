add_library(hybridConfigurator
    hybridConfigurator.cpp
)

target_include_directories(hybridConfigurator
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/hybridConfigurator
)

target_link_libraries(hybridConfigurator
    PUBLIC
    simulationBox
)

install(TARGETS hybridConfigurator
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)