set(resetKinetics_sources
    resetKinetics.cpp
)

add_library(resetKinetics
    ${resetKinetics_sources}
)

target_include_directories(resetKinetics
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include/resetKinetics
)

target_link_libraries(resetKinetics
    PUBLIC
    simulationBox
    physicalData
    settings
    timings
)

install(TARGETS resetKinetics
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)