add_library(resetKinetics
    resetKinetics.cpp
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
)

install(TARGETS resetKinetics
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)