add_library(virial
    virial.cpp
)

target_include_directories(virial
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include/virial
)

target_link_libraries(virial
    PUBLIC
    simulationBox
    physicalData
    timings
)

install(TARGETS virial
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)