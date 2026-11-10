add_library(kinetics
    resetKinetics.cpp
)

target_include_directories(kinetics
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include/kinetics
)

target_link_libraries(kinetics
    PUBLIC
    molsys
    physicalData
    settings
    timings
)

install(TARGETS kinetics
    DESTINATION lib
)
