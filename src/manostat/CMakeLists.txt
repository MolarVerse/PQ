add_library(manostat
    manostat.cpp
)

target_include_directories(manostat
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include/manostat
)

target_link_libraries(manostat
    PUBLIC
    config
    virial
    physicalData
)

install(TARGETS manostat
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)