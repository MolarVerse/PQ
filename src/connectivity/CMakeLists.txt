add_library(connectivity
    bond.cpp
    angle.cpp
    dihedral.cpp
    connectivityElement.cpp
)

target_include_directories(connectivity
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include/connectivity
)

target_link_libraries(connectivity
    PUBLIC
    simulationBox
    config
)

install(TARGETS connectivity
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)