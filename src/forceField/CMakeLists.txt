add_library(forceField
    bondForceField.cpp
    angleForceField.cpp
    dihedralForceField.cpp
    forceField.cpp
)

target_include_directories(forceField
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/forceField
)

target_link_libraries(forceField
    PUBLIC
    simulationBox
    connectivity
    vector3d
)

install(TARGETS forceField
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)