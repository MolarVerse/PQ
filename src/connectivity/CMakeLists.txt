add_library(connectivity
    bond.cpp
    angle.cpp
    dihedral.cpp
    connectivityElement.cpp
)

target_include_directories(connectivity
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/connectivity
)

target_link_libraries(connectivity
    PRIVATE
    simulationBox
)

install(TARGETS connectivity
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)