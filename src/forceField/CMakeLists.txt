add_library(forceField
    bondForceField.cpp
    bondType.cpp
    angleForceField.cpp
    angleType.cpp
    dihedralForceField.cpp
    dihedralType.cpp
    forceField.cpp
    nonCoulombicPair.cpp
)

target_include_directories(forceField
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/forceField
)

target_link_libraries(forceField
    PUBLIC
    simulationBox
    connectivity
    linearAlgebra
    utilities
)

install(TARGETS forceField
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)