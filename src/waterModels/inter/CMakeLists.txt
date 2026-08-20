add_library(interWater
    interWater.cpp
    interWaterBruteForce.cpp
    interWaterCellList.cpp
)

target_include_directories(interWater
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/waterModels/inter
    ${PROJECT_SOURCE_DIR}/include/potential
)

target_link_libraries(interWater
    PUBLIC
    coulombPotential
    nonCoulombPotential
    physicalData
    settings
    molsys
)

install(TARGETS interWater
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)


