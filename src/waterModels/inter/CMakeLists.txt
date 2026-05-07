add_library(interWater
    interWater.cpp
    interWaterBruteForce.cpp
    interWaterCellList.cpp
)

target_include_directories(interWater
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/waterModels/inter
)

target_link_libraries(interWater
    PUBLIC
    coulombPotential
    nonCoulombPotential
    physicalData
    settings
    simulationBox
)

install(TARGETS interWater
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)


