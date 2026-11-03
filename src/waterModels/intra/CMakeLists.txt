add_library(intraWater
    mTRIntraWater.cpp
)

target_include_directories(intraWater
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/waterModels/intra
)

target_link_libraries(intraWater
    PRIVATE
    coulombPotential
    nonCoulombPotential
    physicalData
    settings
    molsys
    config_headers
)

install(TARGETS intraWater
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)


