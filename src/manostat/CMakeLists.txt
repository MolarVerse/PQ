add_library(manostat
    manostat.cpp

    berendsenManostat.cpp
    monteCarloManostat.cpp
)

target_include_directories(manostat
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/manostat
)

target_link_libraries(manostat
    PUBLIC
    virial
    physicalData
    linearAlgebra
    settings
)

install(TARGETS manostat
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)