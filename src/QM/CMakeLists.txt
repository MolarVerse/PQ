add_library(QM
    qmRunner.cpp
)

target_include_directories(QM
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/QM
    ${PROJECT_SOURCE_DIR}/include/config
)

target_link_libraries(QM
    PUBLIC
    simulationBox
    settings
    utilities
    physicalData
    exceptions
)

if(BUILD_SHARED_LIBS)
    set(BUILD_STATIC "OFF")
else()
    set(BUILD_STATIC "ON")
endif()

install(TARGETS QM
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)

add_subdirectory(external)

if(BUILD_WITH_ASE)
    add_subdirectory(ase)
endif()