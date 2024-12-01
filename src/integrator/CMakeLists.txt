set(integrator_source_files
    integrator.cpp
)

if(BUILD_LEGACY)
    set(integrator_source_files
        ${integrator_source_files}
        velocityVerlet_legacy.cpp
    )
else()
    set(integrator_source_files
        ${integrator_source_files}
        velocityVerlet_nonLegacy.cpp
    )
endif()

if(BUILD_WITH_KOKKOS)
    set(integrator_source_files
        ${integrator_source_files}
        integrator_kokkos.cpp
    )
endif()

add_library(integrator
    ${integrator_source_files}
)

target_include_directories(integrator
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include/integrator
)

if(BUILD_WITH_KOKKOS)
    target_include_directories(integrator
        PUBLIC
        ${Kokkos_INCLUDE_DIRS_RETS}
    )
endif()

target_link_libraries(integrator
    PUBLIC
    simulationBox
    settings
    timings
)

if(BUILD_WITH_KOKKOS)
    target_link_libraries(integrator
        PUBLIC
        Kokkos::kokkos
    )
endif()

install(TARGETS integrator
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)