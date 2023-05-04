add_executable(pimd_qmcf
  pimd_qmcf.cpp
)

target_include_directories(pimd_qmcf
  PUBLIC ${CODE_INCLUDE_DIR}
)

target_link_libraries(pimd_qmcf
    PUBLIC
        setup
        simulationBox
        settings
        utilities
        config
        output
)

install(TARGETS pimd_qmcf
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/bin
)