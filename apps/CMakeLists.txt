set(executables pimd_qmcf_mpi)

foreach(executable ${executables})
  find_package(MPI REQUIRED)

  add_executable(${executable}
    ${executable}.cpp
  )

  target_include_directories(${executable}
    PUBLIC ${CODE_INCLUDE_DIR}
  )

  target_link_libraries(${executable}
    PUBLIC
    setup
    simulationBox
    engine
    utilities
    config
    output
    MPI::MPI_CXX
  )

  install(TARGETS ${executable}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/bin
  )
endforeach()
